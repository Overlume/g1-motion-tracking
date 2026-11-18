"""量化评估 v2：分段感知 + 漂移速率。

与 v1 的差别（v1 的 bug）：
- v1 只在 metrics.csv 里记了 done，却从不处理；环境 termination 后会静默 reset 并把 motion 相位归零，
  于是 track_completion 用 step 数除以总帧数会得 1.0（假的），也无法知道"动作播放了几次"。
- v2 记录每个 reset 的步号与终止项，按 episode 分段统计，用真实到达的最远 motion 帧号算完成度。

指标：
- MPKPE (cm)：14 个 body 的世界坐标平均位置误差，含全局漂移。
- R-MPKPE (cm)：把参考重新锚定到机器人根位姿后计算的姿态误差（去漂移）。
- root xy err (cm)：pelvis 世界 XY 与参考 pelvis 的水平距离，漂移的直接度量。
- drift rate (cm/s)：最长不间断 episode 内 root xy err 的线性拟合斜率。
- 脚滑 (cm)：参考脚处于支撑相（脚高 < 8 cm、水平速度 < 1 cm/步、机器人脚倾角 < 15 度）时，
  累加机器人该脚的水平位移。
- 跌倒：pelvis 高度 < 0.15 m 连续 5 步。

用法：
  MUJOCO_GL=egl environment/.venv/bin/python scripts/eval_tracking.py --motion-file data/processed/fight1_s3_000s_20s.npz --out runs/xxxx
  ... --ignore-terminations   # 关闭 termination，观察连续行为（不会有 reset）
"""
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp.metrics import (
  compute_joint_velocity_error,
  compute_mpkpe,
  compute_root_relative_mpkpe,
)
from mjlab.utils.torch import configure_torch_backends

TASK = "Mjlab-Tracking-Flat-Unitree-G1"
FALL_HEIGHT = 0.15
FALL_STEPS = 5
FOOT_HEIGHT = 0.08
REF_PLANT_SPEED = 0.01


def tilt_deg(q) -> float:
  w, qx, qy, qz = [float(v) for v in q]
  r33 = 1.0 - 2.0 * (qx * qx + qy * qy)
  return float(np.degrees(np.arccos(np.clip(r33, -1.0, 1.0))))


def yaw_deg(q) -> float:
  w, qx, qy, qz = [float(v) for v in q]
  return float(np.degrees(np.arctan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))))


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint-file", default="scratch/model_49999.pt")
  ap.add_argument("--motion-file", required=True)
  ap.add_argument("--max-steps", type=int, default=0, help="0 = full motion")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--ignore-terminations", action="store_true")
  ap.add_argument("--keep-dr", action="store_true",
                  help="保留 play 模式的 startup 域随机化（默认关闭，保证同一 seed 可复现）")
  ap.add_argument("--out", required=True)
  # v2.1: 可选受控外力（默认全空 = 与 v2 完全相同）
  ap.add_argument("--perturb-body", default="", help="施加外力的 body 名（如 torso_link）；空=不施加")
  ap.add_argument("--perturb-force", default="", help="世界系力向量 N，如 60,0,0")
  ap.add_argument("--perturb-start-s", type=float, default=8.0)
  ap.add_argument("--perturb-duration-s", type=float, default=0.2)
  args = ap.parse_args()

  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  cfg = load_env_cfg(TASK, play=True)
  agent_cfg = load_rl_cfg(TASK)
  cfg.commands["motion"].motion_file = args.motion_file
  cfg.scene.num_envs = 1
  cfg.seed = args.seed
  if not args.keep_dr:
    for _name in ("base_com", "encoder_bias", "foot_friction"):
      cfg.events.pop(_name, None)
  print("seed:", args.seed, "startup DR events:", sorted(cfg.events.keys()))

  base = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
  tm = base.termination_manager
  if args.ignore_terminations:
    tm.compute = lambda: torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    print("!! terminations disabled")
  env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)

  ckpt = Path(args.checkpoint_file).resolve()
  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  term = base.command_manager.get_term("motion")
  body_names = list(term.cfg.body_names)
  i_pelvis = body_names.index("pelvis")
  feet = [body_names.index("left_ankle_roll_link"), body_names.index("right_ankle_roll_link")]
  n_frames = int(term.motion.joint_pos.shape[0])
  max_steps = args.max_steps if args.max_steps > 0 else n_frames
  print("motion frames:", n_frames, "-> steps:", max_steps)

  perturb = None
  robot = None
  if args.perturb_body:
    robot = base.scene["robot"]
    ids, names = robot.find_bodies(args.perturb_body)
    if not ids:
      raise SystemExit(f"--perturb-body {args.perturb_body} 未匹配；可选: {', '.join(robot.body_names)}")
    vec = [float(x) for x in args.perturb_force.split(",")] if args.perturb_force else [0.0, 0.0, 0.0]
    if len(vec) != 3:
      raise SystemExit("--perturb-force 需要 3 个数，如 60,0,0")
    f = torch.tensor(vec, dtype=torch.float32, device=base.device)
    f = f.reshape(1, 1, 3).repeat(base.num_envs, len(ids), 1)
    perturb = dict(ids=list(ids), names=list(names), force=f, zero=torch.zeros_like(f),
                   start=args.perturb_start_s, end=args.perturb_start_s + args.perturb_duration_s)
    print(f"perturb: bodies={names} force={vec} N window=[{perturb['start']}, {perturb['end']}) s")

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  rows = []
  obs = env.get_observations()
  low_run = 0
  failed_at = None
  prev_rf = None
  prev_ref = None
  resets = []

  with torch.no_grad():
    for step in range(max_steps):
      if perturb is not None:
        t_now = step * 0.02
        active = perturb["start"] <= t_now < perturb["end"]
        robot.write_external_wrench_to_sim(
          forces=perturb["force"] if active else perturb["zero"],
          torques=perturb["zero"],
          body_ids=perturb["ids"],
        )
      obs, _, dones, _ = env.step(policy(obs))
      done = bool(dones[0].item())
      fired = [n for n in tm.active_terms if bool(tm.get_term(n)[0].item())] if done else []

      mpkpe = compute_mpkpe(term).item()
      rmpkpe = compute_root_relative_mpkpe(term).item()
      jpos_err = term.joint_pos - term.robot_joint_pos
      jrmse = torch.sqrt(torch.mean(jpos_err ** 2)).item()
      jvel = compute_joint_velocity_error(term).item()

      robot_pos = term.robot_body_pos_w[0]
      ref_pos = term.body_pos_w[0]
      rp = robot_pos[i_pelvis]
      fp = ref_pos[i_pelvis]
      pelvis_z = rp[2].item()
      ref_pelvis_z = fp[2].item()
      root_err_cm = float(torch.linalg.norm(rp[:2] - fp[:2]).item()) * 100.0
      tr = tilt_deg(term.robot_body_quat_w[0, i_pelvis])
      tf = tilt_deg(term.body_quat_w[0, i_pelvis])
      ye = yaw_deg(term.robot_body_quat_w[0, i_pelvis]) - yaw_deg(term.body_quat_w[0, i_pelvis])

      rf = robot_pos[feet].cpu().numpy()
      ref = ref_pos[feet].cpu().numpy()
      rqf = term.robot_body_quat_w[0][feet]
      flat = [tilt_deg(rqf[k]) < 15.0 for k in range(2)]
      slip = [0.0, 0.0]
      if prev_rf is not None and not done:
        for k in range(2):
          ref_speed = float(np.linalg.norm(ref[k, :2] - prev_ref[k, :2]))
          if ref[k, 2] < FOOT_HEIGHT and ref_speed < REF_PLANT_SPEED and flat[k]:
            slip[k] = float(np.linalg.norm(rf[k, :2] - prev_rf[k, :2]) * 100.0)
      prev_rf, prev_ref = rf, ref

      low_run = low_run + 1 if pelvis_z < FALL_HEIGHT else 0
      fall = low_run >= FALL_STEPS
      rows.append(
        dict(
          step=step,
          time_s=round((step + 1) / 50.0, 4),
          motion_frame=int(term.time_steps[0].item()),
          done=done,
          fired=",".join(fired),
          mpkpe_cm=round(mpkpe * 100, 4),
          r_mpkpe_cm=round(rmpkpe * 100, 4),
          root_err_cm=round(root_err_cm, 4),
          yaw_err_deg=round(ye, 4),
          joint_rmse_rad=round(jrmse, 6),
          joint_vel_rmse=round(jvel, 6),
          pelvis_z=round(pelvis_z, 6),
          ref_pelvis_z=round(ref_pelvis_z, 6),
          tilt_deg=round(tr, 4),
          ref_tilt_deg=round(tf, 4),
          slip_left_cm=round(slip[0], 4),
          slip_right_cm=round(slip[1], 4),
        )
      )
      if done:
        resets.append(dict(step=step, time_s=round((step + 1) / 50.0, 3), terms=fired))
        print("RESET step=%d t=%.2fs terms=%s" % (step, (step + 1) / 50.0, ",".join(fired)))
      if fall:
        failed_at = step
        print("FALL at step", step, "pelvis_z:", round(pelvis_z, 4))
        break

  with open(out / "metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

  a = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0] if k not in ("fired",)}
  # 分段：以 reset 为界（reset 那一步的样本是 reset 后的新 episode 第 0 帧附近）
  bounds = [r["step"] for r in resets]
  segs = []
  start = 0
  for b in bounds:
    segs.append((start, b))
    start = b + 1
  segs.append((start, len(rows) - 1))
  segs = [(s, e) for s, e in segs if e - s >= 10]
  seg_stats = []
  for s, e in segs:
    seg_rows = rows[s:e + 1]
    err = np.array([r["root_err_cm"] for r in seg_rows])
    t = np.arange(len(seg_rows)) / 50.0
    slope = float(np.polyfit(t, err, 1)[0]) if len(seg_rows) > 5 else float("nan")
    seg_stats.append(dict(start_step=s, end_step=e, seconds=round((e - s + 1) / 50.0, 2),
                          root_err_mean_cm=round(float(err.mean()), 2),
                          root_err_last_cm=round(float(err[-1]), 2),
                          drift_rate_cm_s=round(slope, 2)))
  longest = max(seg_stats, key=lambda x: x["seconds"]) if seg_stats else None

  perturb_stats = None
  if perturb is not None:
    def _win(lo, hi):
      v = [r["root_err_cm"] for r in rows if lo <= r["step"] * 0.02 < hi]
      return round(float(np.mean(v)), 3) if v else None
    after = [r for r in resets if r["step"] * 0.02 >= perturb["start"]]
    perturb_stats = dict(
      body=perturb["names"], force_n=[round(float(x), 1) for x in perturb["force"][0, 0].tolist()],
      start_s=perturb["start"], duration_s=round(perturb["end"] - perturb["start"], 3),
      root_err_pre_1s_cm=_win(perturb["start"] - 1.0, perturb["start"]),
      root_err_during_cm=_win(perturb["start"], perturb["end"]),
      root_err_post_0_1s_cm=_win(perturb["end"], perturb["end"] + 1.0),
      root_err_post_1_2s_cm=_win(perturb["end"] + 1.0, perturb["end"] + 2.0),
      resets_after=len(after),
      first_reset_after_s=round(after[0]["step"] * 0.02, 2) if after else None,
    )

  furthest = int(max(r["motion_frame"] for r in rows))
  top = [r for r in resets]
  summary = dict(
    motion_file=str(args.motion_file),
    checkpoint=str(ckpt),
    ignore_terminations=bool(args.ignore_terminations),
    frames_total=n_frames,
    steps_run=len(rows),
    seconds_run=round(len(rows) / 50.0, 3),
    reset_count=len(resets),
    reset_steps=[r["step"] for r in resets],
    reset_terms=[r["terms"] for r in resets],
    motion_frame_furthest=furthest,
    track_completion=round(min(1.0, furthest / n_frames), 4),
    seed=args.seed,
    keep_startup_dr=bool(args.keep_dr),
    perturb=perturb_stats,
    fall_detected=failed_at is not None,
    fall_step=failed_at,
    segments=seg_stats,
    longest_segment=longest,
    mpkpe_mean_cm=round(float(a["mpkpe_cm"].mean()), 3),
    mpkpe_p95_cm=round(float(np.percentile(a["mpkpe_cm"], 95)), 3),
    mpkpe_max_cm=round(float(a["mpkpe_cm"].max()), 3),
    r_mpkpe_mean_cm=round(float(a["r_mpkpe_cm"].mean()), 3),
    r_mpkpe_p95_cm=round(float(np.percentile(a["r_mpkpe_cm"], 95)), 3),
    root_err_mean_cm=round(float(a["root_err_cm"].mean()), 3),
    root_err_p95_cm=round(float(np.percentile(a["root_err_cm"], 95)), 3),
    yaw_err_absmean_deg=round(float(np.abs(a["yaw_err_deg"]).mean()), 3),
    joint_rmse_mean_rad=round(float(a["joint_rmse_rad"].mean()), 4),
    joint_rmse_p95_rad=round(float(np.percentile(a["joint_rmse_rad"], 95)), 4),
    joint_vel_rmse_mean=round(float(a["joint_vel_rmse"].mean()), 4),
    pelvis_z_min=round(float(a["pelvis_z"].min()), 4),
    pelvis_z_mean=round(float(a["pelvis_z"].mean()), 4),
    tilt_deg_max=round(float(a["tilt_deg"].max()), 3),
    tilt_deg_p95=round(float(np.percentile(a["tilt_deg"], 95)), 3),
    ref_tilt_deg_max=round(float(a["ref_tilt_deg"].max()), 3),
    tilt_err_mean_deg=round(float(np.abs(a["tilt_deg"] - a["ref_tilt_deg"]).mean()), 3),
    slip_left_total_cm=round(float(a["slip_left_cm"].sum()), 3),
    slip_right_total_cm=round(float(a["slip_right_cm"].sum()), 3),
  )
  (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(json.dumps(summary, indent=2))
  env.close()


if __name__ == "__main__":
  main()
  