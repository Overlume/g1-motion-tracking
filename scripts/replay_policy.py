"""官方 pre-trained checkpoint 回放（无 viewer，直接录 mp4）。

重要：VideoRecorder 的 video_length 是"固定录 N 帧"，**不会**在 episode 结束处截断。
环境在 termination 触发时会自动 reset（动作回到第 0 帧），所以录像里会出现"动作重头再来"。
本脚本逐帧记录 dones 与触发的终止项，写到 <out_dir>/episode_log.json，便于事后核对。

取景（2026-11-14 修复，两次迭代）：
  背景问题 1：mjlab 默认只渲染 env_idx + 最近 max_extra_envs(默认 2) 个邻居
    （repos/mjlab_research/src/mjlab/viewer/offscreen_renderer.py:186-217、
     viewer_config.py:38），所以 --num-envs 4 时画面里只有 3 台机器人。
  背景问题 2：默认相机 origin_type=AUTO 是 TRACKING 模式，跟着 env 0 的 pelvis 走，
    cfg.lookat 被忽略；env 之间漂移散开后装载不下。
  现在的做法：
   - num_envs>1 时把 max_extra_envs 设成 num_envs-1，所有 env 都提交渲染；
   - 默认 --camera follow：每帧把自由相机的 lookat 放在所有机器人根位置的中心，
     距离按当前实际散布自适应（带平滑），低机位（elevation -12、azimuth 45）；
   - --camera fit：静态自由相机，只按 env 网格算一次（机器人散开后可能出画）；
   - --camera auto：mjlab 原来的跟随相机（只看 env 0 附近）。

用法:
  MUJOCO_GL=egl environment/.venv/bin/python scripts/replay_policy.py \
    --checkpoint-file scratch/model_49999.pt --motion-file data/processed/fight1_s3_000s_20s.npz \
    --video-length 999 --num-envs 4 --width 960 --height 540 --tag fight1_s3

  --ignore-terminations 把 terminations 强制为 False，用于观察"不打断时"的真实行为。
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer.viewer_config import ViewerConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_fit import fit_camera, grid_origins  # noqa: E402

TASK = "Mjlab-Tracking-Flat-Unitree-G1"


def install_follow_camera(raw_env, args, init_distance: float, lookat_z: float = 0.8) -> dict:
  """把 env.render 包一层：每帧按机器人实际位置重设自由相机（跟随 + 自适应距离）。"""
  cam = raw_env._offline_renderer._cam
  entity = next(iter(raw_env.scene.entities.values()))
  orig_render = raw_env.render
  state = {"d": float(init_distance), "d_min": 1e9, "d_max": 0.0, "calls": 0}

  def render_follow():
    # 世界系根位置（不能用 sim.data.qpos：每个 world 的 qpos 是相对各自 origin 的）
    pos = entity.data.root_link_pos_w
    xy = pos[:, :2].detach().float().cpu().numpy()
    pts = [(float(p[0]), float(p[1])) for p in xy]
    c = fit_camera(pts, args.width, args.height,
                   margin=args.camera_margin,
                   elevation=args.camera_elevation,
                   azimuth=args.camera_azimuth,
                   min_distance=args.camera_min_distance)
    d = 0.75 * state["d"] + 0.25 * c["distance"]
    state["d"] = d
    state["d_min"] = min(state["d_min"], d)
    state["d_max"] = max(state["d_max"], d)
    state["calls"] += 1
    cam.lookat[:] = c["lookat"]
    cam.distance = d
    cam.elevation = args.camera_elevation
    cam.azimuth = args.camera_azimuth
    return orig_render()

  raw_env.render = render_follow
  return state


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint-file", required=True)
  ap.add_argument("--motion-file", required=True)
  ap.add_argument("--num-envs", type=int, default=1)
  ap.add_argument("--video-length", type=int, default=300)
  ap.add_argument("--width", type=int, default=960)
  ap.add_argument("--height", type=int, default=540)
  ap.add_argument("--out-dir", default="outputs/videos")
  ap.add_argument("--tag", default="", help="subfolder under <ckpt>/<WxH> for this motion")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--keep-dr", action="store_true",
                  help="保留 startup 域随机化（默认关闭，保证同一 seed 可复现）")
  ap.add_argument("--ignore-terminations", action="store_true",
                  help="禁用 terminations（episode 不会中途 reset），用于观察连续行为")
  ap.add_argument("--camera", choices=("follow", "fit", "auto"), default="follow",
                  help="follow=跟着所有机器人的中心并自适应距离（默认）；fit=静态装下网格；auto=mjlab 跟随相机")
  ap.add_argument("--camera-margin", type=float, default=1.3,
                  help="follow/fit 模式的取景余量倍数（默认 1.2）")
  ap.add_argument("--camera-elevation", type=float, default=-16.0)
  ap.add_argument("--camera-azimuth", type=float, default=45.0)
  ap.add_argument("--camera-min-distance", type=float, default=5.2,
                  help="follow 模式的距离下限（默认 5.2，越大机器人越小但越不容易出画）")
  args = ap.parse_args()

  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  env_cfg = load_env_cfg(TASK, play=True)
  agent_cfg = load_rl_cfg(TASK)
  env_cfg.commands["motion"].motion_file = args.motion_file
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.viewer.width = args.width
  env_cfg.viewer.height = args.height
  env_cfg.seed = args.seed
  if not args.keep_dr:
    for _name in ("base_com", "encoder_bias", "foot_friction"):
      env_cfg.events.pop(_name, None)
  print("seed:", args.seed, "startup DR events:", sorted(env_cfg.events.keys()))

  # --- 取景：所有 env 都要渲染；相机按模式设置（必须在建 env 前写好 cfg）---
  cam_log = dict(mode=args.camera)
  if args.num_envs > 1:
    env_cfg.viewer.max_extra_envs = args.num_envs - 1
  origins = grid_origins(args.num_envs, env_cfg.scene.env_spacing)
  base_cam = fit_camera(origins, args.width, args.height,
                        margin=args.camera_margin,
                        elevation=args.camera_elevation,
                        azimuth=args.camera_azimuth)
  if args.camera in ("fit", "follow"):
    env_cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
    env_cfg.viewer.lookat = base_cam["lookat"]
    env_cfg.viewer.distance = base_cam["distance"]
    env_cfg.viewer.elevation = args.camera_elevation
    env_cfg.viewer.azimuth = args.camera_azimuth
    cam_log.update(base_cam)
  cam_log["max_extra_envs"] = int(env_cfg.viewer.max_extra_envs)
  print("camera:", json.dumps(cam_log))
  print("env_origins:", origins)

  d = np.load(args.motion_file)
  n = d["root_pos_w"].shape[0]
  print("motion frames:", n, "at 50 fps ->", round(n / 50.0, 2), "s")

  ckpt = Path(args.checkpoint_file).resolve()
  out_dir = Path(args.out_dir) / ckpt.stem / ("%dx%d" % (args.width, args.height))
  if args.tag:
    out_dir = out_dir / args.tag
  out_dir.mkdir(parents=True, exist_ok=True)

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  follow_state = None
  if args.camera == "follow":
    follow_state = install_follow_camera(raw_env, args, base_cam["distance"])
    print("!! follow camera installed (smoothing 0.75 prev + 0.25 new)")
  tm = raw_env.termination_manager
  if args.ignore_terminations:
    tm.compute = lambda: torch.zeros(raw_env.num_envs, dtype=torch.bool, device=raw_env.device)
    print("!! terminations disabled")

  env = VideoRecorder(
    raw_env,
    video_folder=str(out_dir),
    step_trigger=lambda step: step == 0,
    video_length=args.video_length,
    disable_logger=True,
  )
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  term = raw_env.command_manager.get_term("motion")
  obs = env.get_observations() if hasattr(env, "get_observations") else env.reset()
  resets = []
  steps = 0
  with torch.no_grad():
    while steps <= args.video_length:
      actions = policy(obs)
      obs, _, dones, _ = env.step(actions)
      steps += 1
      dn = dones.cpu().numpy()
      for i in np.nonzero(dn)[0]:
        i = int(i)
        fired = [name for name in tm.active_terms if bool(tm.get_term(name)[i].item())]
        resets.append(dict(env=i, step=steps, time_s=round(steps / 50.0, 2),
                           terms=fired, frame_after_reset=int(term.time_steps[i].item())))
        print("RESET env=%d step=%d t=%.2fs terms=%s -> motion frame %d"
              % (i, steps, steps / 50.0, ",".join(fired), int(term.time_steps[i].item())))
  env.close()

  if follow_state is not None:
    cam_log["follow"] = dict(renders=follow_state["calls"],
                             distance_min=round(follow_state["d_min"], 3),
                             distance_max=round(follow_state["d_max"], 3))

  log = dict(
    motion_file=args.motion_file,
    checkpoint=str(ckpt),
    num_envs=args.num_envs,
    video_length=args.video_length,
    camera=cam_log,
    ignore_terminations=bool(args.ignore_terminations),
    seed=args.seed,
    keep_startup_dr=bool(args.keep_dr),
    motion_frames=int(n),
    steps_recorded=steps,
    reset_count=len(resets),
    resets=resets,
  )
  (out_dir / "episode_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
  print("steps:", steps, "=>", round(steps / 50.0, 2), "s", "| resets:", len(resets))
  for f in sorted(out_dir.rglob("*.mp4")):
    print("video:", f, f.stat().st_size, "bytes")


if __name__ == "__main__":
  main()
