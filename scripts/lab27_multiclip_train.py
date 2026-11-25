#!/usr/bin/env python
"""LAB27：多片段联合训练，其余与 M4 完全一致。

mjlab 的 MotionLoader 支持两类多片段输入（commands.py:_resolve_motion_file / MotionLoader）：
  - list[str] / 逗号 / glob：多个 npz 按 sorted 顺序拼接；
  - packed 目录：<field>.npy + motion_ends.npy + joint_names.json（+ motion_names.json）。
但 scripts/train.py 的启动校验只接受「单个存在的路径字符串」：
  local = motion_cmd.motion_file
  if isinstance(local, str) and local and Path(local).exists(): ...
  else: raise ValueError("provide --env.commands.motion.motion-file /path/to/motion.npz")
所以直接传 list 会在启动阶段被拦下。本脚本的做法：先把片段打包成 packed 目录
（该目录本身就是一个存在的路径），再把它设成 motion_file —— 既能过校验，又保留
每段边界（motion_ends），episode 走到片段末帧就重采样、不跨片段。

片段必须先用 scripts/lab27_ground_fix.py 过地面高度校正并通过 D41 闸
（穿地 max ≤20 mm 且 p99 ≤20 mm）。

用法（项目根目录）：
  # 冒烟（3 iter / 64 envs）
  environment/.venv/bin/python scripts/lab27_multiclip_train.py --run-root scratch/lab27-smoke --mode smoke

  # 正式（3000 iter / 4096 envs，约 1 h 45；中断后再跑同一条命令即续训）
  environment/.venv/bin/python scripts/lab27_multiclip_train.py \
      --run-root runs/20261126-001-lab27-multiclip --mode full --resume
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import mjlab.tasks  # noqa: F401  # 注册任务
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

TASK = "Mjlab-Tracking-Flat-Unitree-G1"
EXPERIMENT = "g1_tracking"
TARGET_ITER = 52998      # M4 / LAB20 / LAB21 / LAB24 的终点迭代
FIELDS = ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel")
DEFAULT_CLIPS = "data/processed/fight1_s3_000s_20s.npz,data/processed/dance1_000s_20s_gfix.npz"


def iter_of(p: Path) -> int:
  try:
    return int(p.stem.split("_")[1])
  except Exception:
    return -1


def latest_ckpt(run_root: Path):
  cands = [p for p in run_root.glob("logs/%s/*/model_*.pt" % EXPERIMENT)
           if "/pretrained/" not in str(p)]
  if not cands:
    return None
  return max(cands, key=iter_of)


def pack(clips: list[str], out_dir: Path) -> list[int]:
  """把多个 npz 片段拼成一个 packed 目录（mjlab pack_motions.py 的格式）。"""
  data = [dict(np.load(c)) for c in clips]
  names = [Path(c).stem for c in clips]
  ref = [str(x) for x in data[0]["joint_names"]]
  for c, d in zip(clips, data):
    cur = [str(x) for x in d["joint_names"]]
    if cur != ref:
      raise SystemExit("[ERR] 片段关节顺序不一致：%s" % c)
  out_dir.mkdir(parents=True, exist_ok=True)
  for f in FIELDS:
    arr = np.concatenate([d[f] for d in data]).astype(np.float32)
    np.save(out_dir / (f + ".npy"), arr)
  ends = np.cumsum([int(d["joint_pos"].shape[0]) for d in data])
  np.save(out_dir / "motion_ends.npy", ends)
  (out_dir / "joint_names.json").write_text(json.dumps(ref, ensure_ascii=False))
  (out_dir / "motion_names.json").write_text(json.dumps(names, ensure_ascii=False))
  return [int(x) for x in ends]


def main() -> int:
  p = argparse.ArgumentParser(description="LAB27: 多片段联合训练")
  p.add_argument("--run-root", required=True, help="本次 run 的根目录（如 runs/20261126-001-lab27-multiclip）")
  p.add_argument("--mode", choices=["smoke", "full"], default="full")
  p.add_argument("--warm-start", default="scratch/model_49999.pt", help="热启动权重（与 M4 同一个）")
  p.add_argument("--clips", default=DEFAULT_CLIPS, help="逗号分隔的 npz 片段列表（按给定顺序拼接）")
  p.add_argument("--packed", default="", help="已有 packed 目录（给了就用它，忽略 --clips）")
  p.add_argument("--iters", type=int, default=None)
  p.add_argument("--envs", type=int, default=None)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--run-name", default="lab27-multiclip")
  p.add_argument("--resume", action="store_true", help="从本 run 现有最新 ckpt 续训到 TARGET_ITER")
  args = p.parse_args()

  smoke = args.mode == "smoke"
  iters = args.iters if args.iters is not None else (3 if smoke else 3000)
  envs = args.envs if args.envs is not None else (64 if smoke else 4096)

  run_root = Path(args.run_root).resolve()
  log_root = run_root / "logs"
  pretrained = log_root / EXPERIMENT / "pretrained"
  pretrained.mkdir(parents=True, exist_ok=True)

  start = Path(args.warm_start).resolve()
  if args.resume and not smoke:
    last = latest_ckpt(run_root)
    if last is not None:
      n = iter_of(last)
      if n >= TARGET_ITER:
        print("[lab27] 已有 %s（iter %d ≥ %d），无需再训" % (last.name, n, TARGET_ITER))
        return 0
      iters = TARGET_ITER - n + 1
      start = last
      print("[lab27] 续训：从 %s（iter %d）再训 %d 个 iter" % (last, n, iters))
  if not start.exists():
    raise SystemExit("[ERR] 热启动权重不存在：%s" % start)

  n_clips = 0
  total_frames = 0
  if args.packed:
    motion_src = Path(args.packed).resolve()
    if not motion_src.is_dir():
      raise SystemExit("[ERR] --packed 不是目录：%s" % motion_src)
    print("[lab27] 用已有 packed 目录：%s" % motion_src)
  else:
    clips = [str(Path(c.strip()).resolve()) for c in args.clips.split(",") if c.strip()]
    if len(clips) < 2:
      raise SystemExit("[ERR] --clips 至少要 2 段，当前：%s" % clips)
    for c in clips:
      if not Path(c).exists():
        raise SystemExit("[ERR] 片段不存在：%s" % c)
    motion_src = run_root / "motion_packed"
    ends = pack(clips, motion_src)
    n_clips, total_frames = len(clips), ends[-1]
    print("[lab27] 已打包 %d 段 -> %s" % (len(clips), motion_src))
    print("[lab27] motion_ends（累计帧）= %s，合计 %d 帧" % (ends, ends[-1]))
    prev = 0
    for c, e in zip(clips, ends):
      print("[lab27]   %s：%d 帧" % (Path(c).name, e - prev))
      prev = e

  dst = pretrained / ("warm_" + start.name)
  if start.resolve() != dst.resolve():
    shutil.copyfile(start, dst)

  env_cfg = load_env_cfg(TASK)
  agent_cfg = load_rl_cfg(TASK)

  env_cfg.scene.num_envs = envs
  env_cfg.commands["motion"].motion_file = str(motion_src)
  print("[lab27] motion_file = %s（packed 目录；%s）"
        % (motion_src, ("%d 段 / %d 帧" % (n_clips, total_frames)) if n_clips else "沿用已有目录"))

  agent_cfg.logger = "tensorboard"
  agent_cfg.seed = args.seed
  agent_cfg.max_iterations = iters
  agent_cfg.num_steps_per_env = 24
  agent_cfg.save_interval = 500 if not smoke else 1
  agent_cfg.resume = True
  agent_cfg.load_run = "pretrained"
  agent_cfg.load_checkpoint = dst.name
  agent_cfg.run_name = args.run_name

  print("[lab27] iters=%d envs=%d seed=%d save_interval=%d" % (iters, envs, args.seed, agent_cfg.save_interval))
  print("[lab27] 日志根目录：%s（子目录 %s/%s/日期_*）" % (log_root, EXPERIMENT, args.run_name))
  print("[lab27] 自检：%s/params/env.yaml 里 commands.motion.motion_file 应指向 motion_packed 目录" % log_root)

  cfg = TrainConfig(env=env_cfg, agent=agent_cfg, log_root=str(log_root), gpu_ids=[0])
  launch_training(task_id=TASK, args=cfg)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
