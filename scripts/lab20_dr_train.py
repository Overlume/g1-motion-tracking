#!/usr/bin/env python
"""LAB20：训练期 DR 加强版训练（只改 DR 强度，其余与 M4 完全一致）。

背景（2026-11-21 核实）：mjlab 的 tracking 任务默认就带三项 startup DR——
foot_friction 0.3–1.2（双脚共享）、base_com ±2.5/±5/±5 cm、encoder_bias ±0.01 rad。
所以 LAB20 不是「开 DR」，而是「把 DR 调强」，对照组 = M4（默认 DR）。

用法：
  # 冒烟（300 iter / 512 envs，约 5 min）
  environment/.venv/bin/python scripts/lab20_dr_train.py --run-root scratch/lab20-smoke --mode smoke

  # 正式（3000 iter / 4096 envs，约 1.8 h）
  environment/.venv/bin/python scripts/lab20_dr_train.py --run-root runs/20261121-003-lab20-dr --mode full

只改 DR（其余保持 M4 口径）：摩擦 (0.2, 1.4)、质心 ±4/±6/±6 cm、编码器偏置 ±0.02 rad。
用 --default-dr 可跑「默认 DR」的对照臂（用于验证本脚本没有偷偷改别的）。
"""

import argparse
import shutil
from pathlib import Path

import mjlab.tasks  # noqa: F401  # 注册任务
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

TASK = "Mjlab-Tracking-Flat-Unitree-G1"
EXPERIMENT = "g1_tracking"


def main() -> int:
  p = argparse.ArgumentParser(description="LAB20: 训练期 DR 加强版训练")
  p.add_argument("--run-root", required=True, help="本次 run 的根目录（如 runs/20261121-003-lab20-dr）")
  p.add_argument("--mode", choices=["smoke", "full"], default="full")
  p.add_argument("--warm-start", default="scratch/model_49999.pt", help="热启动权重（与 M4 同一个）")
  p.add_argument("--motion", default="data/processed/fight1_s3_000s_20s.npz")
  p.add_argument("--friction", nargs=2, type=float, default=(0.2, 1.4), metavar=("LO", "HI"))
  p.add_argument("--com", nargs=3, type=float, default=(0.04, 0.06, 0.06), metavar=("X", "Y", "Z"),
                 help="质心偏移半宽（米）：默认 DR 是 0.025/0.05/0.05")
  p.add_argument("--bias", type=float, default=0.02, help="编码器偏置半宽（弧度）：默认 DR 是 0.01")
  p.add_argument("--iters", type=int, default=None)
  p.add_argument("--envs", type=int, default=None)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--run-name", default="lab20-dr")
  p.add_argument("--default-dr", action="store_true", help="不加强 DR，跑默认范围（对照臂）")
  args = p.parse_args()

  iters = args.iters if args.iters is not None else (300 if args.mode == "smoke" else 3000)
  envs = args.envs if args.envs is not None else (512 if args.mode == "smoke" else 4096)

  run_root = Path(args.run_root).resolve()
  log_root = run_root / "logs"
  pretrained = log_root / EXPERIMENT / "pretrained"
  pretrained.mkdir(parents=True, exist_ok=True)
  warm = Path(args.warm_start).resolve()
  if not warm.exists():
    raise SystemExit("热启动权重不存在：%s" % warm)
  dst = pretrained / warm.name
  if not dst.exists():
    shutil.copy2(warm, dst)
    print("[lab20] 已复制热启动权重 -> %s" % dst)

  env_cfg = load_env_cfg(TASK)
  agent_cfg = load_rl_cfg(TASK)

  env_cfg.scene.num_envs = envs
  env_cfg.commands["motion"].motion_file = str(Path(args.motion).resolve())

  if args.default_dr:
    print("[lab20] DR = 默认（对照臂）")
  else:
    env_cfg.events["foot_friction"].params["ranges"] = (args.friction[0], args.friction[1])
    cx, cy, cz = args.com
    env_cfg.events["base_com"].params["ranges"] = {
      0: (-cx, cx),
      1: (-cy, cy),
      2: (-cz, cz),
    }
    env_cfg.events["encoder_bias"].params["bias_range"] = (-args.bias, args.bias)
    print("[lab20] DR 已加强：friction=(%.2f,%.2f) com=±%.2f/±%.2f/±%.2f m bias=±%.3f rad"
          % (args.friction[0], args.friction[1], cx, cy, cz, args.bias))

  agent_cfg.logger = "tensorboard"
  agent_cfg.seed = args.seed
  agent_cfg.max_iterations = iters
  agent_cfg.num_steps_per_env = 24
  agent_cfg.save_interval = 500 if args.mode == "full" else 100
  agent_cfg.resume = True
  agent_cfg.load_run = "pretrained"
  agent_cfg.load_checkpoint = dst.name
  agent_cfg.run_name = args.run_name

  print("[lab20] iters=%d envs=%d seed=%d save_interval=%d" %
        (iters, envs, args.seed, agent_cfg.save_interval))
  print("[lab20] 日志根目录：%s（子目录 %s/%s/%s_*）" % (log_root, EXPERIMENT, args.run_name, "YYYY-mm-dd"))
  print("[lab20] 自检：训练开始后看 %s/params/env.yaml 里的 foot_friction/base_com/encoder_bias" % log_root)

  cfg = TrainConfig(env=env_cfg, agent=agent_cfg, log_root=str(log_root), gpu_ids=[0])
  launch_training(task_id=TASK, args=cfg)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
