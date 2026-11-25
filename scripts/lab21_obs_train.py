#!/usr/bin/env python
"""LAB21：actor 观测追加世界系根位置/速度，其余与 M4 完全一致。

为什么需要单独的训练脚本：actor 观测从 160 维变 166 维后，M4 的第一层权重形状对不上，
不能直接 resume。本脚本先做一次「维度扩展」再热启动：
  - 兼容两代 ckpt 格式（旧：model_state_dict + actor.*；新：actor_state_dict + mlp.*），
    旧格式先按 mjlab 的迁移规则转成新格式再处理；
  - actor 第一层 mlp.0.weight 右侧补 0 列（新输入初始被忽略，等价于「原策略 + 6 个哑输入」起步）；
  - 优化器里同一张量的 exp_avg / exp_avg_sq 也补 0 列（动量从 0 开始，不做假的继承）；
  - 观测归一化的新维度设 mean=0 / var=1 / std=1（新维度先按原值进网络，训练中再被归一化器吸收）。
其余口径（动作/超参/DR/迭代数/种子）与 LAB20、LAB24 的训练完全一致，所以任何差异都只来自观测。

用法（项目根目录）：
  # 冒烟（3 iter / 64 envs，约 40 s）
  environment/.venv/bin/python scripts/lab21_obs_train.py --run-root scratch/lab21-smoke --mode smoke

  # 正式（3000 iter / 4096 envs，约 1.6 h；中断后再跑同一条命令即续训）
  environment/.venv/bin/python scripts/lab21_obs_train.py --run-root runs/20261125-003-lab21-world-obs --mode full --resume
"""

import argparse
import shutil
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401  # 注册任务
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from obs_world import EXTRA_DIMS, add_world_root_obs

TASK = "Mjlab-Tracking-Flat-Unitree-G1"
EXPERIMENT = "g1_tracking"
TARGET_ITER = 52998      # M4 / LAB20 / LAB24 的终点迭代
BASE_ACTOR_OBS = 160     # M4 的 actor 观测维度


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


def migrate_legacy(d: dict) -> dict:
  """把旧格式（model_state_dict + actor.*）转成 mjlab/rsl-rl 5.x 的新格式（原地改 d）。"""
  if "actor_state_dict" in d or "model_state_dict" not in d:
    return d
  msd = d.pop("model_state_dict")
  actor, critic = {}, {}
  for k, v in msd.items():
    if k.startswith("actor_obs_normalizer."):
      actor["obs_normalizer." + k[len("actor_obs_normalizer."):]] = v
    elif k.startswith("actor."):
      actor["mlp." + k[len("actor."):]] = v
    elif k in ("std", "log_std"):
      actor["distribution.std_param"] = v
    if k.startswith("critic_obs_normalizer."):
      critic["obs_normalizer." + k[len("critic_obs_normalizer."):]] = v
    elif k.startswith("critic."):
      critic["mlp." + k[len("critic."):]] = v
  d["actor_state_dict"] = actor
  d["critic_state_dict"] = critic
  return d


def expand(src: Path, dst: Path) -> int:
  """把 M4 口径的 ckpt 扩成 166 维 actor 观测，写进 dst；返回新维度。"""
  d = migrate_legacy(torch.load(src, map_location="cpu", weights_only=False))
  a = d["actor_state_dict"]
  w = a["mlp.0.weight"]
  old, want = int(w.shape[1]), BASE_ACTOR_OBS + EXTRA_DIMS
  if old == want:
    if src.resolve() != dst.resolve():
      torch.save(d, dst)
    print("[lab21] ckpt 已是 %d 维 actor 观测，直接使用：%s" % (want, dst))
    return want
  if old != BASE_ACTOR_OBS:
    raise SystemExit("[ERR] actor 第一层输入维度 %d 既不是 %d（M4 口径）也不是 %d（已扩展），无法自动处理"
                     % (old, BASE_ACTOR_OBS, want))
  nw = torch.zeros(w.shape[0], want, dtype=w.dtype)
  nw[:, :old] = w
  a["mlp.0.weight"] = nw
  for key, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
    k = "obs_normalizer." + key
    t = a[k]
    a[k] = torch.cat([t, torch.full((t.shape[0], EXTRA_DIMS), fill, dtype=t.dtype)], dim=1)
  n_fix = 0
  for _i, st in d["optimizer_state_dict"]["state"].items():
    if "exp_avg" in st and tuple(st["exp_avg"].shape) == (int(w.shape[0]), old):
      for k in ("exp_avg", "exp_avg_sq"):
        t = st[k]
        nt = torch.zeros(t.shape[0], t.shape[1] + EXTRA_DIMS, dtype=t.dtype)
        nt[:, :t.shape[1]] = t
        st[k] = nt
      n_fix += 1
  torch.save(d, dst)
  print("[lab21] 已扩展 ckpt -> %s（actor 第一层 %d -> %d，优化器动量修了 %d 个张量）"
        % (dst, old, want, n_fix))
  return want


def main() -> int:
  p = argparse.ArgumentParser(description="LAB21: actor 观测追加世界系根位置/速度")
  p.add_argument("--run-root", required=True, help="本次 run 的根目录（如 runs/20261125-003-lab21-world-obs）")
  p.add_argument("--mode", choices=["smoke", "full"], default="full")
  p.add_argument("--warm-start", default="scratch/model_49999.pt", help="热启动权重（与 M4 同一个）")
  p.add_argument("--motion", default="data/processed/fight1_s3_000s_20s.npz")
  p.add_argument("--iters", type=int, default=None)
  p.add_argument("--envs", type=int, default=None)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--run-name", default="lab21-world-obs")
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
        print("[lab21] 已有 %s（iter %d ≥ %d），无需再训" % (last.name, n, TARGET_ITER))
        return 0
      iters = TARGET_ITER - n + 1
      start = last
      print("[lab21] 续训：从 %s（iter %d）再训 %d 个 iter" % (last, n, iters))
  if not start.exists():
    raise SystemExit("[ERR] 热启动权重不存在：%s" % start)

  dst = pretrained / ("warm_" + start.name)
  dim = expand(start, dst)

  env_cfg = load_env_cfg(TASK)
  agent_cfg = load_rl_cfg(TASK)

  env_cfg.scene.num_envs = envs
  env_cfg.commands["motion"].motion_file = str(Path(args.motion).resolve())
  add_world_root_obs(env_cfg)

  agent_cfg.logger = "tensorboard"
  agent_cfg.seed = args.seed
  agent_cfg.max_iterations = iters
  agent_cfg.num_steps_per_env = 24
  agent_cfg.save_interval = 500 if not smoke else 1
  agent_cfg.resume = True
  agent_cfg.load_run = "pretrained"
  agent_cfg.load_checkpoint = dst.name
  agent_cfg.run_name = args.run_name

  print("[lab21] actor 观测 %d -> %d 维（追加 root_pos_w / root_vel_w）" % (BASE_ACTOR_OBS, dim))
  print("[lab21] iters=%d envs=%d seed=%d save_interval=%d" % (iters, envs, args.seed, agent_cfg.save_interval))
  print("[lab21] 日志根目录：%s（子目录 %s/%s/日期_*）" % (log_root, EXPERIMENT, args.run_name))
  print("[lab21] 自检：训练开始后看 %s/params/env.yaml 里 observations.actor.terms 是否有 root_pos_w / root_vel_w"
        % log_root)

  cfg = TrainConfig(env=env_cfg, agent=agent_cfg, log_root=str(log_root), gpu_ids=[0])
  launch_training(task_id=TASK, args=cfg)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
