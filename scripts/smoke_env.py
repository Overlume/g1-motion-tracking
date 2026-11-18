"""最小 GPU 冒烟：创建 G1 tracking 环境、加载动作、跑若干步，测显存与耗时。

不做训练，不写检查点；只回答"本机能否跑起来、多快、占多少显存"。
"""
import os
import time

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK = os.environ.get("TASK", "Mjlab-Tracking-Flat-Unitree-G1")
NUM_ENVS = int(os.environ.get("NUM_ENVS", "8"))
STEPS = int(os.environ.get("STEPS", "100"))
MOTION = os.environ.get(
    "MOTION", "data/processed/lafan_dance1_subject1.npz"
)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = NUM_ENVS
cfg.commands["motion"].motion_file = MOTION

t0 = time.perf_counter()
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env.reset()
torch.cuda.synchronize()
t_create = time.perf_counter() - t0

action_dim = env.action_space.shape[-1]
act = torch.zeros((NUM_ENVS, action_dim), device="cuda:0")

# warmup
for _ in range(5):
  env.step(act)
torch.cuda.synchronize()

chunks = 5
per = max(1, STEPS // chunks)
chunk_times = []
for _ in range(chunks):
  t1 = time.perf_counter()
  for _ in range(per):
    env.step(act)
  torch.cuda.synchronize()
  chunk_times.append((time.perf_counter() - t1) / per)
dt = sorted(chunk_times)[len(chunk_times) // 2]
print("chunk ms/iter: " + ", ".join(f"{x*1000:.2f}" for x in chunk_times))

print(f"TASK={TASK} NUM_ENVS={NUM_ENVS} STEPS={STEPS}")
print(f"create+reset: {t_create:.2f}s")
print(f"env.step: {dt*1000:.2f} ms/iter -> {1/dt:.1f} iter/s; sim dt={env.step_dt}")
print(f"steps/s (env-steps): {NUM_ENVS/dt:.0f}")
print(f"peak VRAM: {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
print(f"reserved VRAM: {torch.cuda.max_memory_reserved()/2**20:.0f} MiB")
