# run 20261111-001-smoke — 本机 GPU 可行性冒烟（无训练）

- 目的与假设：验证 mjlab v1.6.0 的 G1 tracking 环境能在本机 RTX 3060 6GB / WSL2 上创建并步进；先不训练，只测"能不能跑、多快、占多少显存"。
- 运行位置：本地 WSL2 Ubuntu-22.04，GPU 直通。
- 代码版本：repos/mjlab_research v1.6.0（见 docs/SOURCES.md）；脚本 scripts/smoke_env.py。
- 环境：environment/.venv，Python 3.13.18，torch 2.15.0+cu132，mujoco 3.16.0，mujoco-warp 3.16.0，warp-lang 1.19.0。
- 数据：data/processed/lafan_dance1_subject1.npz（6574 帧，50 fps，由 GCS 旧格式转换）。
- 配置：load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1", play=True)，动作全零（不训练，纯随机/零动作策略），sim dt=0.02。
- 命令：
  - `NUM_ENVS=8 STEPS=400 environment/.venv/bin/python scripts/smoke_env.py`
  - `bash scripts/smoke_mem.sh 512 300`（并行 nvidia-smi 采样）
- 结果（warm，kernel 已缓存）：

| num_envs | create+reset (s) | ms/iter | env-steps/s | nvidia-smi 峰值显存 (MiB) |
|---|---|---|---|---|
| 8 | 7.73 | 17.76 | 450 | ~215（另一次采样） |
| 64 | 7.87 | 19.86 | 3,223 | — |
| 512 | 8.08 | 28.83 | 17,760 | 283 |

- 观察：
  1. 首次运行（cold）create 需 64s，主要耗时在 mujoco_warp kernel 首次 load/编译；缓存后降到 ~8s。
  2. 每步墙钟时间随 env 数增长很慢（8→512 只从 17.8ms 到 28.8ms），说明瓶颈是固定开销（Python + kernel launch），不是 env 数量；因此 6GB 显存能容纳远超 512 的环境。
  3. `torch.cuda.max_memory_allocated()` 只有 2MiB，因为 mujoco-warp 用自带 mempool，不走 torch 分配器；显存必须用 nvidia-smi 量。
  4. 用 `time.time()` 计时曾出现负数，是 WSL 时钟跳变；已改用 `time.perf_counter()`（scripts/smoke_env.py）。
  5. play=True 关闭了 RSI 与 push_robot；训练模式（play=False）的显存/速度会更高，需在 M3 重测。
- 结论：**6GB 不是"能否跑 mjlab"的障碍**；本机可以创建环境并以 ~1.8 万 env-steps/s（512 envs）步进。真正的约束是训练时 CPU 内存（WSL 7.6GB）与单步计算量，需在 M3 实测。
- 未验证：训练模式（含 replay buffer / 4096 envs 默认配置 / RSL-RL PPO 更新）在本机的可行 num_envs 与每迭代耗时。
- 下一步：M1——用 mjlab 的预训练 checkpoint 回放 G1 动作（distinguish 运动学播放 vs 策略闭环）。
- 日志：environment/logs/smoke_8env.log、smoke_8env_warm.log、smoke_64env.log、smoke_512env.log
- 复现注意：首次运行冷启动慢属正常，勿在 300s 内判失败。
