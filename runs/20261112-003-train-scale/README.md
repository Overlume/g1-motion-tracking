# run 20261112-003-train-scale — 训练规模与显存/耗时扫描

- 目的：测本机 6GB 上不同 num_envs 的迭代耗时与显存峰值，为 M3 选训练规模。
- 位置：本地 WSL2 / RTX 3060 6GB；脚本 scripts/smoke_train.sh（训练时每 0.5 s 用 nvidia-smi 采样显存）。
- 共同设置：Mjlab-Tracking-Flat-Unitree-G1，play=False，seed=0，动作 lafan_dance1_subject1.npz，tensorboard logger。

| num_envs | 稳态迭代耗时 | 显存峰值 | 日志 |
|---|---|---|---|
| 512 | 0.82 s | 未采样（见 002） | environment/logs/train_smoke_512env.log |
| 1024 | 1.03 s | 653 MiB | environment/logs/train_scale_1024env.log |
| 4096 | 1.76-1.90 s | 1871 MiB | environment/logs/train_scale_4096env.log |

- 首次迭代含编译/预热：512 env 时 31 s，4096 env 时 26.8 s。
- 结论：本机 6GB 可以开官方默认的 4096 envs，显存只用了约 1.9GB；耗时增长远低于线性（512→4096 是 8 倍环境、约 2.2 倍时间），说明瓶颈不在显存也不在单环境开销。
- 推断（未验证）：真正的限制是 CPU 内存（WSL 7.6GB）与 PPO 更新的样本量；4096 env × 24 步 ≈ 98k 样本/次。
- 计时异常：1024 env 日志出现一次 -13.25 s 的迭代时间，是 WSL 时钟跳变，已知问题，不影响稳态数据。
