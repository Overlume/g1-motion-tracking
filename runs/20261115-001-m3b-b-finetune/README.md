# run 20261115-001-m3b-b-finetune — M3-b B 组微调（从官方 model_49999 续训）

- 目的：M3-b 单参数对照的 B 组。**仅** B 组配置不同（见 cfg_snapshot.py 与 diff_A.txt）。
- 训练：4096 env × 300 iter，seed 0，num_steps_per_env 24，从 iter 49999 续训到 49999 + 300 - 1
- 动作：data/processed/fight1_s3_000s_20s.npz (sha256 965f2c4bb3267217efee7ad65c91fe244995da88dc47e30e684ddea5edb66c66)
- 基础权重：scratch/model_49999.pt (sha256 f1bc404f5165b833a3933ac98ff223450392b78df3eb76b0c36cd1360ff22319)
- config sha256：fd20593a75603aa0c9c5cf495ba4e99bb246692bd2a9e2110eea4dd06b503d85
- 命令（完整）：见 command.txt
- 墙钟：728 s；退出码：0
- 产物：train.log、vram_mib.txt、logs/g1_tracking/*/model_*.pt、params.txt、cfg_snapshot.py

## 待填（用户）

- 我预测 B 组相对 A 的变化方向与理由：
