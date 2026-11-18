# run 20261116-002-lab06-b-finetune — M3-b 修正对照（LAB06）的 B 组

- 目的：只改 motion_global_root_pos 的 weight（0.5 -> 1.5），std 保持 0.3，检验 LAB05 的“核宽塌缩”解释。
- 训练：4096 env x 300 iter，seed 0，num_steps_per_env 24，从 iter 49999 续训到 50298
- 动作：data/processed/fight1_s3_000s_20s.npz (sha256 965f2c4bb3267217efee7ad65c91fe244995da88dc47e30e684ddea5edb66c66)
- 基础权重：scratch/model_49999.pt (sha256 f1bc404f5165b833a3933ac98ff223450392b78df3eb76b0c36cd1360ff22319)
- config sha256：40072a171af50bcb8183099390bff46498f9f61a23025d2f470c9b5c381f3eaf
- 命令（完整）：见 command.txt
- 墙钟：650 s；退出码：0
- 产物：train.log、gpu_samples.csv（每 2 s 一次显存/温度/功耗）、gpu_before.txt / mem_before.txt / gpu_after.txt / mem_after.txt、logs/g1_tracking/*/model_*.pt、params.txt、cfg_snapshot.py

## 待填（用户）

- 我预测 B 组相对 A 的变化方向、幅度与理由：
