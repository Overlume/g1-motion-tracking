# run 20261116-004-lab07-w10-finetune — LAB07 权重扫描的 w1.0 档

- 唯一变量：motion_global_root_pos 的 weight = 1.0（std 固定 0.3）
- 训练：4096 env x 300 iter，seed 0，num_steps_per_env 24，从 iter 49999 续训到 50298
- 动作：data/processed/fight1_s3_000s_20s.npz (sha256 965f2c4bb3267217efee7ad65c91fe244995da88dc47e30e684ddea5edb66c66)
- 基础权重：scratch/model_49999.pt (sha256 f1bc404f5165b833a3933ac98ff223450392b78df3eb76b0c36cd1360ff22319)
- config sha256：c266bc869fcf616f17183f202bea95d901e79180e00e119370e3f2482204dcba
- 命令（完整）：见 command.txt
- 墙钟：629 s（实时钟口径，见 D14）；退出码：0
- 产物：train.log、gpu_samples.csv（每 2 s 一次显存/温度/功耗）、gpu_/mem_ before/after、cfg_snapshot.py、params.txt

## 待填（用户）

- 我对这一档 root 误差的预测与理由（相对上一档）：
