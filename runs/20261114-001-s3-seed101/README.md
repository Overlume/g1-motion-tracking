# 20261114-001~005：S3 五种子摔倒分布

## 目的

S3（fight1_subject3 前 600 帧，999 帧 @50fps = 19.98 s）在官方预训练策略下极不稳定，
同命令同 seed 的结果差异很大（D11）。本组实验用 5 个不同种子量化"摔不摔、多久摔"。

## 设置

- checkpoint：官方预训练 `scratch/model_49999.pt`（sha256 f1bc404f...f22319）
- 动作：`data/processed/fight1_s3_000s_20s.npz`
- 脚本：`scripts/eval_tracking.py`
- 参数：`--keep-dr`（保留 base_com / encoder_bias / foot_friction 启动随机化）、
  `--ignore-terminations`（关掉 episode 终止，跑到真实摔倒）
- 判据：pelvis_z < 0.15 m 连续 5 步 = 摔倒
- 机器：WSL2 / RTX 3060 6 GB / 1 env

命令（每个种子）：

    MUJOCO_GL=egl environment/.venv/bin/python scripts/eval_tracking.py \
      --motion-file data/processed/fight1_s3_000s_20s.npz \
      --seed 101 --keep-dr --ignore-terminations \
      --out runs/20261114-001-s3-seed101

## 结果

| seed | 摔倒 | 摔倒步 | 摔倒时间 s | 完成度 | MPKPE cm | R-MPKPE cm | root_err cm | 脚滑 L/R cm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | 是 | 928 | 18.56 | 0.931 | 153.4 | 22.7 | 149.5 | 149 / 129 |
| 102 | 是 | 277 | 5.54 | 0.279 | 91.6 | 34.1 | 85.7 | 69 / 43 |
| 103 | 是 | 583 | 11.66 | 0.586 | 125.1 | 11.0 | 124.8 | 179 / 184 |
| 104 | 是 | 510 | 10.20 | 0.513 | 91.1 | 12.7 | 89.2 | 92 / 97 |
| 105 | 是 | 607 | 12.14 | 0.610 | 155.5 | 12.9 | 155.9 | 170 / 204 |

- 摔倒率：5/5
- 摔倒时间中位数：11.66 s；范围 5.54–18.56 s
- 5 次运行各自耗时 5.6–18.6 s 仿真时间（摔倒即停止统计）

## 解读

1. 官方策略在 S3 上**必然失败**，只是时间不同（5.5–18.6 s）。
2. MPKPE 均值与摔倒时间正相关（漂移累积久了均值大），R-MPKPE 与动作激烈程度相关
   （seed102 最早被甩开，姿态误差最大 34.1 cm）。
3. 单次 fall_step 不能当结论（D11）；对外只报分布。

## 未验证

- 5 个种子是否覆盖最坏情形未知；如需更稳的分布建议 N≥10。
- 摔倒后继续跑的行为（地上挣扎）未逐帧分析。
