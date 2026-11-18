# run 20261117-001-m4-w15-finetune — M4 规模训练（weight 1.5 / std 0.3）

- 目的：在 LAB07 选定的 weight 1.5 上把预算从 300 iter 提到 3000 iter，看 root 误差是否随预算下降、姿态是否付钱。
- 训练：4096 env x 3000 iter，seed 0，num_steps_per_env 24，49999 -> 52998，save_interval 500
- 环境步数：3000 x 4096 x 24 ≈ 294912000
- 动作：data/processed/fight1_s3_000s_20s.npz (sha256 965f2c4b…66c66)；基础权重：scratch/model_49999.pt (sha256 f1bc404f…22319)
- 真实墙钟（/proc/uptime）：6412 s（1.78 h）；退出码 0
- 命令：见 command.txt；配置：cfg_snapshot.py（sha 40072a17…3eaf，仅 212 行 weight=1.5 与 A 态不同）

## 用户预测（docs/learning/LAB08-prediction.md 摘要）

- Q1 root 误差：约 10 cm（理由：迭代 ×10、误差减半，非线性）
- Q2 R-MPKPE：变差（理由：策略会更用力压根位置，对姿态更宽容）
- Q3 error_anchor_pos：约 0.05 m；并注明“不清楚两者差在哪”
- Q4（假想 root 24 cm 时）：下一步先改 std（排查链：硬件 / 数据 / 策略）

判分见 runs/20261117-002-m4-eval/README.md（Q1 方向对幅度保守、Q2 错、Q3 口径混淆、Q4 前提错但推理链合理）。

## 训练侧读数

| 量 | 起点 | 500 | 1000 | 1500 | 2000 | 2500 | 终点 52998 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| error_anchor_pos (m) | 0.1314 | 0.1547 | 0.1261 | 0.1256 | 0.1183 | 0.1119 | 0.1141 |

- Mean reward：0.67 → 24.65；Mean episode length：19.35 → 296.09 步（≈5.9 s）。
- 迭代耗时：首次 14.99 s，稳态中位 1.99 s；3000 条里 142 条为负值（D14 时钟回跳，4.7%）。
- GPU（1265 采样 / 每 5 s）：显存中位/p95/峰值 1967 / 1967 / 1975 MiB（6144 MiB 的 32%）；温度中位 86 ℃、峰值 88 ℃；利用率中位 75 %、p95 85 %；功耗中位 78 W、峰值 126.7 W。
- ckpt：model_50000 / 50500 / … / 52998（save_interval 500 + 收尾），另有 onnx 与 tfevents。

## 与评估侧的关系

训练侧 error_anchor_pos 只降 13%，评估侧 root xy 误差降 85%。两者口径不同（3D vs xy、随机抽帧 vs 确定性 rollout、含噪声动作 vs 均值动作），不能互相校验；详见 runs/20261117-002-m4-eval/README.md。

## 结论

预算轴是主要瓶颈：在该片段上，300 → 3000 iter 把 root xy 误差从 25.72 压到 3.90 cm（−84.8%），姿态/关节/滑移同向变好，无 H3 的“位置换姿态”。

**但**：同 ckpt 换未见片段即崩（runs/20261117-003-generalization-probe/，完成度 0.10 / 0.02，0.3–2 s 内 ee_body_pos 终止），所以这是“该片段的专家”，不是“会跟踪”。引用 3.90 cm 时必须标注 in-distribution。

