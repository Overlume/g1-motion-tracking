# M3-b A/B 评估（LAB05 结果）

- 目的：检验“把 `motion_global_root_pos` 的 weight 0.5 -> 1.0、std 0.3 -> 0.15”是否让官方预训练策略在 S3 上的**全局根位置跟踪**变好。
- 训练侧（各 4096 env x 300 iter、seed 0、从 `scratch/model_49999.pt` 续训，49999 -> 50298）：
  - A = 原奖励：`runs/20261114-007-m3b-a-finetune`（日志 `logs/g1_tracking/2026-11-14_22-04-38_m3b-a`，墙钟 10 分 10 秒，尾迭代 2.16 s）
  - B = 修改后：`runs/20261115-001-m3b-b-finetune`（日志 `logs/g1_tracking/2026-11-15_09-38-26_m3b-b`，中位 2.07 s；`diff_A.txt` 证明只改了那两行）
- 评估：`scripts/eval_tracking.py`，动作 `data/processed/fight1_s3_000s_20s.npz`（999 帧 / 19.98 s），**终止检测开**（未传 `--ignore-terminations`），启动 DR 关，种子 101-105，命令见 `A/seedXXX.log` / `B/seedXXX.log` 首行。
- 汇总：`comparison.md` / `comparison.json`（脚本 `scripts/lab05_ab_report.py`）。

## 结果

| 指标（5 种子） | A 中位 (min-max) | B 中位 (min-max) | 方向 |
| --- | --- | --- | --- |
| 摔倒次数 | 0/5 | 0/5 | 持平 |
| track_completion | 0.999 (0.999-0.999) | 0.999 (0.999-0.999) | 持平 |
| root xy 误差 均值 cm | 43.34 (40.68-45.80) | 52.72 (51.27-53.79) | B 差 +9.38 cm |
| 漂移速率 cm/s | 0.06 (-0.26-0.37) | 0.70 (0.66-0.77) | B 差 +0.64 cm/s |
| R-MPKPE 均值 cm | 5.707 (5.64-5.808) | 6.016 (5.999-6.069) | B 差 +0.31 cm |
| MPKPE 均值 cm | 44.27 (41.56-46.55) | 53.47 (52.09-54.71) | B 差 +9.20 cm |

两个分布的 5 个种子**互不重叠**（A 最大 45.80 < B 最小 51.27），不是噪声。

训练侧（最后 30 次迭代均值，来自 `train.log`）：

| 项 | A | B |
| --- | --- | --- |
| Mean reward | 9.847 | 9.886 |
| Mean episode length | 186.0 | 188.4 |
| Metrics/motion/error_anchor_pos | 0.3859 m | 0.5081 m |
| Episode_Termination/anchor_pos | 0.1306 | 0.1083 |
| Episode_Reward/motion_global_root_pos | 0.0633 | 0.0453-0.0483 |
| 迭代耗时中位 s | 2.17 | 2.07 |

训练侧 `error_anchor_pos`（0.386 -> 0.508 m）与评估侧 `root_err_mean`（43.3 -> 52.7 cm）是两套独立测量，方向一致——这是本例最有力的证据，比单看评估更强。

## 判读

### 1) 最大的结果不是 A/B 差异，而是“300 iter 就不摔了”

官方预训练策略在 S3 上 5/5 摔倒（`runs/20261114-001~005`，中位 11.66 s）。续训 300 iter（4096 env x 24 步 x 300 = 29,491,200 env-steps，约为官方 50k iter 的 0.6%）后两组都变成 0/5 摔倒、跑满 999 步。

但**不摔 != 学会**：root xy 误差 43 cm、MPKPE 44 cm，而根相对姿态 R-MPKPE 只有 5.7 cm。也就是“身体姿态学得像、在世界里站错了地方”。F3 的 `metrics.csv` 显示 root 误差在 5-112 cm 之间来回摆动（不是单调漂移），说明轨迹形状跟上了、起点相位/尺度对不上。

### 2) 收窄 std 让根位置跟踪变差，与直觉相反

奖励实现是 `exp(-||d||^2 / std^2)`（`repos/mjlab_research/src/mjlab/tasks/tracking/mdp/rewards.py:26-33`）。在观测到的误差水平 d ≈ 0.45 m 上：

- A（std 0.3）：核值 `exp(-(0.45/0.3)^2)` ≈ 0.11，梯度幅值 `(2d/std^2) * exp(...)` ≈ 1.05；乘 weight 0.5 得 ≈ 0.52
- B（std 0.15）：核值 `exp(-(0.45/0.15)^2)` ≈ 1.2e-4，梯度幅值 ≈ 4.9e-3；乘 weight 1.0 得 ≈ 0.0049

**同一误差水平上 B 的梯度比 A 小约 100 倍**，权重 x2 完全补不回来。核一收窄，该项在 0.3 m 以外就“失效”了，只剩别的项（body_pos 等，根相对）在塑形，于是根可以在世界里自由漂——这与评估看到的“R-MPKPE 好、root 误差大”完全一致。

这条也解释了训练侧数字：A 的 `Episode_Reward/motion_global_root_pos` = 0.0633，B 尽管权重翻倍却只有 0.045-0.048（奖励项本身塌了）。

### 3) 用户三道预测的判分

见 `docs/learning/LAB05-prediction.md`（用户原文保留）。

- 问题 1（摔倒次数不变）**结论对**：确实 0/5 vs 0/5。但理由不完整：真正值得预测的是“两组都会从 5/5 摔变成不摔”，这一点没人预判到。
- 问题 2（漂移变好）**错**：0.06 -> 0.70 cm/s，B 差 12 倍。
- 问题 3（error_anchor_pos 变小）**错**：0.3859 -> 0.5081 m，B 更大。

教训：问题 2 和 3 是同一个物理量的两种量法（评估 vs 训练），两题同方向错 = 一个一致的判断错误，而不是两处笔误。根因是把“权重更大”直接等价于“跟得更紧”，忽略了核宽收窄先把信号掐掉。

## 局限（写报告时必须带）

- **训练集 = 评估集**：微调用的就是 S3，5 个种子只改初始随机化。这里的“不摔”是拟合同一条动作，不是泛化。
- **口径不同**：训练 `episode_length_s = 10.0`（`repos/mjlab_research/src/mjlab/tasks/tracking/tracking_env_cfg.py:315`，最多 500 步），episode 主要由 `motion_end` 截断；评估跑满 999 步 / 19.98 s。A/B 同协议可比，但训练 `Mean episode length`（186/188）不能与评估的 999 步对照。
- **LAB05 同时改了两个参数**（weight 与 std），严格上分不开各自贡献；上面的梯度分析只是给“std 主导”提供了机制解释，不是分离实验。要分离得补一个只改 weight 的对照。
- `Episode_Termination/*` 的绝对值不是次数：mjlab 在每步记录 reset 计数再按步平均（`repos/mjlab_research/src/mjlab/managers/termination_manager.py:94-96`），只能看相对比例（本例 `motion_end` 占绝对多数，`anchor_pos` 极少）。
- 20 秒片段的末端样本受 reset 影响（`root_err_last_cm` 出现 0.0），对均值的偏置 < 0.05 cm，可忽略但要写。

## 复现

    bash scripts/lab05_ab_finetune.sh train A      # 需要 config 为 A 态（sha256 f756b4d0...b9a82）
    # 手动改两行（见 docs/learning/LAB05-ab-finetune.md），再
    bash scripts/lab05_ab_finetune.sh train B
    bash scripts/lab05_ab_finetune.sh eval
    bash scripts/lab05_ab_finetune.sh report

训练脚本当时用的完整命令见 `runs/20261114-007-m3b-a-finetune/command.txt`；评估脚本单次命令见 `A/seed101.log` 首行。