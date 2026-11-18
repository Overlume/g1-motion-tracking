# LAB07 —— 权重扫描：把 weight 从两点扩成四点（0.5 / 1.0 / 1.5 / 2.5）

预计 35 分钟（2 次训练 ≈ 22 min，评估 10 次 ≈ 10 min，其余是你的思考时间）。

## 1. 为什么要做这次扫描

LAB05 与 LAB06 只给了两个点（0.5 与 1.5），我们因此只能写下“weight 越大越好，至少到 1.5”。但这句话有三种完全不同的可能，而它们对 M4 的影响完全相反：

| 曲线形状 | 含义 | 对 M4 的影响 |
| --- | --- | --- |
| 单调下降 | 权重还不够大 | M4 应该用更大的权重（2.5 甚至更大） |
| 先降后平（膝点） | 1.5 已经吃到大部分收益 | M4 用 1.5，别浪费预算追 2.5 |
| U 形（1.5 是最低点） | 权重过大开始有代价（姿态/滑移/动作变形） | M4 必须停在最优点，且要报告代价 |

两个点分不出这三种。四点（0.5 / 1.0 / 1.5 / 2.5）才有形状。**这是你提出的严谨性要求，也是这次实验的全部价值。**

顺带把 LAB05 的混淆彻底清掉：LAB05-B 是 w1.0 + std0.15，本表里的 w1.0 是 w1.0 + std0.3，两者可以直接对照，把“std 收窄”与“weight 变大”各占多少说清。

## 2. 你不用重训已有的两档

| 臂 | weight | 训练 run | 评估 | 状态 |
| --- | --- | --- | --- | --- |
| w0.5 | 0.5 | `runs/20261116-001-lab06-a-finetune` | `runs/20261116-003-lab06-eval/A` | 已有 |
| w1.0 | 1.0 | 本次新建 | 本次新建 | **要跑** |
| w1.5 | 1.5 | `runs/20261116-002-lab06-b-finetune` | `runs/20261116-003-lab06-eval/B` | 已有 |
| w2.5 | 2.5 | 本次新建 | 本次新建 | **要跑** |

所以只需两次训练、两次评估（各 5 个种子），然后四个臂进同一张表。

## 3. 先写预测（`docs/learning/LAB07-prediction.md`），四题

1. **四个点的 root 误差排序**，以及曲线形状（单调下降 / 先降后平 / U 形）。理由要说清“为什么 1.5 之后加权重可能反而变差”。
2. 如果继续加权重开始有代价，**最先暴露的是哪个指标**：R-MPKPE、joint_rmse、tilt 误差、左脚滑移、还是摔倒次数？为什么？
3. **w1.0 相对 w0.5 的改善幅度**，与 **w1.5 相对 w1.0 的改善幅度**，哪个更大？如果你认为是前者更大，那说明什么？
4. 如果结果是 2.5 最好（或 1.0 最好），**你敢不敢直接用那一档去跑 M4**？写出判断依据与风险。

提示（不是答案）：评估噪声底约 4%（同配置重跑 43.34 vs 41.52），相邻两档差小于 4% 时只能写“不可分辨”，不能排序。

## 4. 操作步骤

```bash
# 1) 先写预测
#    docs/learning/LAB07-prediction.md

# 2) 第一档：把 weight 改成 1.0（std 不要动）
#    repos/mjlab_research/src/mjlab/tasks/tracking/tracking_env_cfg.py:212
#      weight=0.5,  ->  weight=1.0,
bash scripts/lab07_weight_sweep.sh train 1.0

# 3) 第二档：改成 2.5（同样只动这一行）
#      weight=1.0,  ->  weight=2.5,
bash scripts/lab07_weight_sweep.sh train 2.5

# 4) 评估两档，各 5 个种子（默认 101-105）
bash scripts/lab07_weight_sweep.sh eval

# 5) 四臂汇总：comparison.md / sweep.csv / sweep.png
bash scripts/lab07_weight_sweep.sh report
```

脚本会在动手训练前校验：**与 LAB06-A 快照相比恰好 +1/-1 行、改的那一行含 `weight=`、第 210-214 行里 weight 等于你传的值、std 仍是 0.3**。校验不过直接退出，不会白跑 11 分钟。

## 5. 跑完看什么（顺序很重要）

1. **先看 `comparison.md` 的“曲线”表**：四个 weight 的 root 中位，以及相对上一档的百分比。形状就藏在这一列里。
2. **再看 `sweep.png` 左图**（root 误差 vs weight，带 min–max 误差棒）：误差棒重叠 = 不可分辨。
3. **然后看右图**（R-MPKPE vs weight）：如果它随权重上升而变差，说明开始“用姿态换位置”，这就是 U 形右半段的证据。
4. **最后读训练侧的 `train.log`**：error_anchor_pos 是否随权重单调下降（物理量，可与评估互相印证）；episode length 是否还在 190+。

## 6. 常见坑

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `[ERR] config 与 A 态完全相同` | 你还没改 weight | 改第 212 行再跑 |
| `[ERR] 期望只改 1 行` | 顺手改了 std 或别的行 | 还原后只改 weight |
| `[ERR] 第 210-214 行的 weight 应为 2.5` | 传的数和改的数不一致 | 两处对齐 |
| 找不到 `lab06-a-finetune` | 别删 `runs/20261116-001-*` | 它是 w0.5 的基线快照，删了就没有 A 态可比 |
| 迭代耗时偶尔为负 | WSL2 实时钟跳变（D14） | 正常，不影响指标 |
| `eval` 报找不到 `model_50298.pt` | 训练没跑完 | 看 train.log 末尾与退出码 |

## 7. 通过标准

- [ ] `docs/learning/LAB07-prediction.md` 在训练开始前写好（4 题都答）
- [ ] 两个新 run 目录齐全，`params.txt` 除 weight / cfg_sha256 外一致，`diff_A.txt` 各只有 1 行
- [ ] `runs/*-lab07-eval/comparison.md` 里 4 个臂齐全，每个臂 5 个种子
- [ ] `sweep.png` 存在，且你能指着它说出曲线形状
- [ ] config 已还原，sha256 = `f756b4d0…b9a82`

## 8. 收尾还原

```bash
# 把第 212 行改回 weight=0.5
sed -i '212s/weight=.*/weight=0.5,/' repos/mjlab_research/src/mjlab/tasks/tracking/tracking_env_cfg.py
sha256sum repos/mjlab_research/src/mjlab/tasks/tracking/tracking_env_cfg.py   # 应等于 f756b4d0…b9a82
```

## 9. 自测记录（智能体）

2026-11-16 用 64 env x 6 iter 全流程自测（`scratch/lab07-selftest`，gitignored）：`train 1.0` 与 `train 2.5` 各约 1 分钟、校验正确拦住了“没改 weight”与“改错行”的情况；`eval` 两档各 2 种子；`report` 把 selftest 的两档与 `runs/20261116-003-lab06-eval` 的 A/B 合成四臂表，`comparison.md` / `comparison.json` / `sweep.csv` / `sweep.png` 全部写出（selftest 数值无意义，只验证管线）；随后 config 还原并核对 sha256 = `f756b4d0…b9a82`。


## 10. 结果（2026-11-16 实测）

- 主表（中位 cm，5 种子）：root xy 41.52 / 30.86 / 25.72 / 22.95（w0.5 / 1.0 / 1.5 / 2.5），相对上一档 −25.7% / −16.7% / −10.8%；R-MPKPE 6.25 / **5.98** / 6.10 / 6.45；joint_vel_rmse 1.22 / 1.24 / 1.28 / 1.33；左脚滑移 111.9 / 99.6 / 139.7 / 159.5；tilt 4.24 / 4.81 / 4.84 / **4.04**；四臂 0/5 摔倒、完成度 0.999。
- 形状：root **单调下降 + 边际递减**（每翻倍收益减半，地板 ~20 cm）；姿态在 1.0 触底后回升，2.5 时 R-MPKPE +7.9%、滑移 +60%，属“动作变形”而非“站不稳”（tilt 反而改善、训练回合变短）。
- 结论：**M4 用 1.5**（D16）。不做 3.5 档。tilt 不是代价探针。
- 预测判分：Q1 方向对、Q2 错、Q3 对、Q4 规则对。
- 完整判读：`runs/20261116-006-lab07-eval/README.md`。
