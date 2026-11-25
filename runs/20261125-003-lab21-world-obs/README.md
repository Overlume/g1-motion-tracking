# LAB21 actor 观测追加世界系根位置/速度（P7）

| 项 | 值 |
|---|---|
| 日期 | 2026-11-25 15:25:37 – 17:39:57（+08:00） |
| 目的 | 只改 actor 输入（尾部 +6 维绝对位姿），其余全同 M4，检验「相对位姿观测有盲点」这个假设 |
| 假设 | 机器人与参考同步漂移时 anchor 相对误差为 0，策略察觉不到自己在世界里的偏移；追加世界系根位置/速度后应拿到独立基准 → held-out 应改善 |
| 结论 | **假设被否（C）**：held-out 0.100 → **0.018**、存活 1.98 s → **0.34 s**；新增的 3 个位置维度权重范数只有老维度的 7–13%，模型实际只用了世界系速度 |

## 1 协议核对

| 项 | 期望 | 实测 |
|---|---|---|
| ckpt | `runs/20261125-003-lab21-world-obs/logs/g1_tracking/2026-11-25_15-25-37_lab21-world-obs/model_52998.pt` | iter 52998 ✓ |
| actor 第一层 | (512, 166)（M4 是 160） | (512, 166) ✓ |
| obs_normalizer mean | (1, 166) | (1, 166) ✓ |
| actor terms | 末尾多 `root_pos_w` / `root_vel_w` | env.yaml 两条都在（`obs_world.root_pos_w` / `obs_world.root_vel_w`）✓ |
| 训练动作 | `data/processed/fight1_s3_000s_20s.npz`（sha 965f2c4b…b66c66） | 同 ✓ |
| 热启动 | `scratch/model_49999.pt`（sha f1bc404f…f22319），第一层与优化器动量从 160 扩到 166 | ✓（日志「actor 第一层 160 -> 166」） |
| 其余口径 | 默认 DR（摩擦 0.3–1.2、质心 ±2.5/±5/±5 cm、编码器 ±0.01）、4096 envs、seed 0、3000 iter | 同 ✓ |

## 2 时间与成本

- 训练 **1 h 44 min**（15:25:37 → 17:09:54），一次跑完、无续训。
- 评估 **18.8 min**（17:21:06 → 17:39:57），30 次 ≈ 37.6 s/次（3 窗口 × 10 种子）。
- 合计 ≈ **2 h 03 min**；run 目录 66 MB（train.log 未入库、已删）。

## 3 结果（3 窗口 × 10 种子 101–110，终止开）

| 窗口 | 策略 | 完成度 | MPKPE cm | R-MPKPE cm | root err cm |
|---|---|---|---|---|---|
| 训练片段（S3 0 s） | M4 | 0.999 ± 0.000 | 5.65 | 3.67 | 3.86 |
| 训练片段（S3 0 s） | LAB21 | 0.999 ± 0.000 | 6.86 | 3.78 | 5.57 |
| held-out（S3 120 s） | M4 | 0.100 ± 0.002 | 18.71 | 10.29 | 12.88 |
| held-out（S3 120 s） | LAB21 | **0.018 ± 0.000** | 8.01 | 7.97 | 2.44 |
| unseen（S2 134 s） | M4 | 0.025 ± 0.001 | 17.73 | 9.61 | 14.85 |
| unseen（S2 134 s） | LAB21 | 0.038 ± 0.000 | 18.68 | 11.06 | 16.56 |

配对差（LAB21 − M4）：train ±0.000（R-MPKPE +0.12、root +1.71）；held-out **−0.082**（−0.084 ~ −0.078，0 好 / 10 差）；unseen **+0.013**（+0.012 ~ +0.015，10 好 / 0 差，但 R-MPKPE +1.44、root +1.71 都是变差）。

### 3.1 存活时长与终止原因（比完成度更能说明问题）

| 窗口 | 策略 | 最远动作帧 | 折合秒 | 平均 reset 次数 / 999 步 | 终止项 |
|---|---|---|---|---|---|
| 训练片段 | M4 / LAB21 | 998 / 998 | 19.96 / 19.96 | 1 / 1 | motion_end |
| held-out | M4 | 99（96–102） | 1.98 | 23.5 | ee_body_pos |
| held-out | LAB21 | **17（17–18）** | **0.34** | **60.9** | ee_body_pos |
| unseen | M4 | 24（23–26） | 0.48 | 48.6 | ee_body_pos |
| unseen | LAB21 | 38（38–38） | 0.76 | 40.5 | ee_body_pos |

- held-out 的 LAB21 每 0.32 s 就被 `ee_body_pos`（末端身体位置误差）打停一次，999 步里 61 次 → 最远只到第 17 帧。
- **两个策略都不是摔倒**：`fall_detected = false`、终止时骨盆 0.81 m（直立）；所以这不是「加观测把人弄摔了」，是「跟不住参考、被误差终止」。
- unseen 两策略都贴着地板（0.48 s vs 0.76 s），差别只是多撑 0.3 s。

## 4 机制证据（为什么是负结果）

### 4.1 位置维度被模型丢弃（第一层权重列范数）

| 新增维度 | 索引 | 列范数 |
|---|---|---|
| root_pos_w x | 160 | **0.120** |
| root_pos_w y | 161 | **0.124** |
| root_pos_w z | 162 | **0.220** |
| root_vel_w x | 163 | 1.139 |
| root_vel_w y | 164 | 0.917 |
| root_vel_w z | 165 | 1.474 |

老 160 维同口径：中位 **1.677**、p90 2.315、最大 3.163。→ 三个位置维在全 166 维里排**最后三名**（约为老维度中位的 7–13%），网络基本没给绝对位置分配权重；世界系速度则是正常量级，被真正使用了。

### 4.2 新增 6 维的归一化统计是退化的

| 维度 | ckpt 里的 std | 训练数据实测 std |
|---|---|---|
| root_pos_w x / y | 4.502 / 4.504 | 0.311 / 0.582 |
| root_pos_w z | 0.972 | 0.041 |
| root_vel_w x / y / z | 0.974 / 0.977 / 0.973 | 0.354 / 0.413 / 0.240 |

热启动时新维度按 mean=0 / var=1 / std=1 初始化，但 ckpt 里 `obs_normalizer.count` 继承了旧权重的 **5.21e9** → 再训 3000 iter 也只能让统计挪动一丁点，新维度实际上等于**未归一化直接进网络**（z 与速度三维的 std ≈1 就是初始值）。这是本轮实现的一个真问题，见 D42 决定 3。

### 4.3 绝对坐标在单片段训练下只能是「片段标签」

三段参考的世界区域（世界系，米）：s3_000s 均值 xy **(0.219, −0.563)**、s3_120s **(0.437, −1.689)**、s2_134s **(−0.529, −0.793)**。同一个「绝对坐标」在不同片段里指向动作的不同段落，单片段训练时它只可能被学成「我在这一段的哪个位置」的标签。

## 5 判读

1. **判 C（假设被否）**：held-out Δ = −0.082（10/10 更差），存活从 1.98 s 掉到 0.34 s。加原始绝对位姿没有提供「独立基准」，反而把 held-out 直接压到 LAB20（全套加强 DR）的同一个 0.018。
2. **「姿态更准」是短存活假象**：held-out 的 CP-MPKPE 18.71 → 8.01 只是因为只跑了 0.34 s、机器人还站在起始位姿。在共同存活的前 17 帧上对齐比较：LAB21 MPKPE 7.87 / R-MPKPE 7.78 / root 2.51 cm vs M4 8.82 / 8.45 / 2.37 cm → 姿态误差略好、root 略差，**没有实质改善**。
3. **机制假设没实现**：网络没学「世界系定位」，只学了世界系速度（4.1）。
4. **这不等于「观测无用」**：本轮同时叠了三件事——原始绝对坐标、单片段训练、未预热归一化。要证伪「观测缺绝对基准」这个假设，必须把这三件拆开（D42 决定 4）。
5. **train 完全没变**（0.999，10/10 相同，root 3.86 → 5.57 cm 变差）：专精仍在，不是练废。

## 6 预测判分

4.5/8 ≈ 2.8/5（Q1 2/2、Q2 0/2、Q3 0.5/1、Q4 1/2、Q5 1/1），参考回答见 `docs/learning/LAB21-prediction.md`。

## 7 局限

- 单训练 seed（seed 0）、单片段训练：绝对坐标的可迁移性本来就没有测试条件。
- 新维度归一化统计退化（4.2），所以结论只对「按当前实现」成立；没有做「冻结归一化器 / 用数据预热统计」的对照。
- 跨窗口指标不可直接比（存活差 6×），已用匹配视界（前 17 帧）补算，但没有在全部帧上重做。
- 只测了 60 个评估点（3 窗口 × 10 种子），没有叠加 LAB12 抗推协议。

## 8 文件清单

- `logs/g1_tracking/2026-11-25_15-25-37_lab21-world-obs/`：model_50000…52998.pt（7 个）、`params/`（env.yaml / agent.yaml 等）、tfevents、onnx
- `eval/obs/{train,heldout,unseen}/seed101..110/`：metrics.csv（999 行）+ summary.json
- `lab21_obs.md` / `lab21_obs.json`：三窗口表 + 配对差 + 判定（由 `scripts/lab21_report.py` 生成）
- `outputs/figures/lab21_obs.png`：三窗口完成度 + 配对差图

## 9 复现

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
environment/.venv/bin/python scripts/lab21_obs_train.py --run-root runs/20261125-003-lab21-world-obs --mode full --resume
LAB21_RUN=runs/20261125-003-lab21-world-obs bash scripts/lab21_eval.sh run
```

机制复核（本 README 第 4 节的三个数）：

```bash
environment/.venv/bin/python - <<PY
import glob, torch, numpy as np
ck = sorted(glob.glob("runs/20261125-003-lab21-world-obs/logs/g1_tracking/*lab21-world-obs/model_*.pt"))[-1]
a = torch.load(ck, map_location="cpu", weights_only=False)["actor_state_dict"]
w = a["mlp.0.weight"].detach().numpy()
print("位置列范数", np.round(np.linalg.norm(w[:, 160:163], axis=0), 3))
print("速度列范数", np.round(np.linalg.norm(w[:, 163:166], axis=0), 3))
print("老维度中位范数", round(float(np.median(np.linalg.norm(w[:, :160], axis=0))), 3))
print("新维度 std", np.round(a["obs_normalizer._std"].flatten()[160:166].numpy(), 3))
print("count", float(a["obs_normalizer.count"]))
PY
```
