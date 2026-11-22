# 20261122-002-transient-audit：判定层审计——k=0 的两种含义

## 1. 目的与假设

**问题**：LAB20b（D35）发现 DR 加强的 −x 60 N·s 是「0/20 倒地、但 20/20 全部踉跄」，所以「倒地数 k」不能单独当鲁棒性证据。
由此产生一个回溯疑问：LAB19 的 +x 80 N·s 也是 0/20 但有 19/20 transient——它是不是同一种「退让」？如果是，D32「+x 最稳」就要加限定。

**假设**：transient（崩了又恢复）至少有两种物理含义：

- **擦边型**：偏离刚过「连续 10 帧 = 崩溃」的门槛（0.2–0.3 s），很快回到参考轨迹，末尾 root 误差接近基线 → 本质是稳的，只是判定阈值擦线；
- **退让型**：被打退后长时间跟丢（≥0.5 s），末尾 root 误差很大，但骨盆没触地、没被判 down → 失败率被 k=0 掩盖的「隐性退化」。

**不跑仿真**：只用已有 run 的 `recovery.json`（每格每种子每遍一行，字段含 cls / longest_dev_s / end_drop / root_err_end_cm / pelvis_z_min）。
数据来源：LAB16（−x/−y 60，M4）、LAB17（−y 45/50，M4）、LAB19（±x/±y 60/80，M4）、LAB19b（+x 100，M4）、LAB20b（−x/−y 60，DR 加强）。

## 2. 方法

1. 沿用 `scripts/lab12_recovery_report.py` 的判定字段，不再重算分类（cls 直接用当时落盘的结论）。
2. 只在 **transient 子集**上算三个派生指标：
   - 推后窗口内最长连续偏离时长（`longest_dev_s`，帧数 × 0.02 s）；
   - 末尾 0.5 s 平均 root 误差（`root_err_end_cm`）；
   - 骨盆最低高度（`pelvis_z_min`，用于确认「没触地」：常态约 0.44 m，倒地是 0.06–0.13 m）。
3. 分类阈值（D36）：**退让型** = 偏离 ≥ 0.5 s **且** root ≥ 30 cm；**擦边型** = 偏离 ≤ 0.3 s。
   - 0.5 s = 25 帧 = 崩溃门槛（10 帧）的 2.5 倍；
   - 30 cm ≈ M4 非倒地基线（18.7 cm）的 1.6 倍，且 x 方向两簇之间有空洞（+x ≤ 28 cm vs DR −x ≥ 87 cm），阈值移动不改变结论。
4. 脚本：`scripts/lab_transient_audit.py`（输出 audit.md/json）、`scripts/lab_transient_audit_plot.py`（散点图）。

## 3. 结果

| 权重 | 格 | down | transient | 偏离中位 (s) | 末尾 root 中位 (cm) | 退让型 | 擦边型 |
|---|---|---|---|---|---|---|---|
| M4 | +x 60 | 0/60 | 0 | — | — | 0 | 0 |
| M4 | +x 80 | 0/60 | 57 | 0.20 | **4.5** | 0 | **57** |
| M4 | +x 100 | 0/60 | 37 | 0.20 | 20.7 | 0 | 33 |
| M4 | −x 60 | 11/40 | 29 | 0.30 | 18.2 | 0 | 0 |
| M4 | −y 60 | 36/40 | 4 | 1.08 | 92.3 | 4 | 0 |
| M4 | −y 45 | 34/60 | 26 | 0.86 | 22.1 | 8 | 0 |
| M4 | −y 50 | 45/60 | 15 | 0.80 | 12.6 | 0 | 0 |
| M4 | +y 60 / 80 | 60/60、60/60 | 0 | — | — | 0 | 0 |
| DR 加强 | **−x 60** | **0/60** | **60** | **0.76** | **93.4** | **60** | 0 |
| DR 加强 | −y 60 | 59/60 | 1 | 1.24 | 132.5 | 1 | 0 |

补充明细：

- **+x 80 的 57 次 transient 偏离帧数**：9 帧 ×3（这 3 次其实是 stable，不计入）、10 帧 ×31、11 帧 ×18、12 帧 ×5、13 帧 ×3 → 几乎全在判据线附近；末尾 root 误差全部 < 10 cm（中位 4.5 cm、基线 2.9 cm）。
- **+x 100 的 37 次 transient 偏离**：0.2 s ×21、0.22 s ×8、0.28 s ×3、0.3 s ×1、0.32 s ×1、0.4 s ×3 → 33 次 ≤0.3 s，4 次超；root 中位 20.7 cm、最大 28.4 cm，0 次 ≥30 cm。
- **DR −x 60 的 60 次 transient**：偏离 0.72–0.90 s（36 帧起）、root 86.8–108.7 cm、骨盆最低 0.447–0.470 m → 全部退让型。
- **M4 −x 60 的 29 次 transient**：偏离 0.26–0.50 s、root 13.8–25.8 cm → 0 次退让，与 DR 版差 5 倍 root 误差。

图：`outputs/figures/lab_transient_audit.png`（x = 偏离时长，y = 末尾 root；虚线 = 0.5 s / 30 cm，右上那团就是退让）。

## 4. 判读

1. **D32「+x 最稳」成立**，但要补一句限定：px_j80 的 57/60 transient 是判定门槛擦边（0.20–0.26 s、root 4.5 cm），不是跟踪丢失。px_j60 更干净（60/60 stable、root 2.9 cm）。
2. **D35 的「−x 收益」要补量化**：DR 加强的 −x 60 是 **60/60 全退让**；M4 自己的 −x transient 只有 18.2 cm。也就是说「失败率 30% → 0%」背后的物理变化是「从真摔变成被打退 0.76 s、跟丢 93 cm」。
3. **+x 100 是中间态**：37 次 transient 里有 4 次偏离超过 0.3 s，root 中位 20 cm 但没到 30 cm → 轻度跟踪损失，按 D36 判据不算退让。这也解释了 LAB19b「stable 份额非单调」的观察：判定层口径在 80→100 之间变了性质，不宜当能力指标读。
4. **y 方向本来就偏退让**：−y 60 的 4 次 transient 全退让；−y 45 的 26 次里 8 次退让且 root 很散（7.0–65.9）→ 这解释了 LAB17 里 45 N·s 三遍 11/11/12 的不稳（同一批种子里「退让」与「真稳」混着）。
5. **方法学**：报鲁棒性必须三数齐全——倒地数 k、扰动后 root 误差、transient 退让型占比（D36）。只报 k 会把 93 cm 的退让写成「0 失败」。

## 5. 局限

1. 0.5 s / 30 cm 是**事后选定**的阈值；x 方向两簇之间有空洞（≤28 cm 与 ≥87 cm）所以不敏感，但 y 方向是连续分布（12–145 cm），退让型计数会随阈值移动，只能作定性描述。
2. 分类（cls）直接沿用当时落盘的判定，没有重算；如果当时的判据有 bug，这次审计不会发现。
3. 没有独立的「触地」指标（如骨盆 <0.2 m 帧数）；这里用 pelvis_z_min 近似确认「没触地」，但没有区分「脚着地/躯干着地」。
4. transient 的 root 误差是末尾 0.5 s 平均，不能反映恢复过程中的峰值偏离。
5. 只有单训练 seed、单动作片段，与 REPORT §4 的局限一致。

## 6. 复现

    environment/.venv/bin/python scripts/lab_transient_audit.py --out runs/20261122-002-transient-audit
    environment/.venv/bin/python scripts/lab_transient_audit_plot.py

只读输入：各 run 的 `rep*/recovery.json`；不跑 MuJoCo。

## 7. 文件清单

- `README.md`（本文件）
- `audit.md`、`audit.json`：11 个格 × 每格统计
- `outputs/figures/lab_transient_audit.png`：散点图（在仓库根 outputs 下，不在本目录）
- `scripts/lab_transient_audit.py`、`scripts/lab_transient_audit_plot.py`

## 8. 证据路径

- 结论 1（+x 真稳）：本目录 audit.md 的 px_j60 / px_j80 行；REPORT §2.5.14 第 1 条
- 结论 2（DR −x 全退让）：audit.md 的 LAB20b nx_j60 行；`runs/20261122-001-lab20b-perturb/recovery.json`（rep1–3）
- 结论 3（+x 100 中间态）：audit.md 的 px_j100 行
- 口径：`docs/DECISIONS.md` D36；`docs/REPORT.md` §2.5.14
