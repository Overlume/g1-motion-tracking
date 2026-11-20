# LAB16 边界格判定翻转率：−x 60 会翻 4/20，−y 60 只翻 1/20

**日期**：2026-11-20 11:20–12:11 ｜ **墙钟**：2567 s ≈ 42.8 min（80 次新评估，均值 32 s；rep1 复用 LAB15）｜ **策略**：`model_52998`（M4，weight 1.5，sha256 `fc8f975b…ceed9`）｜ **动作**：`data/processed/fight1_s3_000s_20s.npz`
**规模**：2 格（`nx_j60` / `ny_j60`）× 种子 101–120 × 3 遍（rep1 = LAB15 的 20 种子结果，rep2/rep3 本次新跑）= 120 次判定
**命令**：`bash scripts/lab16_fliprate.sh run`（内部复用 `scripts/lab12_recovery.sh run`；统计由 `scripts/lab16_report.py` 写 `fliprate.{md,json}` + `outputs/figures/lab16_flipmatrix.png`）

## 0. 目的

LAB14 证明「同配置重跑，判定层零变化」，但那只覆盖**远离判据**的样本（−y 60/80，5 种子）。
LAB15 把边界格加到 20 种子时，101–105 子集里 `nx_j60` seed102 由 transient 翻成 down（D27）。
于是问题变成：**边界格的单次分类到底有多不可复现？** 这决定 LAB15 的 `6/20` 该写成单点还是区间。

采用 3 遍 × 20 种子的设计：每格 20 个种子里，数「至少有一遍与其它遍不同」的种子数（二进制：down ↔ 非 down；三分类：stable/transient/down）。

## 1. 结果

### 1.1 每格 × 每遍

| 格 | 遍 | n | stable | transient | down | 倒地种子 |
| --- | --- | --- | --- | --- | --- | --- |
| `nx_j60` | rep1（LAB15） | 20 | 0 | 14 | 6 | 102, 104, 111, 112, 113, 119 |
| `nx_j60` | rep2 | 20 | 0 | 14 | 6 | 104, 111, 112, 113, 116, 119 |
| `nx_j60` | rep3 | 20 | 0 | 15 | 5 | 104, 111, 112, 115, 119 |
| `ny_j60` | rep1（LAB15） | 20 | 0 | 1 | 19 | 除 102 外全部 |
| `ny_j60` | rep2 | 20 | 0 | 2 | 18 | 除 102、116 外全部 |
| `ny_j60` | rep3 | 20 | 0 | 2 | 18 | 除 102、116 外全部 |

### 1.2 翻转统计

| 格 | 完整种子 | 二进制翻转 | 95% CI | 三分类翻转 | 逐遍 k | k 极差 |
| --- | --- | --- | --- | --- | --- | --- |
| `nx_j60` | 20 | **4** | 20% [6%, 44%] | 4 | 6 / 6 / 5 | 1 |
| `ny_j60` | 20 | **1** | 5% [0%, 25%] | 1 | 19 / 18 / 18 | 1 |
| 合计 | 40 | 5 | 12% [4%, 27%] | 5 | — | — |

翻转明细（rep1 -> rep2 -> rep3）：

- `nx_j60` seed 102：down -> transient -> transient（**LAB15 的那一遍才是离群**）
- `nx_j60` seed 113：down -> down -> transient
- `nx_j60` seed 115：transient -> transient -> down
- `nx_j60` seed 116：transient -> down -> transient
- `ny_j60` seed 116：down -> transient -> transient

### 1.3 101–105 子集复核（与 LAB12 对齐）

| 格 | rep1 | rep2 | rep3 | LAB12 基线 |
| --- | --- | --- | --- | --- |
| `nx_j60` | 2/5 | **1/5** | **1/5** | 1/5 |
| `ny_j60` | 4/5 | 4/5 | 4/5 | 4/5 |

→ rep2/rep3 回到 LAB12 的 1/5：**LAB15 的 2/5 是那一次运行的离群**，不是环境变化（三遍 ckpt sha 一致，见各遍 `params.txt`）。

### 1.4 翻转的机制：两种，不是一种

判据裕度 `r = max(end_drop / 0.05 m, end_tiltex / 8 deg)`：`r ≥ 1` 就该判 down（或直接 `fell` 命中）。

| 种子 | rep1 | rep2 | rep3 | 骨盆最低 (m) | 读法 |
| --- | --- | --- | --- | --- | --- |
| `nx_j60` 102 | r=1.20 | r=0.72 | r=0.77 | 0.442 / 0.456 / 0.454 | **判据擦边**：都没摔，一遍的末尾下移/倾角刚跨过恢复阈值 |
| `nx_j60` 113 | 9.72(fell) | 9.72(fell) | r=0.84 | 0.133 / 0.133 / 0.463 | **双稳态**：一遍摔到 0.13 m，另一遍根本没崩（0.46 m）后恢复 |
| `nx_j60` 115 | r=0.80 | r=0.74 | 7.36(fell) | 0.455 / 0.453 / 0.117 | 同上 |
| `nx_j60` 116 | r=0.78 | 8.75(fell) | r=0.78 | 0.460 / 0.069 / 0.460 | 同上 |
| `ny_j60` 116 | 8.55(fell) | r=0.84 | r=0.56 | 0.082 / 0.429 / 0.433 | 同上 |

**结论**：翻转的来源不是单一「判据阈值抖动」——除 seed102 属真正的擦边外，其余 4 例是**同一种子在不同进程里落到不同吸引子**
（一遍真摔到骨盆 0.06–0.13 m，另一遍全程没崩到 0.43–0.48 m）。这类双稳态不是把判据调宽就能消掉的。

稳定核心（三遍一致）：`nx_j60` 必倒 {104, 111, 112, 119}、必不倒 12 个；`ny_j60` 必倒 18 个、必不倒 {102}。
所以更准确的写法是 **k = 4 + f（f ∈ {1,2}）/20** 与 **k = 18 + 1 /20**，而不是「6/20」「19/20」这种单点。

## 2. 结论

1. **−x 60 N·s 的倒地数不是单点**：三遍 6/6/5（核心 4 + 1–2 个摆动种子），20 个种子里 4 个会翻（20%，CI 6–44%）。
   报告里应写「20 种子下约 5–6/20（25–30%），含 4 个会在倒/不倒之间摆的种子」，不写「6/20 = 30%」。
2. **−y 60 N·s 基本稳定**：三遍 19/18/18，只有 seed116 翻（1/20，5%，CI 0–25%）；102 三遍都是 transient。写作「18–19/20（90–95%）」。
3. **两向差异对翻转稳健**：LAB15 的 p = 3.9e-5 用区间端点重算仍显著（5/20 vs 19/20 **p = 1.0e-5**；6/20 vs 18/20 p = 2.4e-4）。
   「−y 在 60 N·s 下比 −x 更脆」不因翻转而失效。
4. **LAB15 的 6/20 偏高约 1 个种子**：101–105 子集 rep2/rep3 都回到 1/5（与 LAB12 一致），5 种子层面的历史值因此复现；
   LAB15 rep1 的 2/5 是那一次运行的结果。
5. **翻转率是「≥1 次」的观测率**（3 遍只采样 3 次），按种子计、不按 120 次计；pooled 5/40 = 12%（CI 4–27%）。

## 3. 局限

- 仍只有单一施加时刻（8.0 s）、单一波形（0.2 s 方波）、单一作用点（torso_link）、单一 ckpt（`model_52998`）。
- 3 遍只能给翻转率的下界；要估「每遍独立的翻转概率」需要更多遍（预计 10 遍以上）。
- 只测了两格的 60 N·s；80 N·s 两格在 LAB15 是 20/20，未做翻转率（预期更低但未测）。
- 双稳态的物理来源（步态相位？接触摩擦？初始 1e-3 差异的放大？）未定位，只观测到「摔/不摔」两种结局。

## 4. 文件清单

- `rep2/`、`rep3/` — 本次新跑的两遍，每遍 `<cell>/seedNNN/{metrics.csv,summary.json,eval.log}`（逐次文件不入库）
- `fliprate.md`、`fliprate.json` — 翻转率统计（每格×每遍、逐种子矩阵、翻转明细、判定）
- `params.txt`、`env_versions.txt`、`wallclock_s.txt`（各遍）— 配置快照与逐次耗时
- `outputs/figures/lab16_flipmatrix.png` — 种子 × 遍的判定矩阵
- `run.log` — 本次运行的完整日志

## 5. 复现

```bash
cd /home/overlume/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
bash scripts/lab16_fliprate.sh run      # 新跑 2 遍 + 判分 + 翻转率统计 + 图（约 40-50 min）
bash scripts/lab16_fliprate.sh show     # 只看统计表
bash scripts/lab16_fliprate.sh report   # 只重算统计与图
```

单次评估（等价命令，用于抽查某格某种子）：

```bash
MUJOCO_GL=egl environment/.venv/bin/python scripts/eval_tracking.py \
  --checkpoint-file runs/20261117-001-m4-w15-finetune/logs/g1_tracking/2026-11-17_09-25-17_m4-w15/model_52998.pt \
  --motion-file data/processed/fight1_s3_000s_20s.npz --seed 113 --max-steps 0 \
  --ignore-terminations --perturb-body torso_link --perturb-force="-300,0,0" \
  --perturb-start-s 8.0 --perturb-duration-s 0.2 --out runs/20261120-002-lab16-fliprate/rep2/nx_j60/seed113
```
