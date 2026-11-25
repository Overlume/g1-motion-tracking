# LAB25 摩擦上界收敛：跨训练 seed 复测 + f30 的 −y 扩测

> 目标（一句话）：回答 LAB24 留下的两个问题——**f20（0.2–2.0）的 65% 失败率是一次训练不稳，还是「摩擦上界非单调」的真效应？3.0 档会不会也改 −y？**
>
> 全部命令都能直接粘贴；路径写死，**不要换成尖括号**（`< >` 在 bash 里是重定向，2026-11-20 踩过）。

## 0. 先看清楚要回答什么

### 0.1 已知（都不需要重测）

| 事实 | 数字 | 来源 |
| --- | --- | --- |
| M4（默认 DR 0.3–1.2），nx_j60 | 逐遍 **6/6/5**，池化 17/60（28%），非倒地 root 中位 18.2 cm | LAB15/LAB16、D26/D28 |
| f14（0.2–1.4，LAB23 fric ckpt） | **2/2/1**（5/60，8%），root 中位 82.3，55/55 漂移 | LAB24、D38 |
| f20（0.2–2.0，训练 seed 0） | **10/14/15**（39/60，**65%**），root 中位 68.7 | LAB24、D38 |
| f30（0.2–3.0，训练 seed 0） | **0/0/0**（0/60），root 中位 **18.7**（p75 38.6、max 61.8），21/60 漂 ≥30 cm | LAB24、D38 |
| M4，ny_j60（300 N） | 逐遍 **19/18/18** | LAB16 |
| M4，ny_j45（225 N） | 逐遍 **11/11/12**（58%，[32%,81%]） | LAB17、D30 |
| 「−y 对 DR 不敏感」 | LAB23 三臂 ny 全 20/20；LAB20b 加强 DR 为 20/19/20（p=1） | D35/D37 |

### 0.2 为什么先做这两件事

- LAB24 的 65% 只有**一个训练 seed**。单 seed 的 65% 可能来自「这一次训练刚好崩了」，也可能说明 2.0 这个上界本身是个坏点。**不复测就不能写「非单调」**（D38 决定 3）。
- 3.0 档是唯一满足 D37 判据（k ≤10% 且非倒地 root 中位 ≤40 cm）的臂，但它只在 −x 上验证过。如果 3.0 档连 −y 也改得动，「摩擦档只对 −x 有效」就要改写；如果改不动，这条结论反而被加固。**零训练就能测，性价比最高**。
- 不继续加档（2.5 / 3.5）的理由：现在连 2.0 是「坏点」还是「方差」都没分清，加档只会多一个不可解释的点。

### 0.3 四个臂、跑多少次、多久

| 臂 | 做什么 | 训练 | 评估 | 预计 |
| --- | --- | --- | --- | --- |
| `f30ny60` | f30 现有 ckpt × **ny_j60（300 N）** × 20 种子 × 3 遍 | 0 | 60 次 | ≈36 min |
| `f30ny45` | f30 现有 ckpt × **ny_j45（225 N）** × 20 种子 × 3 遍 | 0 | 60 次 | ≈36 min |
| `f20s1` | 重训 0.2–2.0，**训练 seed 1** → nx_j60 × 3 遍 | ≈1 h 40 | 60 次 | ≈2 h 16 |
| `f30s1` | 重训 0.2–3.0，**训练 seed 1** → nx_j60 × 3 遍 | ≈1 h 40 | 60 次 | ≈2 h 16 |

**合计 240 次评估（≈2 h 25）+ 2 次训练（≈3 h 20）≈ 5 h 50。** 中途可停，重跑同一条命令自动续训/跳过已完成。

成本口径：单次评估 ≈36 s（LAB24 实测 180 次 1 h 47）；单次训练 1 h 35–1 h 41（LAB20/23/24 三次实测）。

### 0.4 三种结果怎么读（先想清楚，再看数）

先记住三条判定线（来自 D38 决定 3/4，本讲义不临时改）：

- **f20 seed1**：每遍 **≥9/20** = 与 seed0 同向（65% 不是偶然）；**≤4/20** = seed0 那次训练不稳；**5–8/20** = 灰区，只写区间。
- **f30 seed1**：k **≤2/20 且** root 中位 ≤40 cm = 3.0 档收益复现；**≥5/20** = 3.0 档也不稳。
- **ny_j60**：**≤10/20** = 3.0 档也改 −y（新发现，与 D35/D37 冲突，要写清）；**11–15/20** = 部分；**≥16/20** = 不变（加固「只对 −x 有效」）。

| 情形 | 你会看到 | 怎么写结论 |
| --- | --- | --- |
| **A（预期）** | f20 seed1 ≥9/20，f30 seed1 ≤2/20 | 「2.0 在两个训练 seed 上都是 50–65%，3.0 在两个上都是 0–3% → 非单调不是单次训练噪声」；仍不能叫定理（只有 2 个训练 seed × 2 个点） |
| **B** | f20 seed1 ≤4/20 | 「同一 DR 设置的两次训练给出 65% 与 X% → 训练侧方差 ≥ 9 个种子，单 seed 的 DR 结论不可引用」；按 §6 加 seed2 |
| **C** | f30 seed1 ≥5/20 | 「3.0 档的 0/60 也可能是一次好运」→ 主候选降级为「2.0/3.0 都不稳」，下一步必须先量化训练侧方差 |

ny 的三种情形与上表独立，可同时出现。

### 0.5 什么时候可以停

240 次评估全部有 `recovery.json` + 两张表出图，就可以停。**不要**在中途补跑第 4 遍、也不要临时加 2.5 档——那会把「一次只改一个因素」破坏掉（AGENTS.md）。

## 1. 协议（写死，不许改）

| 项 | 值 |
| --- | --- |
| 热启动权重 | `scratch/model_49999.pt`（sha256 前 8 位 `f1bc404f`） |
| 训练 | 3000 iter、4096 envs、S3 0–20 s（`data/processed/fight1_s3_000s_20s.npz`）、目标 iter **52998**、com/bias 回 mjlab 默认（±0.025/0.05/0.05 m、±0.01 rad） |
| 训练 seed | f20s1 / f30s1 都 = **1**（seed0 的结果已在 LAB24） |
| 摩擦 | f20s1 = (0.2, 2.0)；f30s1 = (0.2, 3.0) |
| f30 ckpt | `runs/20261123-001-lab24-fric-scan/arms/f30/.../model_52998.pt`，sha256 `1119f986…455ba0`（脚本启动时校验，不符直接报错） |
| 评估 | nx_j60 = 300 N、ny_j60 = 300 N、ny_j45 = 225 N（力 = 冲量 / 0.2 s）；`torso_link` 世界系；终止关；8.0 s 起推 0.2 s；推后看 3 s |
| 评估种子 | 101–120（每个格 × 每个臂），每臂 3 遍 |

## 2. 七个步骤

### 步骤 0：开工核对（1 min）

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
ls -la scratch/model_49999.pt
sha256sum scratch/model_49999.pt | cut -c1-8
git log --oneline -1
git status --short
```

期望：显存占用 <500 MiB；权重存在；sha 前 8 位 `f1bc404f`；最新提交 ≥ `6f35ed8`；工作树干净（除你刚写的预测）。

### 步骤 1：写预测（5 min，先写再看数）

打开 `docs/learning/LAB25-prediction.md`，把每题「你的答案」填好。不要跑完再改——判分看的是预测，不是事后解释。

### 步骤 2：建 run 目录 + README 骨架（3 min）

```bash
RUN=runs/20261124-001-lab25-converge
mkdir -p "$RUN"
cat > "$RUN/README.md" <<'EOF'
# LAB25 摩擦上界收敛（跨训练 seed + f30 的 −y 扩测）

- 目的：判定 LAB24 的非单调（f20 = 65%）是单次训练不稳还是真效应；并测 3.0 档是否也改 −y。
- 假设：（写完预测后填一句）
- 配置：f30 复用 LAB24 ckpt（sha 1119f986…455ba0）× ny_j60 / ny_j45；f20s1 / f30s1 重训（训练 seed 1，3000 iter、4096 envs、热启动 model_49999）→ nx_j60。
- 命令：见 docs/learning/LAB25-converge.md §2。
- 结果：（跑完填：两张表 + 判定属于 §0.4 哪一种）
- 下一步：（填）
EOF
ls "$RUN"
ls runs/*.md runs/*.json
```

自检：最后一条 `ls runs/*.md runs/*.json` **必须无输出**（有输出说明 RUN 写错、污染了 runs 根目录）。

### 步骤 3：训练 + 评估 + 出表（一条命令，≈5 h 50）

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
LAB25_RUN=runs/20261124-001-lab25-converge bash scripts/lab25_dr_seed.sh run
```

这条命令依次做：训练 f20s1 → 训练 f30s1 → 评估 f30ny60 → f30ny45 → f20s1 → f30s1 → 出表出图。
**中途 Ctrl-C、关机、断电都没关系**：重跑同一条命令会

- 从每个训练臂的最新 ckpt 续到 iter 52998（脚本按剩余迭代自动算，最多丢 500 iter ≈ 16 min）；
- 跳过已经有 `recovery.json` 的遍、已经有 `summary.json` 的种子。

训练中随时查进度（另开一个终端，**不要**在训练窗口里跑）：

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
LAB25_RUN=runs/20261124-001-lab25-converge bash scripts/lab25_dr_seed.sh status
```

期望：f30ny60 / f30ny45 两行显示 `1119f986…`、iter 52998、reps 0/3（还没评估）；f20s1 / f30s1 训练中 iter 递增。训完后两臂都应显示 **52998**。

训练完成后核对 DR（只该动摩擦）：

```bash
grep -A3 "foot_friction" runs/20261124-001-lab25-converge/arms/f20s1/logs/g1_tracking/*/params/env.yaml | head -6
grep -A3 "foot_friction" runs/20261124-001-lab25-converge/arms/f30s1/logs/g1_tracking/*/params/env.yaml | head -6
```

期望：ranges 分别是 (0.2, 2.0) 与 (0.2, 3.0)；同一文件里的 base_com 是 ±0.025/0.05/0.05、encoder_bias 是 ±0.01。

### 步骤 4：评估进度自检（中间看一眼，不用等）

如果上一步被打断，只补评估：

```bash
LAB25_RUN=runs/20261124-001-lab25-converge bash scripts/lab25_dr_seed.sh eval
```

进度：

```bash
ls -d runs/20261124-001-lab25-converge/arms/*/perturb/rep*/recovery.json | wc -l
```

期望最终 **12**（4 臂 × 3 遍）。想先看最有信息量的一个臂（≈36 min）：

```bash
LAB25_RUN=runs/20261124-001-lab25-converge LAB25_ARMS="f30ny60" bash scripts/lab25_dr_seed.sh eval
```

### 步骤 5：出表 + 图 + 读数顺序（2 min）

```bash
LAB25_RUN=runs/20261124-001-lab25-converge bash scripts/lab25_dr_seed.sh report
sed -n '1,40p' runs/20261124-001-lab25-converge/lab25_converge.md
```

**读数顺序（按这个来，别先看结论）**：

1. **表 1 的 f30 两行**：seed0 = 0/0/0、seed1 = ?/?/?。按 §0.4 判定线，先给「复现 / 不复现」。
2. **表 1 的 f20 两行**：seed0 = 10/14/15、seed1 = ?。落 A / B / 灰区。
3. **表 2**：ny_j60 的 f30 vs M4（19/18/18）、ny_j45 的 f30 vs M4（11/11/12）；再按 §0.4 的 ny 判定线给「改 / 部分 / 不变」。
4. **逐遍极差**：任何一遍与另两遍差 ≥4 个种子，结论里必须写「逐遍极差 X」；池化数只是中心估计。
5. 最后写一句结论，必须是 §0.4 三种情形之一，**不许出现「略有改善」「基本一致」**。

### 步骤 6：文档同步（15–20 min，九处）

1. `runs/20261124-001-lab25-converge/README.md`：目的 / 协议核对 / 时间线 / 两张表 / 判读 / 局限 / 下一步。
2. `docs/REPORT.md`：新增 `### 2.5.17`（插在 `## 3. 关键发现` 之前）；§3 新增第 13 条；§4 新增第 13 条。
3. `docs/DECISIONS.md`：新增 **D39**（依据 / 实测 / 决定），并在 D38 末尾加一行补注。
4. `docs/STATE.md`：头部「最后更新」与阶段句；末尾新增 `## LAB25 结果…` 节。
5. `docs/PLAN.md`：LAB25 那一条标 `[x]` 并补「实测」。
6. `docs/ARTIFACT_INDEX.md`：新增 `## 2026-11-24（LAB25 摩擦上界收敛）` 节。
7. `README.md`：结果表加「摩擦上界收敛」一行。
8. `docs/learning/LAB25-converge.md`：把 §7 的「待填」换成实测。
9. `docs/learning/LAB25-prediction.md`：判分（参考回答写在你每题答案的下方）。

> 也可以跑完把两张表贴给我，我按 LAB17–LAB24 的做法代填 1–9（之前的收尾都是我做的），你只负责跑和写预测。

### 步骤 7：提交并 push（5 min）

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
RUN=runs/20261124-001-lab25-converge
git add README.md docs/ARTIFACT_INDEX.md docs/DECISIONS.md docs/PLAN.md docs/REPORT.md docs/STATE.md \
  docs/learning/LAB25-converge.md docs/learning/LAB25-prediction.md \
  scripts/lab25_dr_seed.sh scripts/lab25_report.py \
  outputs/figures/lab25_converge.png \
  "$RUN/README.md" "$RUN/lab25_converge.md" "$RUN/lab25_converge.json"
git commit -m "feat(lab25): 摩擦上界收敛——跨训练 seed + f30 的 -y 扩测；D39"
git log --oneline -1
```

推送（Windows 上的 git credential manager 要被显式指定，见 HANDOFF §7.1）：

```bash
cat > /tmp/push_lab25.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd /home/overlume/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
GIT_TERMINAL_PROMPT=0 git -c 'credential.helper=!"/mnt/c/Program Files/Git/mingw64/bin/git-credential-manager.exe"' push origin main
EOF
bash /tmp/push_lab25.sh
```

## 3. 通过判定（全满足才算做完）

1. `params.txt` 有四个臂、`f30_sha256: 1119f986…455ba0`、`target_iter: 52998`。
2. `status` 里两个训练臂 ckpt_iter = **52998**；`env.yaml` 的摩擦分别是 (0.2, 2.0) / (0.2, 3.0)，com/bias 为默认。
3. 四个臂各有 3 遍 `recovery.json`，每遍 20 个种子（合计 **240** 份 `summary.json`）。
4. 表 1 / 表 2 三数齐全：逐遍 k、池化 k/n、非倒地 root 中位（+ 漂移计数）。
5. 图 `outputs/figures/lab25_converge.png` 两面板正常（左：nx 的 seed0 vs seed1；右：−y 的 M4 vs f30）。
6. 判读按 §0.4 写死：A / B / C / 灰区 之一 + ny 的「改 / 部分 / 不变」之一，没有模糊句。
7. 九处文档同步完成（或已交给代填）。
8. 提交 + push 完成，`git status` 干净。

## 4. 常见问题

| 现象 | 原因 | 怎么办 |
| --- | --- | --- |
| `f30 ckpt sha 不符` | `LAB25_F30_DIR` 指到了别处（例如 pretrained 副本） | 确认它指向 `runs/20261123-001-lab24-fric-scan/arms/f30`；真要换权重才设 `LAB25_F30_SHA` |
| 训练中途断了 | 正常 | 重跑步骤 3 的同一条命令，自动续训 |
| `status` 显示 `（未训练）` 但评估开始报错 | 训练臂没到 52998 | 先跑 `train` 子命令，再 `eval` |
| 显存不足 / 进程被 kill | 还有别的训练在跑 | `nvidia-smi` 查占用，只保留一个训练 |
| 评估特别慢（>60 s/次） | GPU 被占 / 散热降频 | 等散热；不要改 envs 或 seeds |
| 表里某臂是 `—` | 该臂还没评估完 | 正常，跑完 12 个 recovery.json 再出表 |
| 报告 root 中位与 REPORT 的 18.2 差 0.5 cm | 报告是「三遍池化」口径 | 不算矛盾；引用时写清口径 |
| 预测写错了想改 | 规则不允许 | 保留原答案，判分后参考回答会说明差在哪 |

## 5. 六个坑

1. **尖括号不是占位符**：讲义里 `<...>` 出现时是命令语法（重定向），路径都写死了；自己改成 `runs/<日期>-...` 会得到 `-bash: 20261124: No such file or directory`（2026-11-20 踩过）。
2. **训练 seed ≠ 评估 seed**：这轮训练用的是 seed **1**（seed0 已在 LAB24），评估种子仍是 **101–120**。两者不要混。
3. **不要手动改 TARGET**：续训的 `--iters` 由脚本按「目标 52998 − 当前 iter + 1」自动算；手动改会重复计迭代。
4. **别把 pretrained 副本当训练产物**：选 ckpt 时永远排除 `/pretrained/`；本轮 f30 复用臂用 sha 校验挡这一刀（LAB20 事故）。
5. **逐遍漂移是真的**：LAB24 的 f20 是 10→14→15；本轮判定必须带逐遍 k，别只报池化。
6. **单训练 seed 的结论不能升级**：即使本轮 A 情形成立，也只有 2 个训练 seed × 2 个点，写「不是单次训练噪声」可以，写「非单调定理」不行。

## 6. 分支预案

| 触发 | 下一步 | 成本 |
| --- | --- | --- |
| f20 seed1 **≤4/20**（情形 B） | 新 run `runs/20261124-002-lab25-f20s2`，命令：`LAB25_RUN=runs/20261124-002-lab25-f20s2 LAB25_ARMS="f20s2" bash scripts/lab25_dr_seed.sh run`（训练 seed 2、0.2–2.0、nx_j60 × 3 遍） | ≈2 h 16 |
| f20 seed1 **5–8/20**（灰区） | 先出表写区间；再按上一条加 seed2，把「2.0 档」的训练侧分布做成 3 个点 | ≈2 h 16 |
| f30 seed1 **≥5/20**（情形 C） | 先按「3.0 也不稳」收尾；再加 `LAB25_ARMS="f30s2"` 训 seed2，确认是「3.0 档不稳」还是「两次都运气差」 | ≈2 h 16 |
| ny_j60 **≤10/20**（3.0 也改 −y） | 收尾时要改 D35/D37 的表述：先复测 ny_j45 的 3 遍（已在本轮里）再下结论；并在 D39 里写「−y 的 DR 无效只在默认摩擦下成立」 | 0（数据已有） |
| 训练 loss 爆炸 / 数值 NaN | 停，先按 LAB20 的做法核对 env.yaml 与 warm sha，再重跑；不要改学习率 | 30 min |

## 7. 结果（2026-11-24 实测，runs/20261124-001-lab25-converge）

| 臂 | k（逐遍） | 池化 | 非倒地 root 中位 | 漂移/退让 | vs M4 p（逐遍） |
| --- | --- | --- | --- | --- | --- |
| f30ny60 | 20/20/20 | 60/60（100%） | —（全倒） | — | 1.000/0.487/0.487 |
| f30ny45 | **0/0/0** | 0/60（0%） | 87.2 cm | 退让 49、漂移 11 | 1.45e-4/1.45e-4/4.51e-5 |
| f20s1 | 5/5/4 | 14/60（23%） | 71.4 cm | 漂移 46、退让 0 | 1.000/1.000/1.000 |
| f30s1 | 1/3/3 | 7/60（12%） | 27.9 cm | 漂移 10、退让 14、擦边 0 | 0.091/0.451/0.695 |
| 对照 M4 | 6/6/5 | 17/60（28%） | 18.2 cm | 擦边 34、stable 9 | — |
| 对照 f20 seed0（LAB24） | 10/14/15 | 39/60（65%） | 68.7 cm | 漂移 21 | — |
| 对照 f30 seed0（LAB24） | 0/0/0 | 0/60（0%） | 18.7 cm | 漂移 21、擦边 39 | 0.020/0.020/0.047 |

**sha 核对**：f30 复用臂的 `params.txt` 三处都写着 `1119f986…455ba0`（LAB24 f30 的 model_52998）；f20s1 = `6a1323bb…eefe4e4d`、f30s1 = `86940da6…5af873`，与各自训练目录里的 ckpt 一致。训练 `agent.yaml` 的 `seed: 1`，`env.yaml` 的 friction 分别是 (0.2, 2.0) 与 (0.2, 3.0)，com/bias 为默认。

**属于 §0.4 的哪一种**：**B 与 C 各占一半，并多出讲义没列的第 4 种。**

1. f20 seed1 = 5/5/4 → 落在「≤4/20 = seed0 孤立」与「5–8 = 灰区」之间，实际按池化 23% 读：**65% 是 seed0 的离群**；两个 seed 唯一一致的是跟踪代价（root 68.7/71.4，非倒地全部漂移）。
2. f30 seed1 = 1/3/3 → rep1 达标、rep2/rep3 各 3/20 越线 → **部分复现**（方向对、幅度不稳），且与 M4 的差异不再显著（p=0.09–0.70）。
3. −y：ny_j60 仍 20/20（60 N·s 饱和），ny_j45 **0/60**（M4 34/60，p<1.5e-4）但 49/60 是退让型、root 87.2 cm → **抬摩擦上界确实改 −y，只是被 60 N·s 饱和掩盖**；「−y 无收益」「DR 只对 −x 有效」必须限定在 60 N·s。

**结论一句**：摩擦上界的收益对训练 seed 不稳（倒地率 65%→23%、0%→12%），可复现的是「不倒地但退让」；3.0 档把 −y 的失败率从 57% 压到 0（45 N·s，退让型），60 N·s 仍是 100%。

**下一步（写进 D39）**：档位比较至少 2 个训练 seed 并按 seed 报区间；−y 交点定位到 45–60 之间（补 50/55）；评估口径统一为「k + 非倒地 root 池化中位 + 三型占比」。
