# LAB19 — 四向对称性：+x / +y 在 60 与 80 N·s（4 格 × 20 种子 × 3 遍）

> 讲义版本：2026-11-21。参照 LAB17 / LAB18 的组织方式。
> 这一批是单次最长的一条：**240 次评估 ≈ 2.1 h**，命令已经合成一条，贴一次就能跑完。

## 0. 这个 lab 要达成什么

### 0.1 已经知道什么（同 ckpt、同协议、种子 101–120）

| 方向 | 60 N·s（300 N） | 80 N·s（400 N） | 120 N·s（600 N） |
| --- | --- | --- | --- |
| −x | **5–6/20（25–30%）**，三遍 6/6/5，翻转 4/20（LAB16） | **20/20**（LAB15） | 5/5 倒（LAB12） |
| −y | **18–19/20（90–95%）**，三遍 19/18/18，翻转 1/20（LAB16） | **20/20**（LAB15） | 5/5 倒（LAB12） |
| +x | **完全没测过 20 种子** | 没测过 | 5 种子里 **2 稳 / 3 判 down**（LAB12，600 N 唯一有「稳」的方向） |
| +y | **完全没测过 20 种子** | 没测过 | 5/5 倒（LAB12） |

还有两条独立线索：

- **LAB11 姿态审计（80 次 rollout）**：39 次跑完 / 10 次姿态崩 / 31 次站立误差判罚，**10 次 collapse 全部在 +y**。这不是冲量实验，但说明 +y 是姿态上最脆弱的方向。
- **LAB18 的教训**：翻转可能单向系统化（−x 70 的 4 个翻转全是 rep1 偏倒），所以三遍是必需项，不是保险。

### 0.2 本 lab 的目标

1. **补齐 +x / +y 的 60 与 80 两档**（各 20 种子 × 3 遍），把「两向边界不同」扩成**四向同口径对照**；
2. 回答一个问题：**决定倒地的是「推的方向」还是「推的大小」**——如果 +x/+y 在 60 就接近 −y，说明方向图景不对称但都很脆；如果 +x/+y 在 60 只有 5–6/20，说明「−y 特脆」是个方向特异性事实；
3. 检验 80 N·s 的「20/20」是否只对 −x/−y 成立。

### 0.3 为什么是这 4 格

- 60 与 80 是现有曲线唯一两个「四向都会遇到」的档位（45/50 只有 −y 有、70 只有 −x 有）；
- +x / +y 在这两档是**完全空白**，比补 65 / 55 之类的中间点更值钱；
- 4 格 × 20 种子 × 3 遍 = 240 次，正好是「一次性跑完 2 小时」的规模，不用拆两批换口径。

### 0.4 成本

240 次 × 32 s ≈ **128 min ≈ 2.1 h**（判分与画图另加 1–2 min）。单遍 80 次 ≈ 43 min。

---

## 1. 协议（与 LAB12/15/16/17/18 一字不差，只换格）

| 项 | 值 |
| --- | --- |
| checkpoint | `runs/20261117-001-m4-w15-finetune/logs/g1_tracking/2026-11-17_09-25-17_m4-w15/model_52998.pt`（sha256 fc8f975b…8ceed9） |
| 动作 | `data/processed/fight1_s3_000s_20s.npz`（sha256 965f2c4b…b66c66） |
| 终止 | **关**（`--ignore-terminations`） |
| 推力时刻/波形 | 8.0 s 起，方波 0.2 s，推后看 3 s |
| 作用点 | `torso_link` |
| 格名 | `px_j60` `px_j80` `py_j60` `py_j80` |
| 方向与力 | +x = (300,0,0)、(400,0,0)；+y = (0,300,0)、(0,400,0) |
| 种子 | 101–120（20 个） |
| 遍数 | 3（新跑，不复用任何旧结果） |

判分沿用 LAB12 的三分类（stable / transient / down）与 D21/D22 判据；**判据冻结，不为单调性改**。

---

## 2. 动手步骤

### 步骤 0：开工前 60 秒核对

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
git status --short
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv
sha256sum runs/20261117-001-m4-w15-finetune/logs/g1_tracking/2026-11-17_09-25-17_m4-w15/model_52998.pt
ls runs | grep 20261121
```

**预期**：git 无输出；显存 < 500 MiB；sha 以 fc8f975b 开头；`ls runs | grep 20261121` 无输出（今天还没有 run）。

**不对就停**：显存被占先去关进程；sha 不对说明 ckpt 被动过，先查原因不要跑。

另外两条现实的提醒：**接上电源、关掉休眠**（WSL 一休眠这批就断在中途）；这 2 小时里不要开别的 GPU 任务。

### 步骤 1：先写预测（10 分钟）

打开 `docs/learning/LAB19-prediction.md`，在**跑之前**把 5 道题答完。不用对，但每题要写「为什么这么猜」（用 0.1 的表和 LAB11 的 +y 线索去推）。

### 步骤 2：建 run 目录 + 写 README 骨架

```bash
RUN=runs/20261121-001-lab19-symmetry
ls "$RUN" 2>/dev/null && echo "已存在，换编号" || echo "OK，可以用"
mkdir -p "$RUN"
cat > "$RUN/README.md" <<'EOF'
# LAB19 — 四向对称性：+x / +y 在 60/80 N·s（4 格 × 20 种子 × 3 遍）

- 目的：补齐 +x / +y 的 60 与 80 N·s，把「两向边界不同」扩成四向同口径对照；检验 80 档「20/20」是否只对 −x/−y 成立。
- 假设：（填你在 LAB19-prediction.md 里的核心预测）
- 配置：ckpt model_52998（sha 前 8 位 fc8f975b）、fight1_s3_000s_20s.npz、终止关、8.0 s 起推 0.2 s、种子 101–120、3 遍、格 px_j60 px_j80 py_j60 py_j80。
- 命令：见 docs/learning/LAB19-symmetry.md §2。
- 结果：（跑完填：4 格 × 3 遍 k、翻转、CP 区间、方向排序、下一步）
- 下一步：（填）
EOF
ls "$RUN"
```

**预期**：最后 ls 只列出 README.md。
**自检**：`ls runs/*.md runs/*.log` 必须无输出（有输出说明 RUN 又写错到了 runs 根目录）。

### 步骤 3：一条命令跑完三遍（≈2.1 h）

把下面整块**一次粘贴**。它先写一个脚本，再用 tee 前台跑，日志同时落盘：

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
RUN=runs/20261121-001-lab19-symmetry
cat > /tmp/run_lab19.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh"
RUN=runs/20261121-001-lab19-symmetry
CELLS="px_j60 px_j80 py_j60 py_j80"
SEEDS="101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120"
for rep in 1 2 3; do
  echo "=== rep$rep 开始 $(date '+%F %T') ==="
  LAB12_TAG=lab19-symmetry LAB12_CELLS="$CELLS" LAB12_SEEDS="$SEEDS" LAB12_RUN_DIR="$RUN/rep$rep" \
    bash scripts/lab12_recovery.sh run
  echo "=== rep$rep 结束 $(date '+%F %T') ==="
done
echo "=== 全部完成 $(date '+%F %T') ==="
EOF
bash /tmp/run_lab19.sh 2>&1 | tee "$RUN/run.log"
```

**你会看到**：每遍开头 `[1/80] px_j60 seed 101 force=300,0,0`，逐条推进到 `[80/80] py_j80 seed 120`，然后「判分」+「完成：…/repN/recovery.md」；三遍之后最后一行是「=== 全部完成 … ===」。

**如果要关终端/离开**（可选，二选一，别和上面同时跑）：

```bash
nohup bash /tmp/run_lab19.sh > "$RUN/run.log" 2>&1 &
sleep 5; tail -n 5 "$RUN/run.log"
```

**跑完自检（四条都要过）**：

```bash
for r in 1 2 3; do printf "rep%s: " "$r"; ls "$RUN/rep$r" | tr '
' ' '; echo; done
for c in px_j60 px_j80 py_j60 py_j80; do for r in 1 2 3; do printf "%s rep%s: " "$c" "$r"; ls "$RUN/rep$r/$c" | wc -l; done; done
grep -h -E '^(sha256|push|termination|cells|seeds)' "$RUN"/rep*/params.txt | sort | uniq -c
grep -c '判分' "$RUN/run.log"
```

**预期**：每遍有 4 个格子目录；12 条 ls 全部打印 20；grep 显示 sha256 / push / termination / cells / seeds 各出现 3 次且内容一致；最后一条打印 3。

### 步骤 4：合三遍，算翻转率与图

```bash
RUN=runs/20261121-001-lab19-symmetry
environment/.venv/bin/python scripts/lab16_report.py --dir "$RUN" \
  --rep rep1="$RUN/rep1" --rep rep2="$RUN/rep2" --rep rep3="$RUN/rep3" \
  --cells px_j60,px_j80,py_j60,py_j80 \
  --title "LAB19 四向对称性：+x/+y 60/80 N·s" \
  --figure outputs/figures/lab19_symmetry.png \
  --figure-title "LAB19: +x/+y 60/80 N s (20 seeds x 3 reps)"
```

**预期**：终端按格打印 4 行「k=a / b / c  二进制翻转 n/20  三分类翻转 m/20 种子 [...]」，写盘 `$RUN/fliprate.md` / `fliprate.json`，图 `outputs/figures/lab19_symmetry.png`（4 个面板，每面板 3 行 × 20 列）。

### 步骤 5：读数（顺序别跳）

1. **N 列**：4 格 × 3 遍，每个必须 20（`fliprate.md` 的每格表）。
2. **逐遍 k**：4 格 × 3 遍 = 12 个数，先抄到纸上排成 4×3 矩阵。**先看 +y 再看 +x**（+y 有 LAB11 的独立线索）。
3. **翻转**：4 格各一个 n/20；对照「−x 60 = 4/20、−y 60 = 1/20」。若某格 ≥3/20，按 D28 写区间。
4. **方向对照**：`py_j60 vs ny_j60（18–19/20）`、`px_j60 vs nx_j60（5–6/20）`。差异要带 CI 说，别只说「更脆」。
5. **80 档**：`px_j80` / `py_j80` 是不是也 20/20？< 18/20 就说明「80 起全倒」只对 −x/−y 成立，旧表述要改。
6. **101–105 子集**：这份表对 +x/+y 会全是「—」（LAB12 没有这两格的 20 种子基线），只记录数值，不做「一致/不一致」判断。
7. **方向排序**：把三档（60/80/120）排成 4 向表，写清哪些格有 20 种子、哪些只有 5 种子。

### 步骤 6：把结果写进文档（不做完，lab 不算完成）

八处，缺一不算完成：

1. `$RUN/README.md`：结果表（4 格 × 3 遍 k + CI + 翻转）、判读、局限、下一步；
2. `docs/REPORT.md`：在 §2.5.9 之后、`## 3. 关键发现` 之前插 `### 2.5.10`（表格 + 4–5 条结论 + 图）；
3. `README.md` 结果表：扰动行追加四向 60/80 的数字与「方向排序」一句；
4. `docs/DECISIONS.md`：新增 `## D32`（依据/实测/决定），把「方向 vs 大小」的结论定下来；
5. `docs/STATE.md`：头部「最后更新/当前阶段」+ 末尾追加「## LAB19 结果…当前事实以此为准」；
6. `docs/PLAN.md`：勾选 LAB19（写 run 目录与数字），并在路线里把 LAB20 标成下一个；
7. `docs/ARTIFACT_INDEX.md`：插新节（run、图、讲义、D32）；
8. `docs/learning/LAB19-symmetry.md` §7 填结果 + `LAB19-prediction.md` 填判分（判分时**在你的答案下方写参考回答**）。

写完再顺手核对：新旧数字不矛盾（例如「80 起全倒」若被推翻，README/REPORT/HANDOFF 三处都要同步改）。

### 步骤 7：提交

```bash
git status --short
git add -A
git commit -m "feat(lab19): 四向对称性 +x/+y 60/80 N·s 各 20 种子 3 遍；D32"
cat > /tmp/push_lab19.sh <<'EOF'
#!/bin/bash
cd /home/overlume/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh || exit 1
GIT_TERMINAL_PROMPT=0 git -c 'credential.helper=!"/mnt/c/Program Files/Git/mingw64/bin/git-credential-manager.exe"' push origin main
echo "PUSH_EXIT=$?"
EOF
bash /tmp/push_lab19.sh
```

**预期**：push 最后打印 `PUSH_EXIT=0`，`git log --oneline -1` 是你的提交。

---

## 3. 通过判定（我会拿这张表验收）

| # | 检查项 | 通过标准 |
| --- | --- | --- |
| 1 | 目录结构 | `$RUN/rep1..3` 各含 px_j60/px_j80/py_j60/py_j80，每格 20 个 seed 目录 |
| 2 | 协议一致 | 三份 params.txt 的 sha256 / push / termination / cells / seeds 完全相同，cells 含 4 格 |
| 3 | 次数 | run.log 里 240 次 `[i/80]`（每遍 80 次，三遍各从 1 开始） |
| 4 | 汇总产物 | `fliprate.md` 有 4 格 × 3 遍表、CP、翻转统计与明细；`fliprate.json` 可解析 |
| 5 | 图 | `outputs/figures/lab19_symmetry.png` 有 4 个面板、每面板 3 行、20 列种子 |
| 6 | 读数 | run README 里有 4 格 × 3 遍的 k、CI、翻转、四向排序，且标明 60/80 有 20 种子、120 只有 5 种子 |
| 7 | 文档 | §2 步骤 6 的八处全部改完，新旧表述不矛盾 |
| 8 | 入库 | 提交 + push 成功；`git status` 干净；`docs/handoffs/HANDOFF.md` 未被 git add |

## 4. 常见问题与排查

| 现象 | 原因 | 怎么办 |
| --- | --- | --- |
| 每遍 `[i/80]` 都从 1 重新开始 | 每遍是新的 lab12_recovery.sh 进程 | 正常 |
| 某格 k = 0/20 | 可能是真的抗推（+x 在 120 N·s 就有「稳」的先例），也可能是格/力写错 | 先核 params.txt 的 cells，再 `grep force=` run.log 看是不是 300,0,0 / 400,0,0；都对就当真结果 |
| 单次评估 > 60 s | GPU 被占 / 降频 | `nvidia-smi` 看占用与温度；停掉别的进程后重跑该遍 |
| `fliprate.md` 的 101–105 表全是「—」 | +x/+y 没有 LAB12 基线 | 正常，别去改脚本 |
| 中途断了想续跑 | 同名 seed 目录会被覆盖重跑 | 重跑整个 rep（别手工删一半）；已完成的 rep 不用重跑 |
| `runs/` 根目录多了 run.log / README.md | RUN 写错 | 搬进正确目录，看 `ls runs/*.md runs/*.log` 确认清空 |
| push 报 credential.helper 错误 | 引号被 bash 吃掉 | 写成 /tmp/push_lab19.sh 再 bash 执行（步骤 7 已给） |

## 5. 六个坑（每个都有人踩过）

1. **用 `scripts/lab16_fliprate.sh` 跑这个 lab**：它的第 1 遍硬编码引用 LAB15 的 −x/−y 目录，对 +x/+y 是空的 —— 白跑。这里必须用 `lab12_recovery.sh` 三遍全跑。
2. **LAB12_CELLS 用逗号分隔**：它是空格分隔（`CELLS="px_j60 px_j80 …"`），写成逗号会变成一格，报「无法解析冲量」。
3. **只跑一遍就下结论**：D28 起边界/过渡格一律 3 遍；这一批是「第一条基线」，更要 3 遍。
4. **把「对称/不对称」说得没有限定**：必须带方向、冲量、N、CI 与协议（终止关、8.0 s、20 种子）。「+y 更脆」这种句子单独出现是不合格的。
5. **忘了 80 档可能被推翻**：如果 +x/+y 的 80 档不是 20/20，README、REPORT、HANDOFF §2.3 的「80 起全倒」都要同步改，不能只改一处。
6. **讲义里的尖括号占位符原样粘**：`<` 与 `>` 是重定向符，粘进 bash 会静默把文件写到奇怪的地方（2026-11-20 踩过）。本讲义的命令都写死了编号，直接粘即可。

## 6. 数据不对怎么办（分支预案）

| 如果 | 说明 | 怎么办 |
| --- | --- | --- |
| py_j60 ≥ 15/20 | +y 才是最脆方向（与 LAB11 姿态审计一致） | D32 写「四向排序：+y > −y > …」；结论里说明方向特异性，别写成「越靠近 +y 越脆」这种无依据的连续表述 |
| py_j60 在 12–15/20 | +y 与 −y 在 60 档接近，±y 整体比 ±x 脆 | 排序写 ±y > ±x，并注明两向差在 60 档不显著（带 CI） |
| px_j60 ≤ 2/20 | +x 显著抗推 | 先查格/力；确认后写「+x 是四向里最稳」，并说明 −x 5–6/20 的对比 |
| px_j60 ≈ 5–6/20 | +x 与 −x 边界接近 | 结论是「x 向不区分正负，y 向区分」 |
| px_j80 或 py_j80 < 18/20 | 「80 起全倒」被推翻 | 立即改 README/REPORT/HANDOFF 三处；D32 记「80 档也方向相关」 |
| 某格翻转 ≥ 3/20 | 判定不稳（过渡区常见） | 按 D28 区间叙事，写逐遍 k + 中位 |
| 三遍 k 单调下降（像 LAB18 的 14/11/10） | 首遍效应再现 | 记进 D32 的「待验证机制」，不要当成随机噪声，也不要直接归因 |

跑完把实际走到的分支写进 run README 的「判读」，不要只留一句「符合预期」。

## 7. 结果（2026-11-21 实测，供对照）

（跑完由智能体填）
