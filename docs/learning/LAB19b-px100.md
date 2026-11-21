# LAB19b — 补 +x 的 100 N·s：把 +x 的 down 边界夹进 80–120（1 格 × 20 种子 × 3 遍）

> 讲义目标：你照着做，32 分钟拿到「+x 在 100 N·s 的倒地率」，并把它写进四向表，让「+x 最抗推」从一句定性的排序变成一条有中点的曲线。
>
> 一次跑满三遍，不探路（你 2026-11-21 的约定）；命令都是写死的编号，直接粘。

## 0. 这个 lab 要达成什么

### 0.1 已经知道什么（同 ckpt、同协议、种子 101–120）

| 方向 | 60 N·s（300 N） | 70 N·s | 80 N·s（400 N） | 120 N·s（600 N） |
| --- | --- | --- | --- | --- |
| −x | **5–6/20（25–30%）**，翻转 4/20（LAB16） | **55%**（中位，14/11/10；交点 ≈68，LAB18） | **20/20**（LAB15） | 5/5 倒（LAB12） |
| −y | **18–19/20（90–95%）**，翻转 1/20（LAB16） | — | **20/20**（LAB15） | 5/5 倒（LAB12） |
| +x | **0/20**（20/20 stable，连 transient 都没有） | — | **0/20**（1 stable + 19 transient） | 5 种子里 **2 稳 / 3 down**（LAB12） |
| +y | **20/20** | — | **20/20** | 5/5 倒（LAB12） |

（60/80 的 +x、+y 是 LAB19 刚跑的：240 次，2 h 4 min，翻转 0/80，D32。）

一句话：**+x 是唯一"80 N·s 还不倒"的方向**，但 120 N·s 已经 3/5 倒 → 它的 down 边界就藏在 80–120 之间，宽 40 N·s，是四向里唯一还没定位的缺口。

### 0.2 本 lab 的目标（三件，按重要性）

1. 量出 **px_j100 的倒地率 k/20 × 3 遍**，把「80–120」缩成「80–100」或「100–120」。
2. 判断 +x 的过渡是**陡转**（80 还是 0%，100 突然大半倒）还是**缓升**（跟 −x 在 60→80 那样一格一格爬）。这一条比数字本身更有用：它决定 +x 能不能用「线性插值估交点」这种写法。
3. 给四向表补中点，让「+x 最抗推」有曲线支撑，而不是只有两个天花板（0% 和 60%）。

### 0.3 为什么补 100，不补 90 或 110

预算只够一格，100 是**唯一能把 80–120 对半分**的位置：

- 若 px_j100 ≤ 3/20 → 边界落在 **100–120**，区间从 40 N·s 宽缩到 20 N·s 宽；
- 若 px_j100 ≥ 10/20 → 边界落在 **80–100**，且说明 +x 在 80→100 之间**陡转**（80 还是 0/20）；
- 90 或 110 只能各切掉 1/4，最坏情况下区间仍是 30–40 N·s 宽，收益小一半。

### 0.4 成本

- 60 次评估（1 格 × 20 种子 × 3 遍），LAB19 实测 **31 s/次** → **≈32 min**（三遍各约 10.5 min）。
- 显存占用 < 3 GB，可以和别的事情并行开着看，但**别同时跑第二个 GPU 任务**。

### 0.5 为什么还要三遍（不能只跑一遍）

D28 已经定了：**边界格默认 3 遍**。px_j100 恰好可能落在 50% 交点附近（那正是翻转最多的地方：LAB16 的 4/20、LAB17 的 3/20、LAB18 的 4/20 都在 k/N ≈ 40–60% 的格），单遍 k 不可信。相反，如果它像 px_j60 那样离交点很远，翻转大概率是 0 —— 两种结果都有信息量，前提是三遍。

---

## 1. 协议（与 LAB12/15/16/17/18/19 一字不差，只换格）

| 项 | 值 |
| --- | --- |
| checkpoint | `runs/20261117-001-m4-w15-finetune/logs/g1_tracking/2026-11-17_09-25-17_m4-w15/model_52998.pt`（sha256 以 `fc8f975b` 开头） |
| 动作 | `data/processed/fight1_s3_000s_20s.npz`（sha256 以 `965f2c4b` 开头） |
| 终止 | **关**（`--ignore-terminations`，脚本里已写死） |
| 推力时刻/波形 | 8.0 s 起，方波 **0.2 s**，推后看 3 s |
| 作用点 | `torso_link` |
| 格名 | **`px_j100`**（一格） |
| 方向与力 | +x = **(500,0,0)**（500 N × 0.2 s = 100 N·s） |
| 种子 | 101–120（20 个） |
| 遍数 | 3（新跑，不复用任何旧结果） |

⚠️ **不要手写力**。格名里的 `j100` 会被脚本的 `force_of()` 自动换算成 500 N；你只要保证格名是 `px_j100`。写成 `j100` 以外的数字（比如 `j50`）会得到 250 N，那是另一个实验。

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

**预期**：git 无输出；显存 < 500 MiB；sha 以 `fc8f975b` 开头；`ls runs | grep 20261121` 只列出 `20261121-001-lab19-symmetry`（今天已经有一个 run，所以新的编号是 **002**）。

**不对就停**：显存被占先关进程；sha 不对说明 ckpt 被动过，先查原因不要跑。

**接上电源、关掉休眠**（WSL 一休眠这批就断在中途），这半小时里不要开别的 GPU 任务。

### 步骤 1：先写预测（8 分钟，别省）

打开 `docs/learning/LAB19b-prediction.md`，在**跑之前**把 5 题答完，每题写一句「为什么这么猜」。这一格最值得预测的两件事：k 是多少、+x 的过渡是陡还是缓。

### 步骤 2：建 run 目录 + 写 README 骨架

```bash
RUN=runs/20261121-002-lab19b-px100
ls "$RUN" 2>/dev/null && echo "已存在，换编号" || echo "OK，可以用"
mkdir -p "$RUN"
cat > "$RUN/README.md" <<'EOF'
# LAB19b — 补 +x 的 100 N·s（1 格 × 20 种子 × 3 遍）

- 目的：把 +x 的 down 边界从「80–120」缩到「80–100」或「100–120」；判断 +x 的过渡是陡转还是缓升。
- 假设：（填你在 LAB19b-prediction.md 里的核心预测：k 大概多少、陡还是缓）
- 配置：ckpt model_52998（sha 前 8 位 fc8f975b）、fight1_s3_000s_20s.npz、终止关、8.0 s 起推 0.2 s、种子 101–120、3 遍、格 px_j100（500 N）。
- 命令：见 docs/learning/LAB19b-px100.md §2。
- 结果：（跑完填：三遍 k、翻转、CP 区间、+x 边界区间、陡/缓判断）
- 下一步：（填）
EOF
ls "$RUN"
```

**预期**：最后 `ls` 只列出 `README.md`。
**自检**：`ls runs/*.md runs/*.log` 必须无输出（有输出说明 RUN 又写错到了 runs 根目录——2026-11-20 踩过）。

### 步骤 3：一条命令跑完三遍（≈32 min）

整块**一次粘贴**。它先写脚本，再用 tee 前台跑，日志同时落盘：

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
RUN=runs/20261121-002-lab19b-px100
cat > /tmp/run_lab19b.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh"
RUN=runs/20261121-002-lab19b-px100
CELLS="px_j100"
SEEDS="101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120"
for rep in 1 2 3; do
  echo "=== rep$rep 开始 $(date '+%F %T') ==="
  LAB12_TAG=lab19b-px100 LAB12_CELLS="$CELLS" LAB12_SEEDS="$SEEDS" LAB12_RUN_DIR="$RUN/rep$rep" \
    bash scripts/lab12_recovery.sh run
  echo "=== rep$rep 结束 $(date '+%F %T') ==="
done
echo "=== 全部完成 $(date '+%F %T') ==="
EOF
bash /tmp/run_lab19b.sh 2>&1 | tee "$RUN/run.log"
```

**你会看到**：每遍开头 `[1/20] px_j100 seed 101 force=500,0,0`，逐条推进到 `[20/20] px_j100 seed 120`，然后「判分」+「完成：…/repN/recovery.md」；三遍之后最后一行是「=== 全部完成 … ===」。

**如果要关终端/离开**（可选，二选一，别和上面同时跑）：

```bash
nohup bash /tmp/run_lab19b.sh > "$RUN/run.log" 2>&1 &
sleep 5; tail -n 5 "$RUN/run.log"
```

**跑完自检（四条都要过）**：

```bash
for r in 1 2 3; do printf "rep%s: " "$r"; ls "$RUN/rep$r" | tr '\n' ' '; echo; done
for r in 1 2 3; do printf "rep%s seed 数: " "$r"; ls "$RUN/rep$r/px_j100" | wc -l; done
grep -h -E '^(sha256|push|termination|cells|seeds)' "$RUN"/rep*/params.txt | sort | uniq -c
grep -c '判分' "$RUN/run.log"
```

**预期**：每遍只有 `px_j100` 一个格子目录（外加 `recovery.json` / `recovery.md` / `params.txt` 等文件）；三个 seed 数都是 **20**；五个字段各出现 3 次且内容一致（`cells: px_j100`）；最后打印 **3**。

### 步骤 4：合三遍，算翻转率与图

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
RUN=runs/20261121-002-lab19b-px100
environment/.venv/bin/python scripts/lab16_report.py --dir "$RUN" \
  --rep rep1="$RUN/rep1" --rep rep2="$RUN/rep2" --rep rep3="$RUN/rep3" \
  --cells px_j100 \
  --title "LAB19b +x 100 N s：把 +x 的 down 边界夹进 80-120" \
  --figure outputs/figures/lab19b_px100.png \
  --figure-title "LAB19b: px_j100 (20 seeds x 3 reps)"
```

**预期**：终端打印 `px_j100: k=a / b / c  二进制翻转 n/20  三分类翻转 m/20 种子 [...]`；写盘 `$RUN/fliprate.md` 与 `fliprate.json`；图 `outputs/figures/lab19b_px100.png`（1 个面板、3 行 × 20 列）。
（我 2026-11-21 15:42 用合成树验证过单格链路：1 面板图正常出，脚本无需改。）

### 步骤 5：读数（顺序别跳）

1. **N 列**：`fliprate.md` 每格表里 rep1/2/3 的 n 都必须是 **20**；不是 20 说明有 seed 目录缺或跑挂了，先修再读。
2. **逐遍 k**：三个数抄下来，看是 0/0/0 型（一致）、还是像 LAB18 那样 14/11/10（单调下降，疑似首遍效应）。
3. **翻转**：一个 n/20。对照「k 在 40–60% 的格翻 3–4/20」这条先验——**如果 px_j100 的 k 落在 4–16/20 却翻转 0/20，那是反常，要先查再写**。
4. **夹边界**：拿 k 的中位数判断区间 —— ≤3/20 → 边界 **100–120**；4–9/20 → 交点 **≈100**；≥10/20 → 边界 **80–100**，且 +x 是陡转。
5. **陡/缓**：把 +x 的三个点排成 (60, 0%) → (80, 0%) → (100, ?) → (120, 60%)。**80→100 涨得多就是陡转**（+x 不是线性爬，而是「扛到某个力就突然崩」）；100→120 才涨就是缓升。
6. **101–105 子集**：这份表对 +x 会全是「—」（LAB12 没有 px 的 20 种子基线），只记录数值，别做「一致/不一致」判断。

### 步骤 6：把结果写进文档（不做完，lab 不算完成）

七处，缺一不算完成（比上一个 lab 少一处，因为只有一格）：

1. `$RUN/README.md`：填结果表（3 遍 k + CI + 翻转）、陡/缓判断、+x 边界区间、局限、下一步；
2. `docs/REPORT.md`：在 §2.5.10 之后、`## 3. 关键发现` 之前插 `### 2.5.11`（一行表 + 3–4 条结论 + 图）；
3. `README.md` 结果表的扰动行：把「+x 的 down 边界在 80–120 之间」改成你测出来的区间；
4. `docs/DECISIONS.md`：新增 `## D33`（依据 / 实测 / 决定），把「+x 的边界区间 + 陡缓 + 要不要再补点」定下来；
5. `docs/STATE.md`：头部「最后更新 / 当前阶段」+ 末尾追加「## LAB19b 结果…当前事实以此为准」；
6. `docs/PLAN.md`：勾选 LAB19b（写 run 目录与数字），路线里把 LAB20 仍标成下一个；
7. `docs/ARTIFACT_INDEX.md` 插新节（run、图、讲义、D33）；`docs/learning/LAB19b-px100.md` 补 §7 结果；`docs/learning/LAB19b-prediction.md` 填结果与判分（判分时**在答案下方写参考回答**——你自己写也行，我验收时会补）。

写完顺手核对：新旧数字不矛盾（尤其 README / REPORT / STATE / HANDOFF 里「80–120」这个说法——测完就该被替换成新区间）。

### 步骤 7：提交

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
git status --short
git add -A
git commit -m "feat(lab19b): 补 +x 100 N·s 定位 down 边界；D33"
cat > /tmp/push_lab19b.sh <<'EOF'
#!/bin/bash
cd /home/overlume/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh || exit 1
GIT_TERMINAL_PROMPT=0 git -c 'credential.helper=!"/mnt/c/Program Files/Git/mingw64/bin/git-credential-manager.exe"' push origin main
echo "PUSH_EXIT=$?"
EOF
bash /tmp/push_lab19b.sh
```

**预期**：push 最后打印 `PUSH_EXIT=0`；`git log --oneline -1` 是你的提交；`git status --short` 干净（`docs/handoffs/` 被 gitignore，不会出现）。

---

## 3. 通过判定（我会拿这张表验收）

| # | 检查项 | 通过标准 |
| --- | --- | --- |
| 1 | 目录结构 | `$RUN/rep1..3` 各只含 `px_j100`，每格 20 个 seed 目录 |
| 2 | 协议一致 | 三份 params.txt 的 sha256 / push / termination / cells / seeds 完全相同，`cells: px_j100` |
| 3 | 次数 | `run.log` 里 60 次 `[i/20]`（三遍各从 1 到 20） |
| 4 | 力正确 | 每条都是 `force=500,0,0`（不是 300 或 600） |
| 5 | 汇总产物 | `fliprate.md` 有 3 遍表、CP、翻转统计与明细；`fliprate.json` 可解析 |
| 6 | 图 | `outputs/figures/lab19b_px100.png` 有 1 个面板、3 行、20 列种子 |
| 7 | 读数 | run README 里有三遍 k、CI、翻转、+x 边界区间（80–100 或 100–120）、陡/缓判断 |
| 8 | 文档 | §2 步骤 6 的七处全部改完，且「80–120」的旧表述已被替换 |
| 9 | 入库 | 提交 + push 成功；`git status` 干净；`docs/handoffs/HANDOFF.md` 未被 git add |

## 4. 常见问题与排查

| 现象 | 原因 | 怎么办 |
| --- | --- | --- |
| `ls "$RUN/rep1"` 里除 px_j100 还有别的格 | `CELLS` 被改多格了 | 只保留 px_j100，删掉多余格目录后重跑该遍 |
| `[i/20]` 每遍都从 1 重新开始 | 每遍是新的 lab12_recovery.sh 进程 | 正常 |
| 力变成 300 / 600 | 格名写成了 j60 / j120 | 改回 `px_j100`，重跑该遍 |
| k = 0/20 三遍全 0 | 可能是真的抗推（+x 在 80 就 0/20），也可能力没推上 | 先核 `force=500,0,0` 与 params.txt，再看 `recovery.json` 的 `max_drop` / `pelvis_z_min` |
| 单次评估 > 60 s | GPU 被占 / 降频 | `nvidia-smi` 看占用与温度；停掉别的进程后重跑该遍 |
| `fliprate.md` 的 101–105 表全是「—」 | +x 没有 LAB12 基线 | 正常，别去改脚本 |
| 中途断了想续跑 | 同名 seed 目录会被覆盖重跑 | 重跑整个 rep（别手工删一半）；已完成的 rep 可跳过 |
| `runs/` 根目录多了 run.log / README.md | RUN 写错 | 搬进正确目录，`ls runs/*.md runs/*.log` 确认清空 |
| push 报 credential.helper 错误 | 引号被 bash 吃掉 | 用步骤 7 的 `/tmp/push_lab19b.sh` 再执行一次 |

## 5. 六个坑（每个都有人踩过）

1. **`RUN=runs/...` 里的尖括号**：`<YYYYMMDD>` 这类占位符**不能原样粘**（`<` 是重定向，2026-11-20 你踩过，RUN 退化成 `runs/`）。本讲义所有命令都是具体编号，直接粘。
2. **把 `px_j100` 写成 `nx_j100`**：方向反了，测的是 −x 的 100 N·s（那是另一个实验）。跑前看一眼第一条 log 的 `force=500,0,0`。
3. **只跑一遍就下结论**：D28 要求边界格 3 遍。一格跑遍只要 10.5 min，别省。
4. **用旧数字写文档**：`px_j80 = 0/20` 是 LAB19 的结果（不是这一格的），别把 100 的结果抄成 80。
5. **图标题写中文**：matplotlib 默认字体没有中文字形，中文会变豆腐块。所以 `--figure-title` 一律用 ASCII（本讲义已给好）。
6. **`--title` 和 `--figure-title` 写反**：前者是 md 一级标题（可以中文），后者是图标题（必须 ASCII）。

## 6. 数据不对怎么办（分支预案）

| 如果（三遍 k 的中位） | 说明 | 怎么办 |
| --- | --- | --- |
| **≥ 10/20** | 边界在 **80–100**，且 +x 是**陡转**（80 还是 0/20） | D33 写「+x 边界 80–100；80→100 陡转，+x 不能用线性插值」；不必再补 90 |
| **4–9/20** | 交点 ≈100，和 −x 的 ≈68、−y 的 ≈44 并列 | D33 写「+x 交点 ≈100（90–110）」；四向表补齐 |
| **1–3/20** | 边界在 **100–120** | D33 写「+x 边界 100–120」；要不要补 px_j110 由你定（再 32 min） |
| **0/20** | +x 在 100 仍不倒 | D33 写「+x 边界 100–120，且 >100」；补 px_j110 才值得，先记录不再加预算 |
| **翻转 ≥ 3/20** | 这一格在易翻带里 | 按 D28 写区间（逐遍 k + 极差），别写单点；若翻转全部集中在 rep1 → 记一笔「首遍效应」（LAB18 先例） |
| 三遍 k 极差 ≥ 6 个种子 | 落在 50% 交点正上方 | 三遍都报，写「每次运行 k ± 6」；不必加遍数是别的原因 |

---

## 7. 结果（2026-11-21 实测，供对照）

run：`runs/20261121-002-lab19b-px100`。三遍为 15:49:21–16:02:11、16:02:11–16:14:28、16:14:28–16:26:44，**合计 37 min 23 s**（60 次，约 37 s/次）。

| 遍 | stable | transient | down | 倒地率与 CP 95% |
| --- | --- | --- | --- | --- |
| rep1 | 8 | 12 | **0** | 0% |
| rep2 | 8 | 12 | **0** | 0% |
| rep3 | 7 | 13 | **0** | 0% |
| 合计 | 8/8/7 | 12/12/13 | k = **0/0/0** | **0%，[0%, 16.8%]** |

- 二进制翻转 **0/20**；三分类翻转 **6/20**（104/107/109/110/112/113，全 S↔T）—— k=0 的格子二进制不翻，但「崩没崩」会翻，这是目前最多的一格。
- **+x 的 down 边界 >100 N·s，落在 100–120**：120 N·s（LAB12，5 种子）是 3/5（CI 14.7–94.7%），与 100 的上界 16.8% 刚擦边 → 50% 交点只能写「约 110–120」。
- **你走的 §6 分支**：命中「k ≤ 3/20（含 0）」那一行 → 边界写在 100–120，且不必补 px_j110。
- 需要额外记住的观察：**stable 份额非单调**（60 = 20/20 → 80 = 1/20 → 100 = 7–8/20 → 120 = 2/5），机制待查，只作记录（D33）。
- 判分：本次按用户要求只给参考答案，未判分（见 `docs/learning/LAB19b-prediction.md`）。

