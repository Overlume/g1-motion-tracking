# LAB17 — −y 方向的过渡区：45 与 50 N·s 各 20 种子 × 3 遍

> 前置：LAB13（45 N·s 是 −y 的转折点，5 种子 3/5）、LAB15（20 种子：60 N·s 19/20）、LAB16（边界格单遍不可信，要 3 遍）。
> 讲义：`docs/learning/LAB17-y-transition.md`；预测：`docs/learning/LAB17-prediction.md`（**跑之前先填**）。
> 你要产出：`runs/<今天的日期>-00N-lab17-y-transition/` 全套 + 结论 + 文档同步 + 提交。

## 0. 这个 lab 要达成什么

### 0.1 已知道什么

| 冲量 | −y 倒地 | 出处 | 可信度 |
| --- | --- | --- | --- |
| 40 N·s（200 N） | 0/5 | LAB13 | 5 种子、单遍 |
| 45 N·s（225 N） | **3/5** | LAB13 | 5 种子、单遍（未复现过） |
| 60 N·s（300 N） | **18–19/20** | LAB15 + LAB16 | 20 种子、3 遍（翻转 1/20） |
| 80 N·s（400 N） | 20/20 | LAB15 | 20 种子、单遍 |

所以 −y 的 50% 交点**落在 40–60 N·s 之间**，最可能在 45–50。这一带现在只有 5 个种子、跑过一遍的数字，
既没有 20 种子，也没验过翻转——按 D28，单遍数据不能进报告。

### 0.2 本 lab 的目标

1. 把 −y 的过渡区从「40–60 之间」压到「**某两格之间**」，给出每格 20 种子的 Clopper–Pearson 区间。
2. 用 3 遍复核 LAB13 的「45 = 3/5」是不是又一个 D27（单遍多算/少算种子）。
3. 顺手回答：**过渡区中心是不是最不稳**（LAB16 里 ny_j60 只翻 1/20，离阈值更近的 45/50 可能翻得更多）。

### 0.3 为什么是 45 和 50，不是 55 或 40

- 55 夹在 50 与 60（18–19/20）之间，估计已在 85% 以上，信息量低于 45；
- 40 已知 0/5，先不花 60 次去确认它「还是 0」；
- **若 45 直接打满**（20/20），说明交点在 40–45，那时回补 40 N·s（见 §6 分支预案）。

### 0.4 成本

2 格 × 20 种子 × **3 遍** = **120 次评估**，单次约 32 s → 约 **64 min**，GPU 独占。
中途不要跑别的训练；分三遍跑，每遍约 21 min。

## 1. 协议（与 LAB12/15/16 一字不差，只换格）

| 项 | 值 |
| --- | --- |
| ckpt | `runs/20261117-001-m4-w15-finetune/logs/g1_tracking/2026-11-17_09-25-17_m4-w15/model_52998.pt`（sha256 前 8 位 fc8f975b） |
| 动作 | `data/processed/fight1_s3_000s_20s.npz` |
| 终止 | **关**（--ignore-terminations），推后看 3 s |
| 扰动 | --perturb-body torso_link，8.0 s 起、0.2 s 方波 |
| 格 | ny_j45 = (0, −225, 0)；ny_j50 = (0, −250, 0)（F = 冲量 ÷ 0.2） |
| 种子 | 101–120（与 LAB15/16 同一批，才能跨 lab 对比） |
| 判分 | scripts/lab12_recovery_report.py：stable / transient / down |

## 2. 动手步骤（照抄命令，一条一条来）

### 步骤 0：开工前 60 秒核对

```bash
cd /home/overlume/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
git status --short                       # 应为空；不为空先决定怎么处理，别带脏工作区跑
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader   # 显存应 < 500 MiB
ls -d runs/2026* | tail -3               # 看今天的编号到几了
sha256sum runs/20261117-001-m4-w15-finetune/logs/g1_tracking/2026-11-17_09-25-17_m4-w15/model_52998.pt
```

**预期**：git status 无输出；显存 < 500 MiB；sha256 以 fc8f975b 开头。
**不对就停**：显存被占 → nvidia-smi 看 PID；sha 不对 → 这个 ckpt 被动过，先查，不要跑。

### 步骤 1：先写预测（5 分钟）

打开 `docs/learning/LAB17-prediction.md`，在 **跑之前**把 5 道题答完。不需要对，但每条要给「为什么这么猜」。

### 步骤 2：建 run 目录 + 写 README 骨架

run 目录名 = `runs/<今天 YYYYMMDD>-<当天第几个>-lab17-y-transition`。
把下面的 `<YYYYMMDD>-00N` 换成你步骤 0 里看到的实际编号（LAB16 是 20261120-002，所以今天用 003 起）。

```bash
RUN=runs/<YYYYMMDD>-00N-lab17-y-transition
mkdir -p "$RUN"
cat > "$RUN/README.md" <<'EOF'
# LAB17 — −y 45/50 N·s 过渡区（20 种子 × 3 遍）

- 目的：把 −y 的 50% 交点压到 45–50 之间；复核 LAB13 的 45 = 3/5。
- 假设：45 落在 10–14/20（50–70%），50 落在 15–18/20（75–90%）；过渡区中心翻转 ≥ 60 N·s 的 1/20。
- 配置：ckpt model_52998（sha 前 8 位 fc8f975b）、fight1_s3_000s_20s.npz、终止关、8.0 s 起推 0.2 s、种子 101–120、3 遍。
- 命令：见 docs/learning/LAB17-y-transition.md §2。
- 结果：（跑完填：每格三遍 k、翻转、CP 区间、结论）
- 下一步：（填）
EOF
ls "$RUN"
```

**预期**：ls 只列出 README.md。

### 步骤 3：跑三遍（每遍约 21 min）

**不要用 scripts/lab16_fliprate.sh**——它把第 1 遍硬编码成引用 LAB15 的目录，格名对不上。这里直接用底层 LAB12 外壳，三遍全跑：

```bash
SEEDS="101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120"
for rep in 1 2 3; do
  LAB12_TAG=lab17-y-transition   LAB12_CELLS="ny_j45 ny_j50"   LAB12_SEEDS="$SEEDS"   LAB12_RUN_DIR="$RUN/rep$rep"   bash scripts/lab12_recovery.sh run
done
```

**每遍你会看到**：[1/40] ny_j45 seed 101 force=0,-225,0 … 末尾 == 判分，最后「完成：<run>/rep<N>/recovery.md」。

**跑完自检**（三条都要过）：

```bash
for r in 1 2 3; do printf "rep%s: " "$r"; ls "$RUN/rep$r/ny_j45" | wc -l; done   # 应各 20
for r in 1 2 3; do printf "rep%s: " "$r"; ls "$RUN/rep$r/ny_j50" | wc -l; done   # 应各 20
grep -h -E '^(sha256|push|termination)' "$RUN"/rep*/params.txt | sort | uniq -c
```

**预期**：两条 ls 各打印 20；grep 显示 sha256 / push / termination 三行各出现 3 次且内容完全相同。

### 步骤 4：合三遍，算翻转率与图

```bash
environment/.venv/bin/python scripts/lab16_report.py --dir "$RUN"   --rep rep1="$RUN/rep1" --rep rep2="$RUN/rep2" --rep rep3="$RUN/rep3"   --cells ny_j45,ny_j50 --figure outputs/figures/lab17_ythmids.png
```

**预期输出**（最后 4 行）：「ny_j45: k=<a> / <b> / <c>  二进制翻转 <n>/20  三分类翻转 <m>/20 种子 [...]」，
「ny_j50: …」，写盘 fliprate.md / fliprate.json，图 outputs/figures/lab17_ythmids.png。

产物：`$RUN/fliprate.md`（每格每遍表、逐种子 S/T/D、翻转统计、翻转明细、判定段）、fliprate.json、图。

### 步骤 5：读数（顺序别跳）

1. **N 列**：每格每遍必须 20。
2. **101–105 子集**：LAB13 的 45 = **3/5**；这遍应回到 3/5 附近。对不上先看 params.txt 的 sha，不要先怀疑环境。
3. **逐遍 k**：45 应是 10–14 的量级，50 应是 15–18。
4. **翻转**：ny_j45 / ny_j50 的二进制翻转数，参考 ny_j60 的 1/20。
5. **判定段**：脚本按 D28 三档规则给的每格一句话，抄进 README「结果」。
6. **单调性**：40(0/5) < 45 < 50 < 60(18–19/20) 必须严格成立（除非翻转让 45/50 区间重叠——那也是结论，要写）。

### 步骤 6：把结果写进文档（不做完，lab 不算完成）

| 文件 | 写什么 |
| --- | --- |
| `$RUN/README.md` | 结果（逐遍 k、翻转、CP 区间）、判读（交点落在哪）、局限、下一步 |
| docs/REPORT.md | 新小节「### 2.5.8 −y 过渡区（LAB17）」，插在 2.5.7 之后、## 3. 之前 |
| docs/DECISIONS.md | 新条目「## D30」（日期 + 标题 + 依据 + 实测 + 决定），写「−y 50% 交点区间」与「过渡区是否需要更多遍」 |
| docs/STATE.md | 末尾追加「## LAB17 结果」小节；头部「最后更新」改日期 |
| docs/PLAN.md | LAB16 那条后面加一条 LAB17 勾选项 |
| docs/ARTIFACT_INDEX.md | 新增一节，列出 run 目录、图、README |
| README.md | 结果表的扰动行补 −y 45/50 的数字 |
| docs/handoffs/HANDOFF.md | 只改 §0「下一步」与 §4 P2 状态（仅本地，不提交） |

### 步骤 7：提交

```bash
git add -A && git status --short          # 确认只包含这次该提交的东西
git commit -m "feat(lab17): −y 45/50 N·s 过渡区——20 种子×3 遍定位 50% 交点"
GIT_TERMINAL_PROMPT=0 git -c 'credential.helper=!"/mnt/c/Program Files/Git/mingw64/bin/git-credential-manager.exe"' push origin main
```

## 3. 通过判定（我会拿这张表验收）

- [ ] rep1/rep2/rep3 下 ny_j45 与 ny_j50 各 20 个种子目录，共 120 次评估，无缺格；
- [ ] 三遍 params.txt 的 sha256 / push / termination 完全一致；
- [ ] fliprate.md 有两格的逐遍 k、二进制翻转 n/20 + CP 区间、三分类翻转 n/20；
- [ ] 写出**每个翻转种子的方向**（如 ny_j45 seed 116: rep1=D -> rep2=T）；
- [ ] 明确写出「−y 的 50% 交点落在 ___–___ N·s 之间」，并给出 45 与 50 的倒地率区间；
- [ ] 101–105 子集与 LAB13 的 45 = 3/5 对得上，或你解释了为什么对不上；
- [ ] README 五节齐全，图 outputs/figures/lab17_ythmids.png 已生成且能看出翻转；
- [ ] §2 步骤 6 的八处文档全部改到，git status 干净、已 push。

**没过的典型表现**：只有 k 没有区间；把 3 遍当独立样本算 CI；改了判据阈值让数字好看；跳过 README 直接写 REPORT。

## 4. 常见问题与排查

| 现象 | 原因 | 怎么办 |
| --- | --- | --- |
| 找不到 checkpoint | 不在仓库根 / 用了别的 shell | cd 到仓库根再跑；脚本按相对路径找 ckpt |
| unrecognized arguments: 0,-225,0 | 手敲 eval_tracking.py 且用空格传负值 | 别手敲，用外壳脚本；手敲必须 --perturb-force=-300,0,0（等号形式） |
| 显存爆 / 进程被杀 | 同时有别的东西占 GPU | nvidia-smi 找 PID；等它结束或先停掉 |
| 某格某个种子没有 eval.log | 那一次进程被杀 | 单独补：LAB12_TAG=lab17-y-transition LAB12_CELLS="ny_j45" LAB12_SEEDS="115" LAB12_RUN_DIR="$RUN/rep2" bash scripts/lab12_recovery.sh run（会重判整个目录，已有数据不受影响） |
| 判分全是 transient | 崩了又恢复且末尾回到参考附近 | 正常结果，不是 bug；看 max_drop / end_drop 再下结论 |
| 单次远超 32 s | GPU 被抢 / 温度降频 | nvidia-smi 看利用率；不要并行跑第二个 lab |
| fliprate.md 里 101–105 表显示 — | lab16_report.py 的 LAB12_BASE 没有 ny_j45/j50 | 正常，不影响其它表；别去改脚本 |

## 5. 五个坑（每个都有人踩过）

1. **用 lab16_fliprate.sh 跑这个 lab**：它的第 1 遍固定引用 LAB15 目录，会拿 nx_j60/ny_j60 的数据去配 ny_j45 的格名，结果缺数据还不报错。
2. **把 3 遍当 3 个独立样本**：统计单位是种子；3 遍只给「至少翻一次」的下界（D28）。
3. **只报 k 不报逐遍 k**：45 这种边界格，逐遍 k 的极差比中位数更有信息量。
4. **为了让曲线单调去改判据**：判据（0.08 m / 13° / 10 帧）是冻结的（D21/D22）；不单调就是发现，写出来。
5. **跳过 README 先写 REPORT**：REPORT 里的数字必须能从 run 目录反查到；顺序反了就会写出没有证据的数字。

## 6. 数据不对怎么办（分支预案）

- 若 **45 打满 20/20**：交点在 40–45，回补 ny_j40（20 种子 × 3 遍，60 次 ≈32 min），把 P2 变成 40/45 两点。
- 若 **45 与 50 的区间大面积重叠**（如 45 = 12–15、50 = 14–17）：再加 ny_j47 意义不大（1 N·s 一档太细），改成在 REPORT 里给「交点 45–50，宽度 < 5 N·s」并停。
- 若某格**翻转 ≥3/20**：说明过渡区中心最不稳，这本身是对 D28 的加强（边界格 ≥3 遍的必要性），写进 DECISIONS，不要重跑。
