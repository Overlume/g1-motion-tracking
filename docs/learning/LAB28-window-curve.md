# LAB28 窗口距离曲线：记忆的作用半径有多大

> 一句话：LAB27 发现「多练的片段能把**同一动作的另一段**从 0.061 抬到 0.108」。
> 本 lab 零训练，只评估：**同一段动作上，离训练窗口越远，这份记忆还剩多少？**

## 0. 先看清要回答的问题

### 0.1 为什么问这个

- M4 只训练过 **fight1 的 S3 0–20 s** 这一段（999 帧）。
- 但它在 **fight1 S3 120 s**（同一段动作，隔 120 s）上还有 **0.100**，在 **fight1 S2 134 s**（另一段）上只有 **0.025**。
- LAB27 又给它加了 **dance1 0–20 s**，结果 **dance1 60–80 s** 从 **0.061 → 0.108**（9/10 种子更好），而 fight1 的 held-out / unseen **一动不动**。

这两个事实放在一起只有一个解释方向：策略学到的不是「这套动作的规律」，而是「**贴着训练窗口的一段轨迹**」。那就必须问一个定量问题：

> 记忆从训练窗口往外延伸多远开始失效？20 s？40 s？还是根本没有半径、只有「训练窗口本身」？

答案直接决定下一步该怎么做：
- 如果半径是几十秒 → 加数据有用，但要**沿时间轴铺开**，不是换动作。
- 如果半径 < 20 s → 这是**局部记忆**，加多少段都只是「多背几段」，得改方法（观测/参考/训练方式）。

### 0.2 已知数字（同一套评估口径）

| 对象 | M4（只练 fight1 S3 0 s） | LAB27（+dance1 0 s） | 出处 |
|---|---|---|---|
| fight1 S3 0–20 s（M4 训练窗口） | 0.999 | 0.999 | M4 / LAB27 评估 |
| fight1 S3 120 s（同动作，距 120 s） | 0.099 | 0.097 | LAB27 §4 |
| fight1 S2 134 s（换动作） | 0.025 | 0.026 | LAB27 §4 |
| dance1 0–20 s（LAB27 训练窗口） | 未测 | **本 lab 测** | — |
| dance1 60–80 s（同动作，距 40–60 s） | 0.061 | 0.108 | LAB27 §4 |

注意两件事：

1. **dance1 0–20 s 从未被单独评估过**（LAB27 只测了 60–80 s）。它是本 lab 的「距离 = 0」锚点，也是整条曲线的上限。
2. fight1 的 120 s 与 dance1 的 60 s 不是同一动作，**不能**拿来当距离证据——这正是本 lab 要把它换成同一段动作窗口序列的原因。

### 0.3 三种可能的结果，各自意味着什么

| 读法 | 曲线长相 | 结论与下一步 |
|---|---|---|
| **(A) 有半径** | 0 s 最高，之后单调下滑，最远窗口收敛到 M4 水平（≈0.04–0.06） | 记忆有几十秒的作用半径 → 加数据要**沿时间铺满**整段动作（每 20 s 一段），而不是换动作；按半径决定铺多密 |
| **(B) 只认窗口** | 只有 0 s 高（≥0.5），20 s 就掉到 M4 水平 | 几乎是纯局部记忆（半径 < 20 s）→ 加段数只是「多背几段」，必须改方法（观测、参考、训练目标） |
| **(C) 不单调/平台** | 0 s 高，之后忽高忽低或整段平台 | 距离不是主因（可能是相位/参考质量/接触）→ 先查参考与相位对齐，再谈数据量 |

M4 那条线是**地板**：它从没见过 dance1，所以它在本 lab 的 6 个窗口上的水平就是「没记忆时的成绩」。

### 0.4 为什么是 6 个窗口、为什么要有 0 s

- dance1 全长 **6574 帧 = 131.5 s（50 fps）**，切 20 s 一段正好 6 段：0/20/40/60/80/100 s。
- 20 s = 训练窗口长度，和训练/评估口径（999 步 ≈ 20 s）一致，段与段之间不重叠。
- **必须包含 0 s**：没有「距离 0」的点，曲线就没有上限锚点，也看不出 20 s 掉多少。
- 为什么不加 120 s（最后 11.5 s 不足 20 s）、不加密到 10 s：先看粗曲线形状定 (A)/(B)/(C)，加密留给分支预案。

### 0.5 成本

| 步骤 | 内容 | 时间 |
|---|---|---|
| Part A | 6 个窗口地面校正（2 个已存在会跳过，4 个新生成） | 每个 ≈10 s，共 <1 min |
| Part B | 2 策略 × 6 窗口 × 10 种子 = **120 次评估** | ≈75 min（LAB21 实测 37.6 s/次） |
| 出表 | scripts/lab28_report.py | <10 s |
| 合计 | 零训练 | **≈80 min** |

## 1. 协议（只换被评估的动作窗口，其它一律不动）

| 项 | 值 |
|---|---|
| 源数据 | data/processed/lafan_dance1_subject1.npz（6574 帧，131.5 s，50 fps） |
| 窗口 | 起点 0/20/40/60/80/100 s，各 20 s；生成 data/processed/dance1_SSSs_20s_gfix.npz |
| 预处理 | scripts/lab27_ground_fix.py（D41 闸门：穿地 max ≤20 mm 且 p99 ≤20 mm，不过就退出） |
| 策略 M4 | runs/20261117-001-m4-w15-finetune/logs/g1_tracking/2026-11-17_09-25-17_m4-w15/model_52998.pt |
| 策略 LAB27 | runs/20261125-004-lab27-multiclip/logs/g1_tracking/2026-11-25_18-15-01_lab27-multiclip/model_52998.pt（sha **8745f96e**…**125dac**，脚本会校验） |
| 评估 | scripts/eval_tracking.py：终止检测**开**、启动 DR 关（默认）、999 步 |
| 种子 | 101–110（10 个） |
| 完成度 | summary.json 的 track_completion（到过的最远帧 / 总帧） |
| 存活帧 | summary.json 的 motion_frame_furthest（满 999 = 跑完；held-out 类窗口常只有几十帧） |

> ⚠ 口径提醒：非训练窗口的完成度普遍很低（0.02–0.11），**1 帧 = 0.001**。所以差 0.01（≈10 帧）才算「有点差」，差 0.003 基本是噪声。

## 2. 七步

### 步骤 0：开工核对（≈1 min）

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
git log --oneline -1          # 期望 e9e2ea6 feat(lab27): 多片段联合训练……
git status --short            # 期望干净（顶多 scratch/ 里的东西）
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
sha256sum runs/20261125-004-lab27-multiclip/logs/g1_tracking/*/model_52998.pt
ls -l data/processed/lafan_dance1_subject1.npz
environment/.venv/bin/python -c "import numpy as np; d=np.load('data/processed/lafan_dance1_subject1.npz'); print(len(d['joint_pos']), d['joint_pos'].shape)"
```

自检：
- ckpt sha 必须是 **8745f96e…125dac**（不是 f1bc404f… 那个 M4 热启动权重）。
- dance1 帧数必须是 **6574**（= 131.5 s @ 50 fps）。不是这个数就去查数据版本。
- nvidia-smi 至少留 3 GB 空闲（只评估，不需要 4096 envs）。

### 步骤 1：写预测（≈5 min）

打开 docs/learning/LAB28-prediction.md，5 题都先写死数字，**跑之前**提交。

### 步骤 2：建 run 目录 + README 骨架（≈2 min）

```bash
RUN=runs/20261126-001-lab28-window-curve
mkdir -p "$RUN"
cat > "$RUN/README.md" <<'EOF'
# LAB28 窗口距离曲线（dance1 的 6 个 20 s 窗口）

- 目的：同一段动作上，完成度随「距训练窗口的时间距离」怎么衰减 → 判断记忆的作用半径。
- 假设：(A) 有半径 20–60 s / (B) 只有训练窗口本身 / (C) 平台或不单调。
- 配置：M4 ckpt 与 LAB27 ckpt（sha 8745f96e…125dac）× dance1 的 0/20/40/60/80/100 s 各 20 s × 种子 101–110，零训练。
- 命令：见 docs/learning/LAB28-window-curve.md §2。
- 结果：（跑完填：逐窗口完成度、曲线形状、属于 A/B/C）
- 下一步：（填）
EOF
ls "$RUN"
```

自检：`ls "$RUN"» 只应看到 README.md；`ls runs/*.md» 不应有输出（runs 根目录不能有文件）。

### 步骤 3：Part A 生成 6 个窗口（≈1 min）

```bash
LAB28_RUN=runs/20261126-001-lab28-window-curve bash scripts/lab28_window_curve.sh build
```

预期：
- 0 s 与 60 s **已存在** → 打印「已有 … 跳过」。
- 20/40/80/100 s 各打印一段 JSON + 一行 `[gate] PASS（max … mm / p99 … mm，阈值 20/20）»。
- dance1 的接触率极高（LAB22 实测 90%），所以穿地校正的幅度很小，预期 max 在 3–10 mm。

自检：
```bash
ls -l data/processed/dance1_*s_20s_gfix.npz
environment/.venv/bin/python - <<'PY'
import numpy as np, glob
for f in sorted(glob.glob("data/processed/dance1_*s_20s_gfix.npz")):
    d = np.load(f)
    print(f.split("/")[-1], d["joint_pos"].shape, round(len(d["joint_pos"]) / 50, 2), "s")
PY
```

应该是 6 个文件、每个 (1000, 29) 或 (999, 29)、≈20 s。**任何一个 [gate] FAIL 就停下**（把那一行发我），不要带病往下跑。

### 步骤 4：Part B 评估 120 次（≈75 min，可中断）

```bash
LAB28_RUN=runs/20261126-001-lab28-window-curve bash scripts/lab28_window_curve.sh run \
  2>&1 | tee -a runs/20261126-001-lab28-window-curve/eval.log
```

- 这条命令会**先 build 再评估再出表**；已生成的窗口与已有 summary.json 都会跳过，所以中断后重跑同一条即可续。
- 每次评估约 38 s，120 次 ≈75 min。日志里每行形如 `[lab28] m4 w020 seed101»。
- 中途 Ctrl-C 不要紧：已完成的 seed 有 summary.json，重跑会 skip。

自检（跑完）：
```bash
ls runs/20261126-001-lab28-window-curve/eval/m4 runs/20261126-001-lab28-window-curve/eval/lab27
find runs/20261126-001-lab28-window-curve/eval -name summary.json | wc -l    # 必须是 120
```

### 步骤 5：出表、读数（≈10 min）

```bash
bash scripts/lab28_window_curve.sh report
cat runs/20261126-001-lab28-window-curve/lab28_curve.md
```

读数顺序（重要，别跳步）：

1. **先看 0 s 那一行**：LAB27 在训练窗口本身应该很高（≥0.4，可能 0.6–0.9；M4 在 0.1–0.4）。如果 0 s 就很低，说明评估链坏了，别往下读。
2. **再看 M4 整条线**：应该平在 0.03–0.10（它从没见过 dance1）。如果 M4 在远端特别高，怀疑窗口数据或种子有问题。
3. **看 LAB27 的 0 → 20 落差**：掉一半以上还是只掉一点？这是 (A)/(B) 的分水岭。
4. **看最远窗口（100 s）**：LAB27 是否收敛到 M4 水平？还是仍明显高于 M4？
5. **最后看逐种子配对差**：9–10/10 同向才算稳；5/5 分开说明该窗口的差是噪声。
6. **自检行**：报告末尾会打印 `[lab28] 自检：LAB27 60 s 窗口 = …（上一轮 dance 60–80 s = 0.108）»。两个数应在 ±0.02 内一致；差得多说明这轮评估与 LAB27 那轮不可比（查 ckpt sha、窗口文件 sha）。

图：outputs/figures/lab28_window_curve.png（左：完成度曲线；右：存活帧）。

### 步骤 6：填 README + 同步 8 处文档（≈20 min）

按顺序改（细则见 docs/handoffs/HANDOFF.md §8）：

1. runs/20261126-001-lab28-window-curve/README.md：填结果、判读（属于 A/B/C）、局限。
2. docs/REPORT.md：在 §2.5.21（LAB27）之后新增「### 2.5.22 窗口距离曲线」；「## 3. 关键发现」加一条；「## 4. 局限」加一条。
3. docs/DECISIONS.md：末尾新增 D44（判读 + 决定）。
4. docs/STATE.md：头部「最后更新」+ 当前阶段句；末尾新增「## LAB28 结果」。
5. docs/PLAN.md：LAB28 打勾 + 一行实测。
6. docs/ARTIFACT_INDEX.md：新增「## 2026-11-26（LAB28 窗口距离曲线）」。
7. README.md：结果表加一行。
8. 本讲义 §7 填实测；docs/learning/LAB28-prediction.md 写判分 + 参考回答。

### 步骤 7：提交并推送

```bash
git add -A
git commit -m "feat(lab28): 窗口距离曲线——记忆的作用半径；D44"
git log --oneline -1
GIT_TERMINAL_PROMPT=0 git -c 'credential.helper=!"/mnt/c/Program Files/Git/mingw64/bin/git-credential-manager.exe"' push origin main
```

（data/processed 下的 npz 被 .gitignore 忽略，不会进库；不在 data/ 上纠结。）

## 3. 通过判定（8 条）

1. 6 个窗口文件齐全，每个 ≈1000 帧，且每个都过了 D41 闸（max ≤20 mm、p99 ≤20 mm）。
2. eval 下 120 个 summary.json 齐全；两个策略的 ckpt sha 在 eval.log 里记录正确（LAB27 = 8745f96e…125dac）。
3. 自检行对得上：LAB27 的 60 s 窗口与上一轮 0.108 差 ≤0.02。
4. 0 s 窗口的 LAB27 完成度**明显**高于其余 5 个窗口（否则曲线没有上限锚点，先查评估链）。
5. 能说清曲线属于 (A)/(B)/(C) 中的哪一种，并给出**关键落差**（0→20 s 掉多少、100 s 是否收敛到 M4）。
6. 报「逐遍/逐种子」：每个窗口给出「LAB27 更好几/10、更差几/10」，不只给均值。
7. 完成度之外还报了存活帧与 reset 次数（完成度低时两者能区分「跑丢」与「跑歪」）。
8. 八处文档都改完，且没有任何一处写成「多片段训练提升了泛化」这种被 D43 否掉的话。

## 4. 常见问题

| 现象 | 原因 | 怎么办 |
|---|---|---|
| build 打印 [gate] FAIL | 该窗口穿地超 20 mm | 记下窗口起点发我；不要改阈值硬跑 |
| 某窗口 20 s 生成失败 | 源数据不足 20 s（100 s 窗口只剩 31.5 s，没问题；120 s 没排） | 检查 --start-s 与 --dur-s |
| eval 里出现 [ERR] … 失败 | 显存/EGL 偶发 | 重跑同一条命令，已完成的会跳过 |
| summary.json 只有 118 个 | 中途 Ctrl-C | 重跑 run，缺的会补 |
| 曲线 0 s 只有 0.1 | LAB27 ckpt 选错（选到 pretrained 副本）或窗口文件错 | 看 eval.log 里 ckpt 路径与 sha；脚本本身会校验 sha 不符就退出 |
| 60 s 自检差 0.05 | 这轮的窗口文件与 LAB27 那轮不是同一份（重新生成过） | 比对 sha256；必要时用 LAB27 那份 dance1_060s_20s_gfix.npz |
| 图里中文变方块 | matplotlib 无中文字体 | 图标题一律英文（脚本已如此），正文用 md |

## 5. 五个坑

1. **把完成度当精度**：dance1 的窗口完成度都很低，0.108 vs 0.061 只是 108 vs 61 帧。要说「多撑了 1 秒」，不要写「提升了 77%」。
2. **跨动作比距离**：fight1 的 120 s（0.099）和 dance1 的 60 s（0.108）不是同一条曲线，别混着画。
3. **只看均值**：10 个种子里 9 好 1 差 与 6 好 4 差 是两件事，必须报同向个数。
4. **忘看 M4 地板**：LAB27 在远端「比 M4 高 0.02」可能只是 M4 那条线本身抖（±0.01）。
5. **拿 0 s 的绝对高度当「学得好」**：0 s 是训练窗口，它高不代表泛化；曲线形状才是结论。

## 6. 分支预案

| 结果 | 下一步 |
|---|---|
| (A) 有半径（远端收敛到 M4） | 把 dance1 全段（6 段）加进训练，做成「同动作全覆盖」训练，看 held-out 是否随之抬升（LAB29 候选） |
| (B) 只认窗口（20 s 就掉到 M4） | 停加数据；改观测/参考（LAB30 候选：相位或参考速度），因为「多背一段」已被证伪 |
| (C) 平台/不单调 | 先做参考质量与相位对齐审计（LAB22 的 Part A 可复用），再谈数据 |
| 0 s 也很低（<0.2） | 评估链或 ckpt 有问题，停下来查，不要往下推 |

## 7. 结果（跑完填）

（待填）
