# LAB27：多片段联合训练（S3 0 s + dance1 0 s）

> 目标：回答一个之前所有 lab 都绕开的问题——**「单片段训练」本身是不是 held-out 只有 0.100 的原因？**
> 做法：给策略第二段动作（LAFAN1 的 dance1，先做地面高度校正过 D41 闸），其余口径与 M4 完全一致，看 held-out / unseen 会不会动。
> 预计：Part A 1 分钟 + Part B 1 h 45 + Part C 50 min ≈ **2 h 40**。

## 0. 已知与设计

### 0.1 现在的状况（读数时要用到的锚点）

| 项目 | 数值 | 来源 |
|---|---|---|
| M4 训练片段（S3 0–20 s）完成度 | 0.999 | LAB09 / REPORT §2 |
| M4 held-out（S3 120–140 s） | **0.100** | LAB09（同片段、未训练窗口） |
| M4 unseen（S2 134 s，另一段动作） | **0.025** | LAB09 |
| 官方权重同口径 | 0.333 / 0.016 / 0.013 | LAB09 |
| LAB20（加强 DR）held-out | 0.018 | LAB20（D34） |
| LAB21（加世界系根位姿）held-out | 0.018 | LAB21（D42） |

两个已排除的解释：加强 DR 没用（LAB20），加世界系位姿观测没用（LAB21）。**剩下最可能的是数据**：只训了一段 20 s 动作，策略学到的是「记住这段动作」，所以在同一段的邻近帧上还能蹭到 0.1，换一段就塌到 0.025。

### 0.2 本 lab 改什么、不改什么

- **改**：训练数据从 1 段变成 2 段——S3 0–20 s（原样，不改）+ dance1 0–20 s（新片段，做过地面校正）。两段各 999/1000 帧，等长。
- **不改**：热启动权重（scratch/model_49999.pt）、3000 iter、4096 envs、seed 0、目标 iter 52998、DR 默认值、观测维度（160 维）。**任何差异只能来自数据**。

### 0.3 为什么新片段必须先做地面校正

LAB22 审计（D41）发现：重定向产物脚底中位悬空 0.8–2.8 cm、最深穿地 5.3–6.0 cm，即同一段动作里有的帧脚在空中、有的帧穿到地下。D41 定了闸：**任何新 npz 进训练/评估集，穿地 max ≤20 mm 且 p99 ≤20 mm**。原版 dance1 是 53.3 mm（FAIL），所以本 lab 第一步就是用 scripts/lab27_ground_fix.py 做「接触帧地面钳制 + 腾空帧线性插值」，把每个窗口压过闸。S3 0 s 不动（历史基线继承，改了 M4 数字就不可比）。

### 0.4 三种结果怎么读

- **(A) 有效**：held-out 从 0.100 涨到 ≥0.15（Δ ≥ +0.05）且 ≥7/10 个种子同向 → 「单片段训练」确实是瓶颈之一，可以继续扩到 3–4 段。
- **(B) 只涨训练片段**：train 仍然 0.999、dance 窗口（训练里出现的动作、没出现的窗口）涨了，但 held-out / unseen 不动 → 策略学到的是「多记一段」，不是泛化；下一步要换更根本的手段（更大数据量 / 更长片段 / 域随机化策略）。
- **(C) 更差**：held-out 掉到 0.05 以下 → 异质片段损害了 fight 专精（两段动作打架），需要重新想混合比例或 curriculum。

### 0.5 成本

| 阶段 | 内容 | 次数 | 预计 |
|---|---|---|---|
| Part A | 地面校正 + 闸门 | 2 个窗口 | ≈1 min |
| Part B | 多片段训练 3000 iter | 1 次 | ≈1 h 45 |
| Part C | 4 窗口 × 10 种子 × 2 策略 | 80 次评估 | ≈50 min |

## 1. 协议（必须与 M4 一致，否则不可比）

| 项 | 值 |
|---|---|
| 训练起点 | scratch/model_49999.pt（sha256 前 8 **f1bc404f**…后 6 **f22319**） |
| 训练片段 | data/processed/fight1_s3_000s_20s.npz（999 帧）+ data/processed/dance1_000s_20s_gfix.npz（1000 帧） |
| 迭代 | 3000（热门 ckpt 从 49999 走到 **52998**） |
| 环境数 / 种子 | 4096 / agent.seed 0 |
| 评估脚本 | scripts/eval_tracking.py，终止检测**开**、启动 DR 关、999 步 |
| 评估种子 | 101–110（10 个） |
| 评估窗口 | train = S3 0 s；heldout = S3 120 s；unseen = S2 134 s；dance = dance1 60–80 s（gfix） |

## 2. 操作步骤

### 步骤 0：开工核对（1 分钟）

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
sha256sum scratch/model_49999.pt data/processed/lafan_dance1_subject1.npz
ls -l data/processed/*.npz
```

预期：
- GPU 显存占用 < 1 GB（有人占着就先别开训）。
- scratch/model_49999.pt 的 sha 以 **f1bc404f** 开头、**f22319** 结尾。
- data/processed 里有 fight1_s3_000s_20s.npz、fight1_s3_120s_20s.npz、fight1_s2_134s_20s.npz、lafan_dance1_subject1.npz（6574 帧）。

### 步骤 1：先写预测（5 分钟，别跳过）

打开 `docs/learning/LAB27-prediction.md`，把 5 道题的答案写在每题下面。**先写再看结果**，否则读数时会不自觉地事后解释。

### 步骤 2：建 run 目录 + README 骨架

```bash
RUN=runs/20261125-004-lab27-multiclip
mkdir -p "$RUN"
cat > "$RUN/README.md" <<'EOF'
# LAB27 多片段联合训练（S3 0 s + dance1 0 s）

- 目的：单片段训练是不是 held-out 只有 0.100 的原因？给第二段动作后 held-out / unseen 会不会动。
- 假设：held-out 涨到 ≥0.15（Δ ≥ +0.05）且 ≥7/10 同向 = 数据是瓶颈之一。
- 配置：热启动 scratch/model_49999.pt（f1bc404f…f22319）、3000 iter、4096 envs、seed 0、目标 52998；
  片段 = fight1_s3_000s_20s.npz（999 帧）+ dance1_000s_20s_gfix.npz（1000 帧）；DR 默认；观测 160 维。
- 评估：4 窗口（train/heldout/unseen/dance）× 10 种子 × 2 策略 = 80 次，终止开、999 步。
- 结果：（跑完填）
- 下一步：（填）
EOF
ls "$RUN"
```

⚠ **RUN 不能写成 `runs` 本身、也不能留尖括号占位**（`<` 是重定向）。自检：

```bash
ls runs/*.md runs/*.json 2>/dev/null
```

必须**没有输出**（说明你没把文件写到 runs 根目录）。

### 步骤 3：Part A —— 新片段地面校正（1 分钟）

```bash
environment/.venv/bin/python scripts/lab27_ground_fix.py \
  --src data/processed/lafan_dance1_subject1.npz --start-s 0 --dur-s 20 \
  --out data/processed/dance1_000s_20s_gfix.npz
```

```bash
environment/.venv/bin/python scripts/lab27_ground_fix.py \
  --src data/processed/lafan_dance1_subject1.npz --start-s 60 --dur-s 20 \
  --out data/processed/dance1_060s_20s_gfix.npz
```

预期：每条命令打印一段 JSON（before / after）+ 一行闸门结论，after 的穿地应显著变小：

- 0–20 s：before `pen_max_mm` ≈ 32.9 → after ≈ 3.5，p99 ≈ 0.8，**「[gate] PASS」**
- 60–80 s：before ≈ 9.9 → after ≈ 5.9，p99 ≈ 2.5，**「[gate] PASS」**

自检：两次都必须是 PASS（阈值 20/20）。如果 FAIL，把 after 那两个数记下来，先别训练。

### 步骤 4：Part B —— 多片段训练（1 h 45）

```bash
environment/.venv/bin/python -u scripts/lab27_multiclip_train.py \
  --run-root "$RUN" --mode full 2>&1 | tee "$RUN/train.log"
```

预期开头几行：

```txt
[lab27] 已打包 2 段 -> <...>/motion_packed
[lab27] motion_ends（累计帧）= [999, 1999]，合计 1999 帧
[lab27]   fight1_s3_000s_20s.npz：999 帧
[lab27]   dance1_000s_20s_gfix.npz：1000 帧
```

然后开始打 iter。**这个脚本支持暂停续训**：任何时候 Ctrl-C 或断线，重新跑同一条命令（它会自动从最新 ckpt 接着训到 52998）：

```bash
environment/.venv/bin/python -u scripts/lab27_multiclip_train.py \
  --run-root "$RUN" --mode full --resume 2>&1 | tee -a "$RUN/train.log"
```

跑完的自检（三条都要过）：

```bash
grep -n "motion_file" "$RUN"/logs/g1_tracking/*/params/env.yaml
cat "$RUN"/motion_packed/motion_ends.npy > /dev/null && \
  environment/.venv/bin/python -c "import numpy as np,glob;d=sorted(glob.glob('$RUN/logs/g1_tracking/*/model_*.pt'));print(d[-1]);print(np.load('$RUN/motion_packed/motion_ends.npy'))"
```

- env.yaml 里 `commands.motion.motion_file` 指向 `$RUN/motion_packed`（一个目录，不是单个 npz）。
- motion_ends = [999 1999]。
- 最新 ckpt 是 model_52998.pt。

⚠ 评估时**不要用 `pretrained/` 目录里的副本**（那是热启动权重，2026-11-22 在 LAB20 踩过这个坑）。评估脚本已经用 `grep -v "/pretrained/"` 排除。

### 步骤 5：Part C —— 评估（≈50 min）

```bash
bash scripts/lab27_eval.sh run
```

- 80 次评估，逐条打印 `p w seed`；中断后重跑会自动跳过已有 summary.json。
- 时间参考：单次 ≈37 s（LAB21 实测 37.6 s），80 次 ≈50 min。

### 步骤 6：出表 + 读数

```bash
cat "$RUN/eval_summary.md"
```

**读数顺序**（不要先看 dance）：

1. **held-out（S3 120 s）**：主指标。M4 = 0.100。LAB27 是多少？逐种子差有几个是正？
2. **unseen（S2 134 s）**：M4 = 0.025。涨了才说明「换了动作也会一点」。
3. **dance（dance1 60–80 s）**：训练里出现的是 dance1 0–20 s，这一格是同动作的另一窗口 —— 若它涨而 held-out 不涨，就是 (B) 只记住了一段。
4. **train（S3 0 s）**：必须还是 0.999；掉了说明训崩了。
5. 存活帧数面板：完成度 0.1 对应的存活只有 100 帧（2 s），别把它读成「能跑 20 s」。

### 步骤 7：填 README + 提交

把步骤 6 的表贴进 `$RUN/README.md` 的「结果」节，写 3–5 条判读（属于 (A)/(B)/(C) 哪一种，证据是哪两个数），然后：

```bash
sha256sum data/processed/dance1_000s_20s_gfix.npz data/processed/dance1_060s_20s_gfix.npz
```

把这两行 sha256（前 8 + 后 6 位就够）抄进 README 的「配置」节末尾——`data/processed/` 在 .gitignore 里，npz 不入库，sha 是以后唯一能核对「训练用的到底是哪份片段」的凭据。

然后提交文档：

```bash
git add docs/learning/LAB27-prediction.md
git add runs/20261125-004-lab27-multiclip/README.md
git add runs/20261125-004-lab27-multiclip/eval_summary.md runs/20261125-004-lab27-multiclip/eval_summary.json
git commit -m "feat(lab27): 多片段联合训练——S3 0 s + dance1 0 s，测 held-out 是否松动"
```

（文档全局同步与 push 我收尾时来做，你不用管 REPORT/DECISIONS/STATE。）

## 3. 「做完了」的判定（8 条）

1. 两个 gfix npz 都存在，且两次闸门都是 PASS（after max ≤20 mm、p99 ≤20 mm）。
2. `$RUN/motion_packed/` 存在，motion_ends = [999 1999]。
3. env.yaml 里 motion_file 指向 motion_packed 目录。
4. 训练 ckpt 到 model_52998.pt（iter 52998），train.log 里没有 NaN、没有「error」字样。
5. 评估 80 个 summary.json 齐全（`find "$RUN/eval" -name summary.json | wc -l` 应为 80）。
6. `eval_summary.md` 四行窗口都有数，M4 的 held-out ≈0.100（±0.01）——对得上历史，说明评估口径没跑偏。
7. train 窗口仍是 0.999。
8. README 结果节写清：LAB27 的 held-out 值、Δ、逐种子同向数，以及属于 (A)/(B)/(C) 哪一类。

## 4. 常见问题

| 现象 | 排查 |
|---|---|
| 训练一启动就报 ValueError「provide --registry-name / --env.commands.motion.motion-file」 | mjlab 的启动校验只接受「单个存在的路径字符串」，传 list 会被拦。本脚本已改成 packed 目录，若仍报就是 motion_packed 没生成——看 `ls $RUN/motion_packed` |
| 自检 `ls runs/*.md runs/*.json` 输出了 `runs/README.md` | 不是本 lab 产生的：它是 2026-11-20 LAB17 那次「RUN 写成 runs」误写留下的空模板（2026-11-25 已删除）。删掉即可（`rm runs/README.md`）；runs/ 根目录平时不该有任何文件，有输出就说明有人写错了位置 |
| 报「motion joints [...] are not the robot's [...]」 | npz 的 joint_names 顺序不对（多半是自己切片段时改过），重新用 lab27_ground_fix.py 生成 |
| 训练很慢 / iteration time 为负 | resume 后 Iteration time 可能是负的（LAB20 记录过），看 Time elapsed / ETA |
| 评估报缺数据 dance1_060s_20s_gfix.npz | 步骤 3 的第二条没跑 |
| 评估跑到一半断了 | 重跑 `bash scripts/lab27_eval.sh run`，已完成的会跳过 |
| 表里 M4 held-out 不是 0.100 | 评估口径跑偏了（ckpt 选错 / 窗口错），先 `jq .checkpoint` 对一下是不是 M4 的 model_52998.pt |
| dance 窗口 M4 很低（≈0.17） | 正常：dance1 是没训过的动作，M4 只能撑 3.4 s |

## 5. 五个坑

1. **RUN 留成 `runs`**：所有产物会直接躺进 runs 根目录，和历史 run 混在一起（2026-11-20 踩过）。跑完先 `ls runs/*.md` 自检。
2. **尖括号占位**：讲义里所有 `<...>` 都要换成你的真实值，否则 `<` 会被 bash 当重定向，变量变空。
3. **用 pretrained/ 里的权重做评估**：那是热启动副本，不是训练产物，会得到「官方权重」的成绩（LAB20 踩过，差点把结论写反）。
4. **只看完成度不看存活帧**：0.100 的完成度对应存活 100 帧（2 s）。完成度涨 0.05 只多活 0.5 s，别写成「能跟住了」。
5. **把 (B) 当 (A)**：dance 窗口涨不等于泛化，它只是同一段动作的另一窗口。判 (A) 只看 held-out 与 unseen。

## 6. 分支预案

- **(A) 有效（held-out ≥0.15 且 ≥7/10 同向）**：扩到 3–4 段（dance1 再加 2 个窗口 + S3 120 s 也要先过闸），然后复测 LAB12 抗推（60 N·s 两向 × 20 种子 × 3 遍），看「泛化涨了，鲁棒性有没有被牺牲」。
- **(B) 只涨训练片段**：停在这里，转「同片段内换窗口」的更细粒度诊断：把 S3 的 20 s 切成 4 个 5 s 子段，训练只用 3 段、留 1 段当 held-out——这能区分「窗口记忆」与「动作泛化」。
- **(C) 更差（held-out <0.05）**：把两段的比例从 50/50 改成 80/20（S3 权重高），或改用「先训 S3 再微调 dance」的 curriculum；同时记录「异质片段互相干扰」这一现象。
- **闸门 FAIL**：把 after 的 max/p99 写进 README 当负结果，改用更保守的插值（--smooth 5）或放弃该窗口，不要强行训练。

## 7. 结果（跑完填）

```txt
（此处由用户填：4 个窗口的完成度表 + 判读 3–5 条 + 属于 (A)/(B)/(C) 哪一类）
```
