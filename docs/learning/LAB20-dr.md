# LAB20 — 训练期 DR 加强：能否把 held-out 完成度从 0.10 拉起来

> 讲义版本：2026-11-21。参照 LAB17 / LAB19 的组织方式。
> 这是**第一个动训练侧**的 lab：会产出新 ckpt，耗时 ≈ 2.7 h（训练 1.9 h + 评估 40 min + 机动）。

## 0. 这个 lab 要达成什么

### 0.1 已经知道什么

**M4 策略的泛化成绩（LAB09，5 种子 101–105，终止开、启动 DR 关、999 步）**

| 窗口 | 类型 | 完成度 | 说明 |
| --- | --- | --- | --- |
| S3 000–20 s | 训练片段 | **0.999** | 几乎满分，root 误差 3.88 cm |
| S3 120–140 s | 同片段未见窗口 | **0.101** | 平均 2.0 s 就被 ee_body_pos 重置 |
| S2 134 s | 未见片段 | **0.025** | 半秒级就重置 |

结论：M4 是「把这个 20 s 窗口背下来了」，不是「会跟踪新运动」（D17）。

**一个容易搞错的事实：DR 早就开着。** mjlab 的 tracking 任务默认就带三项 startup 随机化：

| DR 项 | 默认范围 | 代码位置 |
| --- | --- | --- |
| 脚底摩擦系数 | 0.3 – 1.2（双脚共享同一个采样） | tracking_env_cfg.py:165-205（foot_friction） |
| 质心偏移 | ±2.5 / ±5 / ±5 cm | 同上（base_com，body = torso_link，见 g1/env_cfgs.py:59-62） |
| 关节编码器偏置 | ±0.01 rad | 同上（encoder_bias） |

所以 LAB20 **不是「开 DR」，而是「把 DR 调强」**：对照组就是 M4 自己（默认 DR）。

### 0.2 本 lab 的目标

只改一个因素——**DR 强度**，其余（起点权重、动作、seed、迭代数、奖励、并行数）与 M4 完全一致，回答：

1. 把 DR 调强后，**held-out 完成度（0.101）能提升多少**？
2. **unseen 片段（0.025）**是否跟着提升？如果只有 held-out 涨、unseen 不涨，说明只是「对同片段别的窗口更宽容」，不是泛化。
3. 训练窗口（0.999）**掉多少**？掉太多说明 DR 太强、把专精能力买走了。

### 0.3 为什么三档一起加强，而不是一次只改一档

- 一次只改一档 → 3 个训练 × 1.9 h ≈ 5.7 h，本次 lab 预算装不下；
- 三档一起改，实验因素写成「DR 强度（默认 → 加强）」，仍然是一个因素，解释时不许拆成「摩擦的功劳」「质心的功劳」；
- 如果这次有正向效果，下一轮再做「单档消融」（那是 LAB20b 的事，不是这次）。

### 0.4 加强到什么值，依据是什么

| DR 项 | 默认 | LAB20 | 理由 |
| --- | --- | --- | --- |
| 脚底摩擦 | 0.3 – 1.2 | **0.2 – 1.4** | 覆盖冰面到橡胶的更宽区间；0.2 仍在地面物理合理范围内 |
| 质心偏移 | ±2.5 / ±5 / ±5 cm | **±4 / ±6 / ±6 cm** | 人形机器人负载/装配误差的保守上限 |
| 编码器偏置 | ±0.01 rad | **±0.02 rad** | 约 1.15°，再大就不是「传感器误差」而是「关节坏了」 |

### 0.5 成本

| 阶段 | 内容 | 预计耗时 |
| --- | --- | --- |
| 冒烟 | 50 iter / 256 envs | 3–5 min |
| 正式训练 | 3000 iter / 4096 envs（同 M4） | **≈ 1.9 h**（M4 实测 6412 s） |
| 评估 | 2 策略 × 3 窗口 × 10 种子 = 60 次 | ≈ 40 min |
| 合计 | | **≈ 2.7 h** |

---

## 1. 协议

| 项 | M4（对照） | LAB20（本次） |
| --- | --- | --- |
| 起点权重 | scratch/model_49999.pt（sha256 f1bc404f…f22319） | **同一个** |
| 动作 | data/processed/fight1_s3_000s_20s.npz（sha256 965f2c4b…b66c66） | 同一个 |
| 并行环境 | 4096 | 4096 |
| 迭代 | 49999 → 52998（+3000） | 同 |
| seed | 0 | 0 |
| num_steps_per_env | 24 | 24 |
| save_interval | 500 | 500 |
| logger | tensorboard | tensorboard |
| DR | 默认（0.3–1.2 / ±2.5,±5,±5 / ±0.01） | **加强（0.2–1.4 / ±4,±6,±6 / ±0.02）** |
| 评估 | eval_tracking.py，终止开、启动 DR 关、999 步 | 同 |

评估种子：**101–110**（10 个）。M4 基线在 LAB09 用的是 101–105，所以本 lab 会在同一台机器上**把 M4 也重跑一遍同样的 10 个种子**，保证配对比较（不拿旧数字比新数字）。

---

## 2. 动手步骤

### 步骤 0：开工前 60 秒核对

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
date
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
sha256sum scratch/model_49999.pt
git log --oneline -3
git status --short
ls -d runs/20261121-003-* 2>/dev/null
```

预期：
- GPU 是 NVIDIA GeForce RTX 3060 Laptop GPU，显存 6 GB，`memory.used` 在 500 MiB 以下；
- sha256 前 8 位 `f1bc404f`、后 6 位 `f22319`；
- `git status --short` 无输出；
- `ls -d runs/20261121-003-*` 无输出（如果**有**输出，说明当天已经用到 003，把下面所有 `20261121-003` 换成 `20261121-004`，只换编号，别改结构）。

### 步骤 1：先写预测（不许跑完再补）

打开 docs/learning/LAB20-prediction.md，把 5 道题的答案写在每题下面。写完再往下。

### 步骤 2：建 run 目录 + README 骨架

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
RUN=runs/20261121-003-lab20-dr
mkdir -p "$RUN"
cat > "$RUN/README.md" <<'EOF'
# LAB20 训练期 DR 加强（vs M4 默认 DR）

- 目的：只改 DR 强度（摩擦 0.2–1.4 / 质心 ±4,±6,±6 cm / 编码器 ±0.02 rad），其余与 M4 一致，看三窗口完成度变化。
- 假设：held-out 完成度从 0.101 提升（≥ 0.20 才算实质），训练窗口不低于 0.99，unseen 至少不变得更差。
- 配置：起点 model_49999.pt、动作 fight1_s3_000s_20s.npz、4096 envs、+3000 iter、seed 0、评估终止开/启动 DR 关。
- 命令：见 docs/learning/LAB20-dr.md §2。
- 结果：（跑完填）
- 下一步：（填）
EOF
ls -la "$RUN"
```

自检：`$RUN` 里只有 README.md；**不要**把 RUN 设成 `runs` 本身（会污染 runs 根目录，2026-11-20 已经踩过一次）。

### 步骤 3：冒烟（3–5 min）

先用小规模跑通「改 DR → 训练循环 → 落 params」，不要直接上 1.9 h。

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
rm -rf scratch/lab20-smoke
environment/.venv/bin/python -u scripts/lab20_dr_train.py \
  --run-root scratch/lab20-smoke --mode smoke --iters 50 --envs 256 2>&1 | tail -20
```

预期看到三行 `[lab20]`：DR 已加强（friction=(0.20,1.40) com=±0.04/±0.06/±0.06 m bias=±0.020 rad）、iters=50 envs=256、日志根目录；随后出现 `Iteration 49999` 起的训练输出与 `Iteration time`。

自检四件事（都要做）：

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
grep -n "num_envs" scratch/lab20-smoke/logs/g1_tracking/*/params/env.yaml | head -3
sed -n "/base_com:/,/mode: startup/p" scratch/lab20-smoke/logs/g1_tracking/*/params/env.yaml | grep -A2 "ranges:" | head -12
grep -n -A3 "bias_range" scratch/lab20-smoke/logs/g1_tracking/*/params/env.yaml | head -6
sed -n "/foot_friction:/,/mode: startup/p" scratch/lab20-smoke/logs/g1_tracking/*/params/env.yaml | grep -A3 "ranges:" | head -6
```

期望：`num_envs: 256`；base_com 的 ranges 是 ±0.04 与 ±0.06；`bias_range: [-0.02, 0.02]`；foot_friction ranges 是 `0.2 / 1.4`。
四项里只要有一项还是默认值（±0.025、±0.01、0.3/1.2），**停下**——说明 DR 没改上，别往下跑。

确认没问题后清掉冒烟目录：`rm -rf scratch/lab20-smoke`。

### 步骤 4：正式训练（≈ 1.9 h，后台跑）

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
RUN=runs/20261121-003-lab20-dr
nohup environment/.venv/bin/python -u scripts/lab20_dr_train.py \
  --run-root "$RUN" --mode full > "$RUN/train.log" 2>&1 &
echo "PID=$!"
sleep 60
tail -5 "$RUN/train.log"
```

自检（训练 2 分钟后）：

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
RUN=runs/20261121-003-lab20-dr
grep -n "lab20\]" "$RUN/train.log" | head -5
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
tail -3 "$RUN/train.log"
```

期望：三条 `[lab20]` 都在（DR 加强、iters=3000 envs=4096、日志根目录）；显存占用 ≈ 3–5 GB；日志里迭代号从 49999 往上走。

训练期间可以关掉终端（nohup 已脱离），进度用：

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
RUN=runs/20261121-003-lab20-dr
tail -2 "$RUN/train.log"
ls "$RUN"/logs/g1_tracking/*/model_*.pt 2>/dev/null | tail -3
```

结束标志：`ls "$RUN"/logs/g1_tracking/*/model_52998.pt` 有输出（49999 + 3000 iter = 52998），且 `pgrep -f lab20_dr_train` 为空。

### 步骤 5：评估（≈ 40 min，60 次）

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
RUN=runs/20261121-003-lab20-dr
nohup env LAB20_RUN="$RUN" bash scripts/lab20_eval.sh run > "$RUN/eval.log" 2>&1 &
echo "PID=$!"
sleep 90
tail -5 "$RUN/eval.log"
```

自检：eval.log 里先打印两条 `[lab20] ... ckpt`（M4 与 DR 各一条，带 sha256），然后是 `m4 train seed 101` 这样的行。

中途断掉也没关系：脚本会跳过已有 summary.json 的格子。跑完自动生成表与图。

### 步骤 6：读数（5 分钟）

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
RUN=runs/20261121-003-lab20-dr
cat "$RUN/eval/lab20_eval.md"
```

按这个顺序读：

1. **heldout 那一行**：DR − M4 的差是正还是负？幅度多大？（0.101 → 0.15 算「有提升但不解决」；→ 0.30 以上算「实质提升」）
2. **逐种子配对差**：`DR 更好 X / 更差 Y / 差不多 Z`。X 和 Y 都很大说明提升不稳定，结论必须写「均值提升但种子间不一致」。
3. **train 那一行**：掉了多少？低于 0.98 就要写「用专精换泛化」。
4. **unseen 那一行**：这是「真泛化」的唯一证据。heldout 涨、unseen 不动 → 只能说「对同片段的其他窗口更宽容」。
5. **图**（`$RUN/eval/lab20_eval.png`）：两根柱子并排，一眼看出哪个窗口改善最大。

### 步骤 7：文档同步（8 处，按 HANDOFF §8）

| # | 文件 | 写什么 |
| --- | --- | --- |
| 1 | `$RUN/README.md` | 结果 + 下一步（替换「（跑完填）」） |
| 2 | docs/REPORT.md | 新增 §2.5.12 + §3 关键发现 + §4 局限各一条 |
| 3 | docs/DECISIONS.md | 新增 D34：DR 加强协议、判定线、结论 |
| 4 | docs/STATE.md | 头部「最后更新/当前阶段」+ 末尾新节 |
| 5 | docs/PLAN.md | LAB20 条目打勾 + 实测耗时 |
| 6 | docs/ARTIFACT_INDEX.md | 新日期节：run 目录、表、图、两份讲义 |
| 7 | README.md | 结果表加/改「训练期 DR」行 |
| 8 | docs/learning/LAB20-prediction.md | 结果 + 判分（判分要在你的答案下方附参考回答） |

### 步骤 8：提交 + 推送

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
git add -A
git status --short
git commit -m "feat(lab20): 训练期 DR 加强（摩擦 0.2-1.4 / 质心 ±4,±6,±6 / 编码器 ±0.02）对比 M4 三窗口"
bash /tmp/push_kam.sh
```

推送脚本（本机 credential manager 必须写在同一条命令里，引号原样写进文件）：

```bash
cat > /tmp/push_kam.sh <<'EOF'
#!/usr/bin/env bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
GIT_TERMINAL_PROMPT=0 git -c 'credential.helper=!"/mnt/c/Program Files/Git/mingw64/bin/git-credential-manager.exe"' push origin main
EOF
bash /tmp/push_kam.sh
```

---

## 3. 通过判定（8 条，全过才算这个 lab 做完）

1. `$RUN/logs/g1_tracking/*/params/env.yaml` 里三项 DR 都是加强值，不是默认值；
2. `$RUN` 里有 `model_52998.pt`，且 train.log 里迭代号从 49999 连续走到 52998；
3. `$RUN/eval` 下有 60 个 summary.json（`find "$RUN/eval" -name summary.json | wc -l` 输出 60）；
4. `$RUN/eval/lab20_eval.md` 的表格 3 行都填满（没有「—」）；
5. 结论对 heldout 只写**配对差与方向**，不写「DR 有用/没用」这种没幅度的话；
6. 结论区分了 heldout 与 unseen，不用 heldout 单独支撑「泛化变好」；
7. 训练窗口的代价写出来了（掉了多少、接不接受）；
8. 8 处文档同步 + 提交推送完成，`git status --short` 干净。

---

## 4. 常见问题排查

| 现象 | 原因 | 怎么办 |
| --- | --- | --- |
| 冒烟三行里没有「DR 已加强」 | 传了 `--default-dr` | 去掉该参数重跑 |
| env.yaml 里还是默认值 | 走了别的入口、改的不是真正被用的 cfg | 必须走 scripts/lab20_dr_train.py，不要直接 `train` |
| 找不到热启动权重 | scratch/model_49999.pt 被删 | 从 runs/20261117-001-m4-w15-finetune/logs/g1_tracking/pretrained/ 拿回，或问 lead |
| CUDA out of memory | 6 GB 显存被占 | `nvidia-smi` 看占用；关掉占显存的程序；必要时 `--envs 2048` 但要在 README 写明「口径变了，不可与 M4 直接比」 |
| 跑完没有 model_52998.pt | 迭代数或 save_interval 不对 | `--iters 3000`；看 params/agent.yaml 里 save_interval |
| 评估报「找不到 LAB20 ckpt」 | 训练还没出 ckpt 或 RUN 写错 | `ls "$RUN"/logs/g1_tracking/*/model_*.pt` |
| 迭代时间显示负数（如 -16.89s） | resume 后计时基准取自 checkpoint 里的 iteration，不从 0 计 | 忽略，只看 Time elapsed 与 ETA |
| 评估中断 | 正常 | 重跑同一条命令，已有 summary.json 会跳过 |
| 表里某行还是「—」 | 那一格缺种子 | `find "$RUN/eval" -name summary.json | wc -l` 查缺哪个策略/窗口 |

---

## 5. 五个坑

1. **把「DR 默认已开」当「没开」**：按「开 DR」写结论，整段解读都错。默认值见 §0.1。
2. **拿旧 M4 数字（5 种子）比新 DR 数字（10 种子）**：设计是「同 10 个种子重跑 M4」，不许省掉 M4 那一半。
3. **只看均值不看配对差**：10 个种子里 6 涨 4 跌，均值可能没变；结论要写分布。
4. **用 heldout 单独宣布「泛化提升」**：heldout 是同片段的另一个窗口，外部证据看 unseen。
5. **训练时同时评估**：6 GB 显存会被抢爆，计时也失真。训练完再评估。

---

## 6. 分支预案

| 结果 | 怎么走 |
| --- | --- |
| heldout 提升 ≥ 0.10 且 train ≥ 0.99 | 记为「DR 加强有效」，下一轮做单档消融（LAB20b：只加摩擦 / 只加质心） |
| heldout 提升但 train < 0.98 | 写「用专精换泛化」，把 DR 强度记成可调旋钮（D 记录） |
| heldout 与 unseen 都不动 | 瓶颈不在 DR，而在 3000 iter + 20 s 单片段；转 LAB21（观测加世界系根位/速度）或换更大动作库 |
| heldout 变差 | 先看 train 是否也掉（DR 太强学不动）；再考虑折中强度（摩擦 0.25–1.3、质心 ±3,±5,±5） |

---

## 7. 结果（跑完填）

（留空：三窗口完成度、配对差、训练代价、结论一句话、证据路径）
