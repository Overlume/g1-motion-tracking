# LAB21 讲义：把「世界系根位置 / 速度」加进 actor 观测（P7）

> 位置：docs/learning/LAB21-world-root-obs.md　｜　预测：docs/learning/LAB21-prediction.md
> 脚本：scripts/lab21_obs_train.py（训练）、scripts/lab21_eval.sh（评估）、scripts/lab21_report.py（出表）、scripts/obs_world.py（观测项定义）
> 一句话：**只改 actor 的输入（+6 维绝对位姿），其余口径与 M4 完全一致，看 held-out 完成度动不动。**

---

## 0 目标与已知

### 0.1 要回答的问题

M4 在训练片段上完成度 0.999，但同一片段的后 20 s（held-out）只有 **0.101**、未见片段只有 **0.025**。
LAB20 把训练期 DR 加强后，held-out 反而掉到 **0.018**（≈官方同口径 0.016），于是 D34 给出一个判断：
**M4 的 0.101 是「训练窗口邻近帧的记忆外溢」，不是可迁移的模仿能力。**

那有没有可能这不是「记不住」而是「看不见」——策略的输入里根本没有判断自己漂没漂走的绝对基准？
本 lab 就测这一条：**把机器人根的世界系位置与线速度喂给 actor，其余一切不变。**

### 0.2 已知（引用前先看口径）

| 事实 | 数值 | 出处 |
| --- | --- | --- |
| M4 训练段完成度 | **0.999 ± 0.000**（10 种子） | LAB09 / LAB20 复跑一致 |
| M4 held-out（S3 120 s） | **0.100 ± 0.002** | LAB20 复跑 |
| M4 unseen（S2 134 s） | **0.025 ± 0.001** | LAB20 复跑 |
| 官方权重同口径 | train 0.333 / held-out 0.016 / unseen 0.013 | LAB09 |
| DR 加强后 | train 0.999 / held-out **0.018** / unseen 0.024 | LAB20，D34 |
| 参考动作本身不落地 | 训练段穿地 max 59.6 mm、接触 34%、策略骨盆 −1.62 cm 被动补偿 | LAB22，D41 |
| actor 观测现状 | 160 维：command（参考关节/身体目标）+ anchor 误差（pos 3 + ori 6）+ base_lin_vel 3 + base_ang_vel 3 + joint_pos 29 + joint_vel 29 + last_action 29 | tracking_env_cfg.py:45-79 |

### 0.3 改什么、不改什么

**只改 actor 的观测**（observation_manager 里 actor 组的 terms 末尾追加两项）：

| 新项 | 内容 | scale | 噪声 |
| --- | --- | --- | --- |
| root_pos_w | 机器人根（pelvis）世界坐标 3 维 | 0.5 | ±0.1 m |
| root_vel_w | 机器人根世界系线速度 3 维 | 0.5 | ±0.1 m/s |

→ actor 输入 **160 → 166**。critic 不动（它本来就有 body_pos/body_ori），奖励/终止/DR/动作空间/超参/迭代数/种子
全部与 M4 一致。**所以任何差异只能来自这 6 个数。**

### 0.4 为什么是「世界系根位置/速度」（机制假设，尚未验证）

现在 actor 能看到的位姿信息全是**相对于参考**的（motion_anchor_pos_b / motion_anchor_ori_b）。
这类输入有个盲点：如果机器人和参考**一起**平移/旋转，相对误差可以一直是 0，但世界坐标已经飘了。
追加绝对位姿后，策略理论上能区分两件事：

- 「我已经被推离了轨迹」（相对误差会变大，本来就能看到）；
- 「我还在相对误差阈值内，但绝对位置已经偏了 N 米」（新增信息）。

反面假设（必须写出来，否则这个 lab 就是在给自己找证据）：
**这个信息也可能是冗余的**——绝对位置 ≈ 参考绝对位置 + 相对误差，而参考绝对位置完全由 command 决定，
command 已经进了观测。若真如此，结果会是「无显著变化」，那也是一个有效结论（P7 的假设被否）。

**还有一个风险**：世界坐标在训练时只见过一段 20 s 的轨迹（x/y 范围很小），策略可能把它当 episode 相位标签用
（相当于给了一个额外的「计时器」），从而在 held-out 窗口上更加过拟合。所以 §0.5 的 B 种结果要特别当心。

### 0.5 三种结果怎么读（先定好，避免事后解释）

- **A 有效**：held-out 完成度 Δ ≥ **+0.05**（例如 0.100 → 0.16 以上），且 10 个种子里 ≥ 7 个同向变好，train 不掉。
- **B 换了个记忆方式**：held-out 涨了但 unseen 不动（或更差）→ 只说明「窗口记忆换了一种形式」，不能写「泛化提升」。
- **C 无显著变化（|Δ| < 0.01）或变差**：P7 的假设不成立 → 记成负结果，回到「数据/参考质量」这条线（D41 的 P9 候选）。

### 0.6 成本

| 阶段 | 内容 | 实测/估计 |
| --- | --- | --- |
| 训练 | 3000 iter / 4096 envs / 热启动 model_49999 → 52998 | **≈1 h 35 min**（LAB20 实测 1:35:01，同口径） |
| 评估 | 3 窗口 × 10 种子 = **30 次**（M4 基线已有，不重跑） | **≈20 min**（≈40 s/次，LAB20 实测） |
| 出表 | 表 + 图 | < 10 s |
| **合计** | | **≈2 h** |

---

## 1 协议（跑之前逐条核对）

| 项 | 值 |
| --- | --- |
| 热启动权重 | scratch/model_49999.pt（sha 前 8 … 后 6：**f1bc404f…f22319**） |
| 训练 ckpt 终点 | iter **52998**（与 M4 / LAB20 / LAB24 同一终点，便于对照） |
| 训练动作 | data/processed/fight1_s3_000s_20s.npz（sha **965f2c4b…b66c66**） |
| DR | mjlab 默认三项（摩擦 0.3–1.2 / 质心 ±2.5,±5,±5 cm / 编码器 ±0.01 rad）——**不加强** |
| 训练超参 | 4096 envs、num_steps_per_env 24、seed 0、save_interval 500 |
| 新增观测 | root_pos_w（scale 0.5、噪声 ±0.1 m）、root_vel_w（scale 0.5、噪声 ±0.1 m/s） |
| 评估窗口 | train = S3 0 s、heldout = S3 120 s、unseen = S2 134 s |
| 评估种子 | **101–110**（10 个，与 LAB20 基线同一批） |
| 评估口径 | scripts/eval_tracking.py，终止检测**开**、startup DR **关**、跑满动作帧数 |
| 基线 | runs/20261121-003-lab20-dr/eval/m4（M4，3 窗口 × 10 种子，LAB20 当时跑的） |

---

## 2 七步（照着做，命令可直接粘贴）

### 步骤 0：开工核对（≈2 min）

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
git status --short && git log --oneline -1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
sha256sum scratch/model_49999.pt data/processed/fight1_s3_000s_20s.npz
```

预期：工作树干净；GPU 是 RTX 3060 Laptop（6 GB），显存占用 < 1 GB；两个 sha 与 §1 对得上（前 8 / 后 6 即可）。

### 步骤 1：先写预测（≈5 min，不要跳过）

打开 docs/learning/LAB21-prediction.md，把 5 道题的答案写在每题下面的「我的答案」行。
**必须在下一条命令之前写完**——这是本项目所有 lab 的规矩（先预测，再看数）。

### 步骤 2：建 run 目录 + README 骨架（≈1 min）

```bash
RUN=runs/20261125-003-lab21-world-obs
mkdir -p "$RUN"
cat > "$RUN/README.md" <<'EOF'
# LAB21 actor 观测追加世界系根位置/速度（P7）

- 目的：只改 actor 输入（+6 维绝对位姿），其余与 M4 一致，看 held-out / unseen 完成度是否改善。
- 假设：相对位姿观测有盲点（机器人与参考一起漂时相对误差为 0）；追加世界系位姿后能提供独立基准。
- 配置：热启动 model_49999.pt、目标 iter 52998、动作 fight1_s3_000s_20s.npz、默认 DR、4096 envs、seed 0。
- 观测：actor 160 → 166 维（root_pos_w scale 0.5 噪声 ±0.1 m；root_vel_w scale 0.5 噪声 ±0.1 m/s）。
- 评估：3 窗口（S3 0 s / S3 120 s / S2 134 s）× 10 种子（101–110），终止开、startup DR 关。
- 结果：（跑完填：三窗口完成度、配对差、判定 A/B/C）
EOF
ls "$RUN"
```

预期：目录里只有一个 README.md。

### 步骤 3：训练（≈1 h 35 min，可中断续训）

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
RUN=runs/20261125-003-lab21-world-obs
mkdir -p "$RUN"
MUJOCO_GL=egl environment/.venv/bin/python scripts/lab21_obs_train.py \
  --run-root "$RUN" --mode full --resume 2>&1 | tee "$RUN/train.log"
```

**开跑后 2 分钟内做两个自检**（另开一个终端）：

```bash
RUN=runs/20261125-003-lab21-world-obs
grep -c "root_pos_w" "$RUN"/logs/g1_tracking/*/params/env.yaml
grep -c "160 -> 166" "$RUN"/train.log
```

预期：第一条输出 **1**（env.yaml 里确实写入了新观测项）；第二条输出 **1**（日志里有「actor 第一层 160 -> 166」）。

**中断了怎么办**：直接重跑步骤 3 的同一条命令。脚本会找本 run 最新的 model_*.pt，
自动算「还差多少 iter」并续训到 52998；已经到 52998 时只打印「无需再训」并退出。

预期结束：$RUN/logs/g1_tracking/<日期>_lab21-world-obs/ 下有 model_52998.pt，
train.log 末尾是 Total time 收尾行（resume 之后 Iteration time 可能显示负数，看 Total time）。

### 步骤 4：评估（≈20 min）

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
LAB21_RUN=runs/20261125-003-lab21-world-obs bash scripts/lab21_eval.sh run 2>&1 | tee runs/20261125-003-lab21-world-obs/eval.log
```

评估脚本会：打印 LAB21 ckpt 与 M4 ckpt 的 sha → 逐窗口逐种子跑 30 次 → 自动调 report 出表出图。
**中断了怎么办**：重跑同一条命令，已有 summary.json 的格会自动跳过。

预期：日志末尾有一行「评估墙钟（/proc/uptime）≈1200 s」，并生成
runs/20261125-003-lab21-world-obs/lab21_obs.md、lab21_obs.json、outputs/figures/lab21_obs.png。

### 步骤 5：按顺序读数（≈10 min）

1. **先看 lab21_obs.md 的「逐窗口配对差」表**——held-out 那一行的「完成度 Δ 均值」与「好/差 种子数」。
2. 再回头看「策略 n 完成度」表里的 train 行：**train 掉了没有**（掉 ≥0.05 说明新输入干扰专精）。
3. 看 R-MPKPE Δ（去漂移的姿态误差）：它若同时变好，改善才可信。
4. 最后看 unseen：动没动。
5. 按 §0.5 的 A/B/C 三支给一个判定，写在 run README 的「结果」行。

### 步骤 6：同步八处文档（≈20 min）

1. runs/20261125-003-lab21-world-obs/README.md（填全，参考 LAB22 / LAB26 的 README 结构）
2. docs/REPORT.md：新增 §2.5.20（表 + 判读 + 图），§3 关键发现补一条，§4 局限补一条
3. docs/DECISIONS.md：新增 **D42**（依据 / 实测 / 判读 / 决定）
4. docs/STATE.md：头部「当前阶段」句 + 末尾新节「## LAB21 结果（…）」
5. docs/PLAN.md：LAB21 行标 [x] 并写实测
6. docs/ARTIFACT_INDEX.md：插「## 2026-11-25（LAB21 世界系根位姿观测）」
7. README.md 结果表：新增一行「观测改造（LAB21）」
8. 本讲义 §7 + docs/learning/LAB21-prediction.md 的判分与参考回答

### 步骤 7：提交并 push（≈2 min）

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
git add -A
git status --short | head -20
git commit -m "feat(lab21): actor 观测追加世界系根位置/速度——held-out 完成度 X（D42）"
bash scripts/lab21_push.sh
```

（scripts/lab21_push.sh 随讲义提供，内容就是本仓库惯用的带 credential.helper 的 push 命令。）

---

## 3 通过判定（八条，全部满足才算这个 lab 做完了）

1. 开工时 git status 干净；nvidia-smi 显示 6 GB 的 3060 Laptop。
2. 步骤 1 的预测**在训练开始前**已写进 LAB21-prediction.md。
3. env.yaml 里能看到 root_pos_w / root_vel_w 两项，且日志有「actor 第一层 160 -> 166」。
4. 训练到 **model_52998.pt**，且 train.log 里没有 NaN / 崩溃。
5. 评估产生 **30 个** summary.json（3 窗口 × 10 种子），没有失败重试痕迹。
6. lab21_obs.md 的配对差表里三个窗口都有数（不是「数据不足」）。
7. run README 的「结果」行写了 **判定属于 A / B / C 哪一支**，并给了 Δ 与好/差种子数。
8. 八处文档 + 讲义 §7 + 预测判分都更新，且有 commit 与 push（git log --oneline -1 能看到）。

---

## 4 常见问题（FAQ）

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| KeyError: actor_state_dict | 用了旧格式 ckpt 且没走迁移 | 仓库脚本已内置迁移；自己改脚本时按 mjlab 的 actor.* -> mlp.* 规则先转格式 |
| size mismatch for mlp.0.weight | 评估时漏了 --obs-world-root，或训练没加观测 | 评估走 lab21_eval.sh（obs 策略自动带开关）；手动 eval 时加 --obs-world-root |
| 训练一开始 loss 很大 | 新维度归一化器用的是先验（mean 0 / std 1） | 前 100 iter 允许抖动；500 iter 还不降再看 value loss 是否发散 |
| 找不到 LAB21 ckpt | 训练没跑完，或误取了 pretrained/ 里的热启动副本 | 确认 logs/g1_tracking/<日期>_lab21-world-obs/model_*.pt 存在 |
| motion file not found | 不在仓库根目录 | 先 cd 到仓库根再跑 |
| lab21_obs.md 出现「数据不足」 | 基线目录被改 | 默认基线是 runs/20261121-003-lab20-dr/eval/m4；用 LAB21_BASELINE= 指定别的 |
| resume 后 Iteration time 是负数 | 计时器从续训点归零 | 忽略，看 Total time |

---

## 5 五个坑

1. **别把「训练段还在 0.999」当成实验成功**：这个 lab 的唯一主指标是 held-out 的 Δ，train 只是护栏。
2. **别用单遍/单种子下结论**：10 个种子 101–110 必须与 LAB20 基线同一批才可配对；换种子就不是配对比较。
3. **别把 held-out 的改善直接写成「泛化提升」**：必须同时看 unseen（D34 的教训：0.101 → 0.018 就是被当成泛化过）。
4. **别改 scale/噪声后只重训一半**：训练与评估必须用同一套定义（都在 scripts/obs_world.py），改一处就整体重来。
5. **别把 LAB21 的 ckpt 混进老的扰动结论**：这是新的主力 ckpt 候选，扰动（LAB12 协议）要另跑一轮才能比较。

---

## 6 分支预案（跑完按结果选一支）

- **A 有效（held-out Δ ≥ +0.05 且 ≥7/10 同向、train 不掉）**：
  1) 补跑 unseen 的 20 种子扩展（10 → 20，≈13 min），确认不是 10 种子的偶然；
  2) 用 LAB12 协议测抗推（60 N·s 两向 × 20 种子 × 3 遍 ≈64 min），看「绝对位姿」是否也让退让变轻；
  3) 把 LAB21 列为新的主力 ckpt 候选，写进 DECISIONS（D42）。
- **B 只换记忆方式（held-out 涨、unseen 不动）**：
  写成「窗口记忆的可替代形式」，不写泛化；下一步转数据侧（多片段训练）而不是继续调观测。
- **C 无变化 / 变差**：
  记为负结果：P7 的「绝对位姿假设」被否；把精力放回 D41 的 P9（修参考）+ 数据侧扩容。
- **训练发散（train < 0.8 或 NaN）**：
  把 root_pos_w 的 scale 从 0.5 降到 0.2 再重训（只改这一处），并在 README 记录这次改动。

---

## 7 结果（跑完填）

- LAB21 ckpt：model_52998.pt，sha __
- 训练墙钟：__（train.log 的 Total time）；中断/续训：有/无
- train：M4 __ → LAB21 __（Δ __）
- held-out：M4 __ → LAB21 __（Δ __，好 __/10、差 __/10）
- unseen：M4 __ → LAB21 __（Δ __）
- R-MPKPE：train __ / held __ / unseen __（LAB21 − M4 的 Δ）
- 判定：**A / B / C**（理由一行）
- 一句话结论：（不超过两行）

---

维护：本讲义由助教在 2026-11-25 生成（对应 P7 / PLAN LAB21）。改观测定义、scale、噪声或判定线时，
必须同步改 scripts/obs_world.py、本讲义 §0.3/§3 与 D42。
