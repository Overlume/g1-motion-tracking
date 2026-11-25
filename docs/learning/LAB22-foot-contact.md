# LAB22 参考动作的脚底接触 / 穿地校验（M2 遗留项）

> 讲义对象：你（操作者）。本讲义自足，不需要先读别的 lab 文档；命令里的路径已写死，可以直接复制执行。
> 这个是**零 GPU** 的验证类 lab（CPU ≈2 min），不需要 nvidia-smi，也不需要评估 120 次。

---

## 0. 目标与已知

### 0.1 一句话目标

给 M2（重定向）补上唯一缺失的验收证据：**我们拿来训练和评估的参考动作，在「脚踩平地」这件事上到底站不站得住**——逐帧 FK 出两脚的脚底板高度，量化穿地深度、接触比例、悬空时长，再和策略自己的骨盆高度对一下，说清「参考不落地」是被谁吸收的。

### 0.2 为什么必须做

1. **M2 的验收清单里只剩这一项没勾**（PLAN 的 LAB22 / P8）。不做，M2 的「重定向物理可行性」就只是断言。
2. **它直接影响怎么解释已有数字**：M4 在 S3 0–20 s 上完成度 0.999、root 误差 3.90 cm。如果这段参考本身有帧在穿地（脚陷到地面以下），那 3.90 cm 就不能全算在策略头上——策略在追一个物理上做不到的目标。
3. **它给后续 DR/观测改造定基线**：如果参考悬空 3 cm，任何「加强扰动/加观测」的收益里都混着这一层系统性误差。
4. 成本极低：纯 CPU FK，没有训练、没有 rollout，几分钟出表。

### 0.3 口径（三个指标）

| 指标 | 定义 | 怎么判 |
| --- | --- | --- |
| **穿地深度** | 最低脚底 z 为负时，|z| 的值（mm），逐帧算 | max ≤20 mm 且 p99 ≤20 mm = OK；≤50 mm = WARN；>50 mm = FAIL |
| **接触比例** | 至少一足脚底 z <20 mm 的帧占比 | 跳舞/格斗类参考允许跳跃；但要能报出来（并给双足支撑、腾空比例） |
| **悬空时长** | 两足都 >20 mm 的最长连续帧段（秒） | 超过 0.5 s 就要在报告里点名——那不是跳跃，是「整体飘着」 |

> 20 mm 这个阈值来自脚底板几何：G1 的脚底碰撞是 7 块 box（底 z = −0.025 m 相对 ankle_roll），2 cm 是「半个脚掌厚度」的量级；50 mm 是「明显不可能用接触力解释」的量级。

### 0.4 附带的第二个问题（Part B）

参考穿地时，策略怎么办？两种可能：

- **跟着参考高度走**：骨盆 offset ≈ 0 → 穿地只能被腿/脚吸收（表现为关节误差、脚滑）；
- **整体抬高**：骨盆 offset 明显为正 → 策略在悬空，完成度与 root 误差要扣这一层。

用已有评估的 metrics.csv（列里有 pelvis_z 与 ref_pelvis_z）就能算，1 秒钟。

### 0.5 成本

- Part A：4 段参考、约 9600 帧 FK，实测 **≈82 s**（单核 CPU，无 GPU）。
- Part B：读 5 个 metrics.csv，**<1 s**。
- 合计 **≈2 min**，不含写文档。

### 0.6 本 lab 不做的事

- 不重定向、不改动作、不重训、不评估。
- 不扫全部 9 段素材，只查已被使用的 4 段（训练 1 + held-out 1 + unseen 1 + 官方用 dance1）。
- 不做「接触力」级别的校验（那要真跑 sim），只做运动学几何校验。

---

## 1. 输入与协议

| 项 | 值 |
| --- | --- |
| 机器人模型 | repos/mjlab_research/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml（sha256 前 8 位 febdcbef） |
| 脚底几何 | left/right_ankle_roll_link 下各 7 块 collision box（contype=1），底 z = −0.025 m |
| 运动片段 1（训练） | data/processed/fight1_s3_000s_20s.npz（sha 965f2c4b…b66c66，20.0 s，999 帧 @50 fps） |
| 运动片段 2（held-out） | data/processed/fight1_s3_120s_20s.npz（sha 4aac9984…1da77b） |
| 运动片段 3（unseen） | data/processed/fight1_s2_134s_20s.npz（sha 4920d3ef…d6e2ef） |
| 运动片段 4（官方基线用） | data/processed/lafan_dance1_subject1.npz（sha 6a19ac79…9d9684，131.5 s，6574 帧） |
| FK | 逐帧把 root_pos_w + root_quat_w(wxyz) + 29 个 joint_pos 写进 qpos，mj_kinematics 后取脚底 box 角点最低 z |
| 接触阈值 | 20 mm（一足接触）；50 mm（腾空） |
| 穿地判据 | max ≤20 mm 且 p99 ≤20 mm = OK；≤50 mm = WARN；>50 mm = FAIL |
| Part B 数据 | runs/20261117-002-m4-eval/final/seed101–105/metrics.csv（pelvis_z、ref_pelvis_z 两列） |

> 注意：dance1 的 npz 里**没有 fps 字段**（其余三段有），脚本会按 50 fps 处理并在表里标出——这不影响「逐帧」指标，只影响「秒」的换算。

---

## 2. 操作：七步

### 步骤 0｜开工核对（约 1 min）

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
git status --short
ls -l data/processed/*.npz
sha256sum data/processed/fight1_s3_000s_20s.npz | cut -c1-16
environment/.venv/bin/python -c "import mujoco; print('mujoco', mujoco.__version__)"
```

预期：

- 工作树干净；
- 4 个 npz 都在（s3_000s、s3_120s、s2_134s、lafan_dance1_subject1）；
- sha256 以 **965f2c4b** 开头；
- mujoco 版本 3.16.0（3.2+ 都行）。

### 步骤 1｜先写预测（约 5 min）

打开 docs/learning/LAB22-prediction.md，在每题「你的答案」下写答案。先写再看结果。

### 步骤 2｜建 run 目录与 README 骨架（约 2 min）

```bash
RUN=runs/20261125-002-lab22-foot-contact
mkdir -p "$RUN"
cat > "$RUN/README.md" <<'EOF'
# LAB22 参考动作脚底接触 / 穿地校验

- 目的：给 M2 补物理可行性证据——参考动作的脚底穿地深度、接触比例、悬空时长；Part B 看策略骨盆是否比参考高。
- 假设：训练用片段（s3_000s）存在 2 cm 级穿地或 3 cm 级悬空；策略骨盆 offset 在 ±2 cm 内（跟着参考走）。
- 输入：4 段 npz（s3_000s / s3_120s / s2_134s / dance1）+ G1 MJCF（febdcbef）+ M4 评估的 metrics.csv。
- 命令：见 docs/learning/LAB22-foot-contact.md §2。
- 结果：（跑完填：四段判定、最深穿地帧、接触比例、策略 offset）
- 下一步：（填）
EOF
ls "$RUN"
```

### 步骤 3｜跑 Part A：参考动作审计（约 82 s）

```bash
RUN=runs/20261125-002-lab22-foot-contact
environment/.venv/bin/python scripts/lab22_foot_audit.py --out-dir "$RUN" --figure 2>&1 | tee "$RUN/audit.log"
```

预期：屏幕打 4 行 [audit]，每行给 pen_max、sole p50、contact either、判定；最后写盘 foot_audit.md / foot_audit.json + 图 outputs/figures/lab22_foot_audit.png。

### 步骤 4｜跑 Part B：策略骨盆偏移（<1 s）

```bash
RUN=runs/20261125-002-lab22-foot-contact
environment/.venv/bin/python scripts/lab22_policy_offset.py --out-dir "$RUN" 2>&1 | tee "$RUN/policy_offset.log"
```

预期：5 行 [offset]（seed101–105），每行给中位 offset 与 |Δ|>2 cm 占比；写盘 policy_offset.md / .json。

### 步骤 5｜读数顺序（按这个顺序看，别跳）

1. **foot_audit.md 的表**：四段的「穿地 max」和「判定」；
2. **哪一段最差**：训练用的 s3_000s 是不是 FAIL？最深穿地在第几帧、第几秒（表下面有逐片段细节）；
3. **接触比例**：s3_000s 的「接触(任一足)」是多少？「最长悬空」多少秒？
4. **对照 dance1**：官方基线用的舞蹈片段接触比例明显更高还是更低？（判断「飘」是 fight1 的问题还是重定向流程的问题）
5. **policy_offset.md**：5 个种子的 offset 中位是正还是负？|Δ|>2 cm 占比多少？

### 步骤 6｜文档同步（八处，可先留空由我代填）

1. runs/20261125-002-lab22-foot-contact/README.md 的「结果」「下一步」两行；
2. docs/REPORT.md：在「## 3. 关键发现」前新建 §2.5.19（表 + 4–6 条 + 图）；
3. docs/DECISIONS.md：末尾加 D41；
4. docs/STATE.md：头部阶段句 + 末尾「## LAB22 结果」节；
5. docs/PLAN.md：LAB22 勾 [x] + 实测一句；
6. docs/ARTIFACT_INDEX.md：新建「## 2026-11-25（LAB22 脚底接触校验）」节；
7. README.md：结果表加一行；
8. docs/learning/LAB22-foot-contact.md：§7 填实测。

改不动就只做第 1、8 项，其余我收尾时补。

### 步骤 7｜提交并推送

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
git add docs/learning/LAB22-foot-contact.md docs/learning/LAB22-prediction.md docs/PLAN.md \
  scripts/lab22_foot_audit.py scripts/lab22_policy_offset.py \
  runs/20261125-002-lab22-foot-contact/README.md \
  runs/20261125-002-lab22-foot-contact/foot_audit.md \
  runs/20261125-002-lab22-foot-contact/foot_audit.json \
  runs/20261125-002-lab22-foot-contact/policy_offset.md \
  runs/20261125-002-lab22-foot-contact/policy_offset.json \
  outputs/figures/lab22_foot_audit.png
git status --short
git commit -m "feat(lab22): 参考动作脚底接触/穿地校验（M2 遗留项）"
GIT_TERMINAL_PROMPT=0 git -c 'credential.helper=!"/mnt/c/Program Files/Git/mingw64/bin/git-credential-manager.exe"' push origin main
```

---

## 3. 通过判定（八条，全过才算做完）

1. run 目录里有 foot_audit.md/.json、policy_offset.md/.json、audit.log、policy_offset.log 六件；
2. foot_audit.md 的表有 4 行（s3_000s / s3_120s / s2_134s / dance1），每行都有判定（OK / WARN / FAIL）；
3. 四段的「穿地 max」都是数值（不是 NaN），且表下有 4 条逐片段细节（含最深帧号与时刻）；
4. 你能说出「哪一段穿地最严重、第几秒」；
5. 你能说出 s3_000s 的「接触(任一足)」与「最长悬空」；
6. dance1 的接触比例明显高于 fight1 三段（若不是，说明 FK 或阈值有问题，去 §4 排查）；
7. policy_offset.md 有 5 行，offset 中位与 |Δ|>2cm 占比都是数值；
8. outputs/figures/lab22_foot_audit.png 生成，左图四条曲线、右图 4 根柱子。

---

## 4. 常见问题（FAQ）

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 报「缺片段」 | npz 名字不对（dance1 是 lafan_dance1_subject1.npz，不是 lafan_dance1.npz） | 用 §1 表里的名字 |
| 报「关节名不在模型里」 | npz 的 joint_names 与 G1 MJCF 不一致 | 检查是不是别的机器人的 npz |
| 接触比例全是 0 | 忘了 mj_kinematics，或 qpos 里 quat 当成 xyzw 写 | 脚本已处理（wxyz）；若改脚本别踩这个 |
| 表里 dance1 的时长不是 131.5 s | 该 npz 没有 fps 字段，按 50 fps 换算 | 正常，逐帧指标不受影响 |
| 判定全是 FAIL | 阈值 20/50 mm 太严？先看「穿地 p99」而不是只看 max | p99 很小、max 大 = 个别帧的尖峰，报告里写清帧号 |
| Part B 报「没有找到 metrics.csv」 | 默认 glob 指向 runs/20261117-002-m4-eval/final/*/metrics.csv | 用 --metrics 显式给路径 |
| Part B 的 offset 有正有负怎么读 | 正 = 策略骨盆比参考高（悬空），负 = 低（往下够地面） | 看中位数的符号与 |Δ|>2cm 占比 |

---

## 5. 六个坑

1. **npz 的 root 不是「地面接触点」**：root_pos_w 是骨盆（≈0.71–0.82 m），别拿它当脚高。
2. **四元数顺序**：npz 是 wxyz（mjlab 约定），写成 xyzw 会让整段姿态错 90°，脚高全错。
3. **脚底不是「body 原点 − 0.025」这么简单**：脚踝 roll 连杆会随踝关节转动，必须用 box 角点做旋转后的最低 z（脚本里就是这么做的）。
4. **接触阈值别乱改**：20 mm 是「半个脚掌厚度」，改了阈值结论会变；要改先改讲义 §0.3 再改脚本。
5. **「穿地」不是「策略失败」**：参考是运动学的、可以穿地；策略是物理的、不能。穿地指标衡量的是参考质量，不是策略质量。
6. **dance1 没有 fps**：如果拿它算「时长」会得到 131.5 s（50 fps 假设），但只要不拿它和别的片段比绝对秒数就没事。

---

## 6. 分支预案

- **s3_000s 判定 FAIL（max ≥50 mm）**：在 REPORT 里写「训练参考存在 ≥2 cm 穿地」，并把 M4 的 3.9 cm root 误差拆成「参考不可行成分」与「策略误差」；D41 记一条「以后重定向产物必须过这道闸（阈值 20 mm）」。
- **四段全 FAIL**：说明是流程问题（GMR 高度标定），不是单段问题；这时 Part B 的 offset 会更有解释力（策略在两侧「够地面」）。
- **四段全 OK**：M2 验收通过，穿地不是解释项；把结论写成「参考物理可行性通过，root 误差归策略」。
- **Part B 的 offset 中位明显为正（>2 cm）**：策略在悬空——要检查是不是评估用的初始高度/地面参数有偏移，这条比 Part A 更严重。
- **Part B 的 offset 中位在 ±1 cm 内**：策略跟着参考走 → 参考的穿地/悬空被关节吸收，后续要看 joint_rmse 与脚滑（metrics.csv 里已有 slip_left_cm / slip_right_cm）。

---

## 7. 结果（跑完填）

- Part A 判定：s3_000s __、s3_120s __、s2_134s __、dance1 __。
- 最深穿地：__ 片段，__ mm，第 __ 帧（__ s）。
- 接触(任一足)：s3_000s __%、dance1 __%；最长悬空：s3_000s __ s。
- Part B：offset 中位 __ cm（正/负），|Δ|>2 cm 占比 __%。
- 一句话结论：（不超过两行）

---

维护：本讲义由助教在 2026-11-25 生成；改阈值或脚本前先同步改 §0.3 与 D41。
