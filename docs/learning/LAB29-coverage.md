# LAB29 全段覆盖训练：把 dance1 六个窗口全喂进去

> 一句话：LAB28 说「记忆没有可测的时间半径，增益更像内容相似」。本 lab 反过来做一次因果检验：
> **把 dance1 的六段全部 + fight1 训练窗口一起训**，看六个窗口会不会一起被拉起来。

## 0. 先看清要回答的问题

### 0.1 为什么问这个

LAB28 的结果（同一段 dance1，六个 20 s 窗口，10 种子）：

| 窗口 | M4（只练 fight1 S3 0 s） | LAB27（+dance1 0–20 s） |
|---|---|---|
| dance1 0–20 s | 0.484 | **0.999** |
| dance1 20–40 s | 0.070 | 0.244 |
| dance1 40–60 s | 0.115 | **0.055** |
| dance1 60–80 s | 0.061 | 0.108 |
| dance1 80–100 s | 0.191 | 0.309 |
| dance1 100–120 s | 0.064 | 0.101 |

曲线**不单调**（40 s 掉、80 s 弹），所以不能说「记忆半径 <20 s」，也不能说「60 s 后失效」。零 GPU 分析给的相关证据是：**新颖度（窗口到训练窗口的关节空间最近距离）vs 增益 Spearman −0.886（p=0.019）**，而纯时间距离只有 −0.600（p=0.208）。

但相关不是因果。能把它变因果的实验只有一个：

> 把缺的五个窗口全部补进训练。如果「内容相似」是主因，六个窗口应该**一起**上去；如果「只有练过的那段会」，那就是 0 s 高、其余不动。

这就是 LAB29 的全部内容。

### 0.2 已知数字（同一套口径）

| 对象 | M4（1 段） | LAB27（2 段） |
|---|---|---|
| fight1 S3 0–20 s（训练窗口） | 0.999 | 0.999 |
| fight1 S3 120 s（同片段未见） | 0.100 | 0.097 |
| fight1 S2 134 s（换片段） | 0.025 | 0.026 |
| dance1 六窗口 | 见上表 | 见上表 |

LAB27 的经验很重要：**多加一段 dance1，只把 dance1 抬起来，fight1 的三个窗口一动不动**。所以本 lab 的预期也有两种读法——「按动作隔离」和「按窗口覆盖」。

### 0.3 三种可能的结果，各自意味着什么

| 读法 | 表长相 | 结论与下一步 |
|---|---|---|
| **(A) 覆盖有效** | dance1 六窗口里 ≥5 个 Δ（LAB29 − LAB27）≥ +0.02，且没有窗口掉 ≤ −0.02 | 覆盖率是主因 → 数据侧路线成立，下一步是多动作覆盖（LAB30）与数据规模曲线 |
| **(B') 局部提升** | 只有 0 s（和可能的 20 s）变好，其余持平或变差；Δ ≥ +0.02 的窗口 ≤2 | 覆盖不是主因，练过 ≠ 会 → 回到方法侧（观测/参考/训练目标） |
| **(C) 部分覆盖** | 3–4 个窗口好、其余不动或变差 | 部分因果：要逐窗口对照 dmin / 关节速度 / 接触，先解释「哪类窗口练了也没用」 |

另外必须单独看一条：**训练窗口（fight1 S3 0 s）有没有掉**。7 段共用 3000 iter、同一个网络，容量可能被摊薄；train 掉 >0.1 的话，后面所有对比都要打折。

### 0.4 为什么是 7 段，为什么不再加别的

- **7 段** = fight1 S3 0–20 s（999 帧，保住原训练窗口）+ dance1 0/20/40/60/80/100 s（各 1000 帧）= 6999 帧。
- 为什么不把 fight1 S3 120 s、S2 134 s 也训进去：它们扮演**held-out / unseen**，训进去就再也问不了泛化；那是 LAB30 的事。
- 为什么不换大网络、不加 iter：AGENTS 的单因素原则——本 lab 只改「训练数据覆盖」，其余与 M4 / LAB27 完全一致（3000 iter、4096 envs、seed 0、热启动 model_49999）。
- 为什么段与段可以任意拼：mjlab 的 MotionLoader 会按 motion_ends 记住每段边界，episode 走到片段末帧就重采样（commands.py:450），**不会跨段**。所以 7 段拼起来等价于「7 个独立 20 s 任务轮着采」。

### 0.5 成本

| 步骤 | 内容 | 时间 |
|---|---|---|
| 步骤 3 | 6 个窗口地面校正（已存在，全部跳过） | <1 min |
| 步骤 4 | 7 段打包 + 训练 3000 iter（4096 envs） | ≈1 h 50 |
| 步骤 5 | 9 窗口 × 10 种子 = **90 次评估** | ≈56 min（按 37.6 s/次） |
| 步骤 6 | 出表 + 图 | <10 s |
| **合计** | | **≈2 h 50** |

## 1. 协议（只改训练数据，别的一律不动）

| 项 | 值 |
|---|---|
| 片段顺序 | data/processed/fight1_s3_000s_20s.npz（999 帧）→ dance1_000s / 020s / 040s / 060s / 080s / 100s_20s_gfix.npz（各 1000 帧） |
| packed 目录 | <run>/motion_packed（每字段 .npy + motion_ends.npy + joint_names.json + motion_names.json） |
| 期望 motion_ends | [999 1999 2999 3999 4999 5999 6999]，合计 6999 帧 |
| 热启动 | scratch/model_49999.pt（sha **f1bc404f**…**f22319**） |
| 训练 | 3000 iter、4096 envs、num_steps_per_env 24、seed 0、save_interval 500 → 终点 iter **52998** |
| 评估 | scripts/eval_tracking.py：终止检测**开**、启动 DR 关（默认）、999 步 |
| 评估窗口 | train = fight1 S3 0 s；heldout = fight1 S3 120 s；unseen = fight1 S2 134 s；dance1 六窗口（w000…w100） |
| 种子 | 101–110（10 个） |
| 指标 | track_completion（到过的最远帧 / 总帧）、motion_frame_furthest（999 = 跑完）、reset_count、mpkpe_cm |
| 对照 | M4（fc8f975b…ceed9，1 段）、LAB27（8745f96e…125dac，2 段），历史数字直接复用（同协议同种子） |

> ⚠ 口径提醒：非训练窗口完成度普遍 0.02–0.31，**1 帧 = 0.001**。Δ 差 0.01（≈10 帧）才算有点差；0.003 是噪声。判据用的是 **+0.02**（≈20 帧）。

## 2. 七步

### 步骤 0：开工核对（≈1 min）

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
git log --oneline -1          # 期望 bb68f58 feat(lab28): 窗口距离曲线……
git status --short            # 期望干净（顶多 scratch/）
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
sha256sum runs/20261117-001-m4-w15-finetune/logs/g1_tracking/*/model_52998.pt
sha256sum runs/20261125-004-lab27-multiclip/logs/g1_tracking/*/model_52998.pt
ls data/processed/dance1_*_gfix.npz | wc -l        # 期望 6
environment/.venv/bin/python -c "import numpy as np; d=np.load('data/processed/dance1_000s_20s_gfix.npz'); print(d['joint_pos'].shape)"
```

自检：
- M4 ckpt sha 必须是 **fc8f975b…ceed9**，LAB27 是 **8745f96e…125dac**（不是 f1bc404f… 那个热启动权重）。
- 6 个 dance1 窗口都在，且每个是 **(1000, 29)**。
- 训练要 4096 envs，GPU 至少留 **4 GB** 空闲。

### 步骤 1：写预测（≈5 min）

打开 docs/learning/LAB29-prediction.md，5 题都先写死数字/选项，**训练之前**提交。

### 步骤 2：建 run 目录 + README 骨架（≈2 min）

```bash
RUN=runs/20261127-001-lab29-cover
mkdir -p "$RUN"
cat > "$RUN/README.md" <<'EOF'
# LAB29 全段覆盖训练（fight1 S3 0–20 s + dance1 六窗口，共 7 段）

- 目的：检验「记忆的增益是内容相似」这个假设——把 dance1 六段全覆盖进训练，六个窗口会不会一起被拉起来。
- 假设：(A) 一起变好 / (B') 只有 0 s 好 / (C) 部分窗口好。
- 配置：7 段（999 + 6×1000 帧）× 3000 iter × 4096 envs × seed 0，热启动 scratch/model_49999.pt；评估 9 窗口 × 种子 101–110。
- 命令：见 docs/learning/LAB29-coverage.md §2。
- 结果：（跑完填：motion_ends、训练墙钟、三策略表、属于 A/B'/C）
- 下一步：（填）
EOF
ls "$RUN"
```

自检：`ls "$RUN"` 只应看到 README.md；`ls runs/*.md` 不应有输出（runs 根目录不能有文件）。

### 步骤 3：补窗口数据（≈1 min）

```bash
RUN=runs/20261127-001-lab29-cover
LAB29_RUN="$RUN" bash scripts/lab29_cover.sh build
```

期望输出结尾是「7 段齐：data/processed/fight1_s3_000s_20s.npz + 6 个 dance1 窗口」，前面 6 行是「已有 …（sha xxxxxxxx），跳过」。若某个窗口缺失，脚本会现场生成并过 D41 闸门（穿地 max ≤20 mm 且 p99 ≤20 mm，不达标直接退出）。

### 步骤 4：训练（≈1 h 50，可中断续训）

```bash
RUN=runs/20261127-001-lab29-cover
mkdir -p "$RUN"
LAB29_RUN="$RUN" bash scripts/lab29_cover.sh train 2>&1 | tee "$RUN/train.log"
```

日志里要看到的（前 20 行内）：

```
[lab27] 已打包 7 段 -> <run>/motion_packed
[lab27] motion_ends（累计帧）= [ 999 1999 2999 3999 4999 5999 6999]，合计 6999 帧
[lab27]   fight1_s3_000s_20s.npz：999 帧
[lab27]   dance1_000s_20s_gfix.npz：1000 帧
...
```

**中断了怎么办**：直接重跑同一条命令。脚本（`lab29_cover.sh train` → `scripts/lab27_multiclip_train.py --resume`）会找本 run 里最新的 ckpt，从它续训到 iter 52998，不会从头再来。日志里会打印「续训：从 …/model_XXXXX.pt（iter N）再训 K 个 iter」。

自检：
```bash
RUN=runs/20261127-001-lab29-cover
ls "$RUN"/logs/g1_tracking/*/model_*.pt | grep -v pretrained | sort -V | tail -3
grep -n "motion_file" "$RUN"/logs/g1_tracking/*/params/env.yaml
```
- 最大 ckpt 必须是 **model_52998.pt**。
- params/env.yaml 里 `commands.motion.motion_file` 应指向本 run 的 `motion_packed` 目录（不是某个 .npz）。

⚠ `train.log` 会有几百 MB（LAB20 那次 840 MB）。它在 `runs/` 下、被 .gitignore 忽略，**不要 git add**，也别删（出问题要查）。

### 步骤 5：评估（≈56 min）

```bash
RUN=runs/20261127-001-lab29-cover
LAB29_RUN="$RUN" bash scripts/lab29_cover.sh eval 2>&1 | tee "$RUN/eval.log"
```

期望：先打印 ckpt 路径与 iter（必须 ≥52998），然后是 90 行「lab29/<窗口>/seed<k>」，最后一行评估墙钟。

自检：
```bash
RUN=runs/20261127-001-lab29-cover
find "$RUN/eval/lab29" -name summary.json | wc -l     # 期望 90
environment/.venv/bin/python -c "import json,glob; f=sorted(glob.glob('$RUN/eval/lab29/w060/seed*/summary.json'))[0]; d=json.load(open(f)); print(d.get('checkpoint')); print(d.get('track_completion'))"
```
- 90 个 summary.json；checkpoint 字段应指向本 run 的 model_52998.pt（不是 pretrained 副本）。
- 想重跑某几个格子：`LAB29_FORCE=1` 会无视已有 summary.json 全部重跑（慎用）。

### 步骤 6：出表 + 读数顺序（<1 min）

```bash
RUN=runs/20261127-001-lab29-cover
LAB29_RUN="$RUN" bash scripts/lab29_cover.sh report
LAB29_RUN="$RUN" bash scripts/lab29_cover.sh show
```

读数顺序（照报告 §4）：
1. **train** 掉没掉：掉 >0.1 = 容量被摊薄，后面结论要打折。
2. **dance1 六窗口的 Δ**：一起好（≥5 个 ≥+0.02）→ (A)；只有 0 s（≤2 个）→ (B')；中间 → (C)。
3. **heldout / unseen**：跨动作，不该因为多练 dance1 而变好；明显变差说明容量被占。
4. 最后看**存活帧**：完成度会被窗口难度污染（LAB28 已证），存活帧 + 首段对齐误差才是跨窗口可比的量。

### 步骤 7：填 README + 文档同步 + 提交（≈15 min）

```bash
RUN=runs/20261127-001-lab29-cover
ls runs/*.md 2>/dev/null    # 必须无输出
git status --short
```

然后按 HANDOFF §8 同步八处（REPORT / DECISIONS / STATE / PLAN / ARTIFACT_INDEX / README / run README / 讲义 §7），提交信息用：

```
feat(lab29): 全段覆盖训练——<一句话结论>；D45
```

推送（credential helper 必须整体单引号，否则路径被空格拆开）：

```bash
GIT_TERMINAL_PROMPT=0 git -c 'credential.helper=!"/mnt/c/Program Files/Git/mingw64/bin/git-credential-manager.exe"' push origin main
```

## 3. 通过判定（做完的打勾）

- [ ] run 目录里有 README，写清目的/假设/配置/命令
- [ ] 训练日志里 motion_ends = [999 1999 2999 3999 4999 5999 6999]，7 段帧数正确
- [ ] 最新 ckpt 是 model_52998.pt，params/env.yaml 的 motion_file 指向 motion_packed
- [ ] eval 里有 90 个 summary.json，checkpoint 字段指向本 run 的 52998
- [ ] lab29_cover.md 三策略表 9 窗口齐全（M4/LAB27 无「—」）
- [ ] 报告自检行打印「lab29 格子 90 个」
- [ ] 明确写出属于 (A) / (B') / (C)，并给窗口级 Δ 数字
- [ ] README 填全 + 八处文档同步 + 已 push

## 4. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 训练启动就 `ValueError: … provide --env.commands.motion.motion-file` | 直接把 npz 列表传给了环境（train.py 只接受单个存在路径） | 必须用 `scripts/lab29_cover.sh train`（它打包成 motion_packed 目录） |
| motion_ends 只有 2 个数 | `--clips` 没带全（少了 dance 窗口） | 检查步骤 3 是否「7 段齐」 |
| 训练爆显存 / 明显变慢 | 4096 envs 需要 ≥4 GB 空闲 | 关掉其它占显存进程；**不要**改 envs（会破坏同口径） |
| eval 全部 skip | 之前跑过（summary.json 已存在） | 要重跑加 `LAB29_FORCE=1` |
| 表里 M4 / LAB27 全是「—」 | 基线 run 路径不对 | 用 `LAB29_L28_RUN=…` 与 `LAB29_L27_RUN=…` 指过去 |
| LAB29 的 w060 与 LAB28 的 LAB27（0.108）差很多 | 可能评估到了别的 ckpt | 查 summary.json 的 checkpoint 字段 |
| push 报 `Exec format error` | WSL interop 坏了（见坑 5） | 见坑 5 的修复命令 |

## 5. 五个坑

1. **RUN 写成 runs** → README/日志/metadata 全泼进 runs 根目录。脚本会拒绝，但你自己 `mkdir` 时别写错。
2. **pretrained/ 副本被当成 ckpt**：热启动权重会复制到 `logs/g1_tracking/pretrained/`，按字典序会排在日期目录后面。脚本已用 `grep -v "/pretrained/"` 排除，别自己用 `ls | tail -1`。
3. **拿完成度当唯一指标**：窗口越难、存活越短，完成度越低。跨窗口比较必须配存活帧/首段对齐误差（D42 决定 2、D44 决定 2）。
4. **train.log 入库**：几百 MB，绝对不要 git add（.gitignore 已忽略 runs/ 下的大文件，但别手贱）。
5. **WSL interop**：凭据助手是 Windows 程序，若 `/proc/sys/fs/binfmt_misc/` 里没有 `WSLInterop` 条目，push 会报 `Exec format error: … git-credential-manager.exe`。修复（需要 sudo 密码，你自己跑）：

```bash
sudo sh -c 'echo :WSLInterop:M::MZ::/init:PF > /proc/sys/fs/binfmt_misc/register'
```

## 6. 分支预案

| 结果 | 下一步 |
|---|---|
| (A) 六窗口一起上去（≥5 个 ≥+0.02） | 数据侧路线成立 → LAB30：多动作覆盖（fight1 S3 120 s / S2 134 s 也进训练）+ 数据规模曲线（3/5/7 段对比） |
| (B') 只有 0 s 好 | 覆盖不是主因 → 转方法侧：观测（相对根位姿 + 速度单因素消融）、参考修（P9）、训练目标 |
| (C) 部分窗口 | 逐窗口查 dmin / 关节速度 / 脚底接触（LAB22 工具可直接复用），先解释「最快窗为什么练了也没用」 |
| train 掉 >0.1（容量被摊薄） | 单因素加 iter（3000 → 6000）或减到 3 段版，别同时改别的 |

## 7. 结果（跑完填）

- 训练：墙钟 __，motion_ends __，ckpt sha __。
- 三策略表（9 窗口）见 `<run>/lab29_cover.md`。
- 判读：属于 (__)，dance1 六窗口 Δ = __。
- 下一步：__。
