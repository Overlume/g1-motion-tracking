# LAB23 讲义：DR 单档消融——三档里到底是谁造成了「退让式不倒地」

> 一次跑完，不探路。全量 3 臂约 **7.5 h**；时间紧可只跑两臂（fric + com）约 5 h，命令见 §2 步骤 3 的分支。
> 你在本课只需要：核对 → 写预测 → 一条命令跑完 → 读表 → 判读 → 改文档 → 提交。

## 0. 你要达成什么

**问题**：LAB20 把三档 startup DR 同时加强（摩擦 0.3–1.2 → 0.2–1.4、质心 ±2.5/±5/±5 → ±4/±6/±6 cm、编码器 ±0.01 → ±0.02 rad），结果是：
- held-out 完成度 **0.100 → 0.018**（负结果，D34）；
- −x 60 N·s 失败率 **30% → 0%**，但扰动后 root 误差 **18.7 → 92.9 cm**，A1 审计确认这 20/20 全是「退让型」（D35/D36）。

三档同改，所以无法回答：**哪一档**造成 held-out 崩塌？**哪一档**换来「不倒地」？本课把三档拆成三个单因素臂，各自只加强一项，其余两项回到 mjlab 默认。

**你要交付的东西**：一个能回答下面两个问题的表——
1. 每个单因素臂的「三数」：倒地数 k、非倒地末尾 root 误差、transient 退让型占比；
2. 每个单因素臂的 held-out 完成度（同口径 10 种子）。

### 0.1 为什么必须做这个（而不是直接调 DR 强度）

- 如果继续「三档同改 + 换强度」，你永远在二维网格上乱走：不知道收益来自哪一维，也不知道该往哪个方向加；
- LAB20 的结果是**收益与代价混在一起**（不倒地 ↑，跟踪 ↓，held-out ↓）。不做单档消融，下一步训练没有方向；
- 三数口径（D36）已经确定：只报 k 会把 93 cm 的退让写成「0 失败」。本课是第一次按这个口径做 A/B。

### 0.2 三个臂（其余两项回 mjlab 默认）

| 臂 | 只加强什么 | 取值（本臂） | 其余两项 |
| --- | --- | --- | --- |
| `fric` | 脚底摩擦 | 0.2 – 1.4 | 质心 ±0.025/0.05/0.05 m、编码器 ±0.01 rad |
| `com` | 质心偏移 | ±0.04 / ±0.06 / ±0.06 m | 摩擦 0.3–1.2、编码器 ±0.01 rad |
| `bias` | 编码器偏置 | ±0.02 rad | 摩擦 0.3–1.2、质心 ±0.025/0.05/0.05 m |

对照（不用重新跑）：
- **M4**（默认 DR）：nx_j60 5–6/20（三遍 6/6/5）、非倒地 root 中位 18.2 cm、退让型 0、held-out 0.100；
- **LAB20**（三档同改）：nx_j60 0/0/0、非倒地 root 中位 93.4 cm、退让型 60/60、held-out 0.018。

### 0.3 成本（预估，基于 LAB20 的实测）

| 阶段 | 单臂 | 三臂 |
| --- | --- | --- |
| 训练（3000 iter / 4096 envs，热启动 model_49999） | ≈1 h 35 min（LAB20 实测 1:35:01） | ≈4 h 45 min |
| 扰动 nx_j60（20 种子 × 3 遍 = 60 次） | ≈37 min（≈37 s/次） | ≈1 h 51 min |
| 扰动 ny_j60（20 种子 × 1 遍 = 20 次） | ≈12 min | ≈36 min |
| held-out 完成度（10 种子 × S3 120 s） | ≈7 min | ≈21 min |
| 合计 | ≈2 h 31 min | **≈7 h 30 min** |

两臂版（只跑 fric + com）≈ 5 h。磁盘：每臂训练日志约 0.9 GB，三臂约 3 GB（`train.log` 不进 git）。

### 0.4 三种结果分别怎么读（先想清楚，再看数）

1. **只有一臂的 held-out 崩**（≈0.02）→ 那一档就是「窗口记忆」的杀手；其余臂保住 0.09–0.10 且 −x 失败率下降 → 可以只保留那一档，得到「鲁棒性收益不付泛化代价」的组合。
2. **三臂的 held-out 都崩** → held-out 下降不是某一档造成的，而是「训练分布变宽」的普遍结果（D34 的解释要加强）；下一步只能改观测/数据，而不是调 DR。
3. **三臂都 0/20 且退让型都高** → 「不倒地」是任何一档 DR 都有的副作用（甚至只是「学会退让」），那么 LAB20b 的收益应改写为「退让式不倒地与具体 DR 档位无关」。

## 1. 协议（与 M4 / LAB20 完全一致，只改 DR 一档）

| 项 | 值 |
| --- | --- |
| 热启动权重 | `scratch/model_49999.pt`（sha f1bc404f…f22319） |
| 训练任务 | `Mjlab-Tracking-Flat-Unitree-G1` |
| 动作 | `data/processed/fight1_s3_000s_20s.npz`（sha 965f2c4b…b66c66） |
| 迭代 / 并行环境 | 3000 / 4096 |
| seed | 0（`--agent.seed`） |
| 日志 | tensorboard，save_interval 500 → 最后 ckpt `model_52998.pt` |
| 扰动 | nx_j60 / ny_j60（60 N·s = 300 N × 0.2 s），torso_link，8.0 s 起推 0.2 s |
| 扰动终止 | **关闭**（`--ignore-terminations`），推后看 3 s |
| 扰动种子 | 101–120；nx 3 遍、ny 1 遍 |
| held-out 评估 | `data/processed/fight1_s3_120s_20s.npz`，10 种子 101–110，终止检测开，999 步 |

## 2. 步骤（逐条照抄，把 `...` 换成你的值——本项目里尖括号是重定向符，不能原样粘）

### 步骤 0：开工核对（1 分钟）

    cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
    git status --short          # 必须无输出
    git log --oneline -1        # 期望 b8916da（A1 判定审计 + D36）或更新
    nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv
    sha256sum scratch/model_49999.pt
    ls data/processed/fight1_s3_120s_20s.npz

- sha 必须是 `f1bc404f…f22319`（前 8 / 后 6 位；完整值见 HANDOFF §6.2）；
- 显存占用应 < 500 MiB（有别的训练在跑就先停掉）；
- 期望文件都在，缺一个就别开始。

### 步骤 1：写预测（10 分钟，先写再跑）

打开 `docs/learning/LAB23-prediction.md`，按每题提示写答案。**不要跳**：本课的判分就看你能不能先验地区分「哪档造成什么」。

### 步骤 2：建 run 目录与 README 骨架（2 分钟）

RUN 必须是完整编号，不能是 `runs` 本身（脚本会拒绝）：

    RUN=runs/20261122-003-lab23-dr-ablation
    mkdir -p "$RUN"
    cat > "$RUN/README.md" <<'EOF'
    # 20261122-003-lab23-dr-ablation：DR 单档消融

    - 目的：三档 startup DR 拆成三个单因素臂，找出 held-out 崩塌与「退让式不倒地」的来源。
    - 臂：fric（摩擦 0.2–1.4）/ com（质心 ±4/6/6 cm）/ bias（编码器 ±0.02 rad），其余两项回默认。
    - 训练：热启动 model_49999、3000 iter、4096 envs、seed 0、S3 0–20 s。
    - 扰动：nx_j60 / ny_j60，20 种子 101–120，nx 3 遍 + ny 1 遍，终止关、8.0 s 起推 0.2 s、推后看 3 s。
    - held-out：S3 120 s，10 种子 101–110。
    - 结果：（跑完填三数表 + held-out + 判读）
    - 下一步：（填）
    EOF
    ls "$RUN"

自检：`ls "$RUN"` 只列出刚建的 README.md，不该出现别的东西。

### 步骤 3：一条命令跑完（≈7.5 h；两臂版 ≈5 h）

全量三臂（推荐挂在 WSL 终端里，窗口别关；也可以加 `nohup ... &` 让它在后台跑）：

    LAB23_RUN=runs/20261122-003-lab23-dr-ablation bash scripts/lab23_ablate.sh run 2>&1 | tee -a runs/20261122-003-lab23-dr-ablation/run.log

两臂版（时间紧）：

    LAB23_RUN=runs/20261122-003-lab23-dr-ablation LAB23_ARMS="fric com" bash scripts/lab23_ablate.sh run 2>&1 | tee -a runs/20261122-003-lab23-dr-ablation/run.log

脚本会依次做：训练每个臂 → nx_j60 三遍 → ny_j60 一遍 → held-out 十种子 → 判分 → 出表出图。
**可中断可续跑**：重跑同一条命令会跳过已完成的训练/评估（看 ckpt、recovery.json、summary.json 是否存在）。

跑的过程中可以另开一个窗口看进度：

    tail -f runs/20261122-003-lab23-dr-ablation/run.log
    nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv -l 30

### 步骤 4：读表（5 分钟）

    LAB23_RUN=runs/20261122-003-lab23-dr-ablation bash scripts/lab23_ablate.sh show
    cat runs/20261122-003-lab23-dr-ablation/lab23_ablation.md

**读数顺序**（照这个顺序看，不要先看 held-out）：

1. **先看 nx 池化 k/n**：每臂 60 次里倒几次。对照 M4 ≈ 6/20（LAB16 池化 11/40）与 LAB20 0/60。
2. **再看「非倒地 root 中位」**：如果 k 降了但这个数涨到 80–100 cm，说明是退让式不倒地（D36）。
3. **再看「退让型」计数**：transient 里偏离 ≥0.5 s 且 root ≥30 cm 的个数。
4. **再看 ny k/20**：按 D35，DR 在 −y 方向没有收益，这里主要看有没有明显变差。
5. **最后看 held-out 完成度**：与 0.100（M4）/ 0.018（LAB20）比。
6. p 值只作参考：脚本给的是逐遍 20 vs 20 的 Fisher；池化 p 不写进结论（重复测量）。

### 步骤 5：判读（10 分钟）

在 run README 里写清楚：

- 哪个臂「保住了 held-out」、哪个臂「崩了」；
- 哪个臂「k 降了但不是因为变强」（退让型多、root 误差大）；
- 三数表逐臂抄一遍；
- 你的结论对应 §0.4 的哪一种；
- 下一步建议（例如：只保留 fric 做大强度扫描、或先改观测 P7）。

### 步骤 6：同步八处文档（20 分钟）

1. `runs/20261122-003-lab23-dr-ablation/README.md`：填满（目的/结论/三数表/局限/下一步）；
2. `docs/REPORT.md`：在 `## 3. 关键发现` 前新增 `### 2.5.15 DR 单档消融（LAB23）`；§3/§4 各加一条（若结论成立）；
3. `docs/DECISIONS.md`：新增 `## D37 日期：…`，写依据/实测/决定（决定里必须写「以后报 DR 收益必须三数齐全」是否被本课验证/推翻）；
4. `docs/STATE.md`：头部「当前阶段」+ 末尾新增「LAB23 结果…当前事实以此为准」节；
5. `docs/PLAN.md`：把 LAB23 条目打勾并写实测；
6. `docs/ARTIFACT_INDEX.md`：新增「2026-11-22（LAB23 DR 单档消融）」节；
7. `README.md`：结果表加一行「DR 单档消融」；
8. `docs/learning/LAB23-dr-ablation.md` 的 §7 填实测；判分写进 `LAB23-prediction.md`（答案下方写参考回答）。

### 步骤 7：提交并 push（5 分钟）

    git add README.md docs scripts runs/20261122-003-lab23-dr-ablation outputs/figures/lab23_dr_ablation.png
    git commit -m "feat(lab23): DR 单档消融——三档归因（结果一句话）；D37"
    git -c credential.helper='!"/mnt/c/Program Files/Git/mingw64/bin/git-credential-manager.exe"' push origin main

注意：`runs/*/*/model_*.pt`、`train.log`、`rep*/` 下的大文件被 .gitignore 排除，只提交 README/md/json/png。

## 3. 「这课做完了」的判定标准（8 条，缺一条都不算）

1. `runs/20261122-003-lab23-dr-ablation/params.txt` 存在，里面有 arms、iters、envs、warm_sha256；
2. 三个臂各有一个最终 ckpt：`arms/<臂>/logs/g1_tracking/*/model_52998.pt`（或最后一个 model_*.pt），且**不在** `pretrained/` 下；
3. 每个臂都有 `perturb/rep1..3/recovery.json` 与 `perturb_ny/rep1/recovery.json`；
4. 每个臂都有 `eval/heldout/seed101..110/summary.json`（10 个）；
5. `lab23_ablation.md` 与 `lab23_ablation.json` 存在，表里 k / root / 退让型 / held-out 四列齐全；
6. `outputs/figures/lab23_dr_ablation.png` 存在，三面板能看出「k、root、held-out」的差异；
7. run README 的「结果」与「判读」写满，并明确写出属于 §0.4 的哪一种情形；
8. 你提交的 commit 里有 run README、docs 六处、讲义 §7、预测判分。

自检命令：

    LAB23_RUN=runs/20261122-003-lab23-dr-ablation bash scripts/lab23_ablate.sh show
    ls runs/20261122-003-lab23-dr-ablation/arms/*/perturb/rep1/recovery.json
    ls runs/20261122-003-lab23-dr-ablation/arms/*/eval/heldout/summary.json 2>/dev/null | wc -l
    git status --short

## 4. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 报错 `热启动权重不存在` | 没在仓库根目录跑，或 scratch/ 被清 | `ls scratch/model_49999.pt`；没有就按 HANDOFF §6.2 重新下载 |
| 报错 `LAB23_RUN 不能是 runs 本身` | RUN 写成 `runs` | 改成完整编号目录 |
| 训练刚开始就退出 | 显存被别的进程占满 | `nvidia-smi` 看占用；关掉旧训练 |
| 某个臂没有 ckpt | 训练中途 OOM/断电 | 直接重跑同一条命令，脚本会从「没 ckpt」的臂继续 |
| `lab23_ablation.md` 里某列是 `—` | 该臂某段评估还没跑完 | 重跑 `report` 或补齐评估后重跑 |
| held-out 全 0 | 用了 `heldout` 窗口以外的 motion 或 seed 错 | 看 `arms/<臂>/eval/heldout/seed*/summary.json` 里的 `motion_file` |
| 扰动评估很快结束（几秒） | 走了 `已有 summary.json 跳过` 分支 | 正常；要重跑就删对应目录 |

## 5. 五个坑

1. **`RUN=runs`**：脚本会在根目录写东西，和 60 多个历史 run 混在一起（2026-11-20 踩过）。脚本已加护栏，但你要确认目录名里有编号。
2. **把 pretrained 的副本当训练产物**：`logs/g1_tracking/pretrained/model_49999.pt` 是热启动权重，不是本臂训练结果（2026-11-21 踩过，D34-a）。脚本已加 `grep -v /pretrained/`，你核对时也要看路径。
3. **只看 k 就宣布「变强」**：LAB20b 的 −x 就是 0/60 但退让 93 cm。三数必须齐（D36）。
4. **把三遍当 60 个独立样本算 p**：重复测量会虚低。按 D35 写 20 vs 20 的逐遍 p。
5. **训练日志进 git**：`train.log` 约 0.9 GB/臂，确认 `git status` 里没有它。

## 6. 分支预案

- **某臂 held-out 崩、另两臂保住且 −x 失败率下降** → 这就是「不付泛化代价的鲁棒性组合」，写进 D37，并进入 LAB24（该档强度扫描 2–3 个点）。
- **三臂 held-out 全崩** → held-out 下降与具体 DR 档位无关；下一步不要再调 DR，直接转 P7（观测加世界系根位置/速度）。
- **三臂 k 都 0 且退让型都高** → 「不倒地」是退让的普遍副作用；D35 的表述改成「与档位无关」，并把 LAB20b 的「−x 有效」降级为「−x 更容易退让」。
- **某臂训练就直接失败（ckpt 缺失/显存 OOM）** → 记录在 run README 的「事故」节，补跑；不要改口径重来。

## 7. 结果（2026-11-23 实测，runs/20261122-003-lab23-dr-ablation）

三臂训练各 1 h 40–41 min（fric 08:53–10:34、com 10:34–12:15、bias 14:35–16:17），230 次评估 ≈2 h 15 min（单次 27–45 s），
从第一个训练目录到出表合计 8 h 08 min。协议核对：每臂 env.yaml 只有一档与默认不同；ckpt sha fric 603fe363…f9e945、
com 3e1ba2ed…a965ae3、bias e31c3160…b00111（均 ≠ 热启动 f1bc404f…f22319）。

| 臂 | nx k（逐遍） | 池化 | 非倒地 root 中位 (cm) | 漂移/退让 | ny k/20 | held-out |
| --- | --- | --- | --- | --- | --- | --- |
| M4 | 6/6/5 | 17/60 | 18.2 | 0/0 | 19/18/18 | 0.100 |
| LAB20 | 0/0/0 | 0/60 | 93.4 | 0/60 | 20/19/20 | 0.018 |
| fric | **2/2/2** | **6/60（10%）** | **81.2** | **54/0** | 20/20 | 0.018 |
| com | 13/15/15 | 43/60（72%） | 89.5 | 17/0 | 20/20 | 0.018 |
| bias | 19/19/19 | 57/60（95%） | 130.7 | 2/1 | 20/20 | 0.018 |

判读：

1. **只有摩擦档降失败率**（28% → 10%），但它是**漂移式**：54/54 非倒地 rollout 都被推走（root 中位 81.2 cm、骨盆 0.418 m
   未触地），「不摔倒」成立、「跟得住」不成立。
2. **质心/编码器档反向变差**（72%、95%，都显著差于 M4 的 28%）→ 「加强 DR 更鲁棒」不成立。
3. **held-out 崩塌与档位无关**：三臂都 0.018（±0.0005），与 LAB20 同值 → 0.100 是脆弱的近邻帧记忆，任何一档都能清零。
4. LAB20 的「0/60 + 60/60 退让」单档复现不出来 → 三档有**交互**。
5. 训练指标（anchor 0.17–0.21、root 奖励 0.18–0.22）不能预测抗推（10/72/95）。
6. −y 三臂全 20/20，无收益。

属于 §0.4 的哪一种：**都不是**（预案第 2 条命中一半）——「三臂 held-out 全崩」成立，但没有一臂同时做到「失败率降 + held-out 不降」；
摩擦档最接近（10% 且 held-out 与另两臂同值），所以下一步按 D37 做**摩擦档强度扫描**，目标 ≤10% 失败率 + 非倒地 root 中位 ≤40 cm。
