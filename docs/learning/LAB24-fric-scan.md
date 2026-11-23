# LAB24 讲义：摩擦档强度扫描（为什么放宽摩擦上界，以及怎么暂停续训）

> 目标读者：你（项目作者）。本讲义自足：只凭这一份就能做完实验并判定通过。
> 前置：LAB23（D37）已完成——三档单因素臂里只有摩擦档降 -x 失败率，但代价是「漂移式不倒地」。

## 0. 先看清楚要回答什么

### 0.1 已知（都不需要重测）

| 已知 | 数值 | 来源 |
| --- | --- | --- |
| M4（默认 DR：摩擦 0.3–1.2） | nx_j60 逐遍 6/6/5（28%），非倒地 root 中位 18.2 cm，无漂移 | LAB15/16 |
| LAB20（三档同改，摩擦也是 0.2–1.4） | 0/0/0，非倒地 root 中位 93.4 cm，60/60 退让 | LAB20b |
| LAB23 fric 臂（只有摩擦 0.2–1.4 加强） | 2/2/2（10%），非倒地 root 中位 81.2 cm，54/54 **漂移型**（骨盆 0.418 m 未触地） | LAB23/D37 |
| LAB23 com / bias 臂 | 13–15/20（72%）、19/19/19（95%），都更差 | LAB23 |

问题只剩一个：**摩擦区间继续放宽（上界 1.4 → 2.0 → 3.0）能不能把「被推走 81 cm」压回去，同时保住 ≤10% 失败率？**

### 0.2 三个臂、要跑多少次、多久

| 臂 | 摩擦区间 | 训练 | 评估（nx_j60） |
| --- | --- | --- | --- |
| f14 | 0.2–1.4 | **不训练**（复用 LAB23 fric 的 ckpt：sha 603fe363…f9e945） | 20 种子 × 3 遍 = 60 次 |
| f20 | 0.2–2.0 | 3000 iter / 4096 envs / seed 0（≈1 h 40 min） | 60 次 |
| f30 | 0.2–3.0 | 同上（≈1 h 40 min） | 60 次 |

其余两项 DR 回 mjlab 默认（质心 ±2.5/5/5 cm、编码器 ±0.01 rad）——只动摩擦一档。
单次评估 ≈35 s，60 次 ≈35 min。**总计 ≈5 h 05 min**（训练 3 h 20 + 评估 1 h 45）。

### 0.3 三种结果怎么读（先想清楚，再看数）

| 结果 | 含义 | 下一步 |
| --- | --- | --- |
| f20/f30 的 k ≤10% **且** 非倒地 root 中位 ≤40 cm | 摩擦上界是真杠杆：既站得住又跟得上 | 写 D38，进入 LAB25（该点 ±1 档确认 + ny_j60 复测） |
| k ≤10% 但 root 仍在 60–80 cm | 只是把「摔倒」换成「漂移」，不是能力提升 | 结论写「摩擦不是杠杆」，转下界方向（0.1 / 0.05）或转 P7 |
| k 反而 >10%（摩擦越大越倒） | 策略学到「靠高摩擦硬顶」，在评估摩擦下更脆 | 按 D37 决定 1 逐档报三数，别写「略有好转」 |

### 0.4 什么时候可以停

- 三臂各 3 遍齐 + 图生成 + 文档同步 = 可停。
- 若 f20 就已满足「≤10% 且 ≤40 cm」，f30 仍要跑（证明不是单点噪声；D28 的「边界格 ≥3 遍」同理）。

## 1. 协议（写死，不许改）

| 项 | 值 |
| --- | --- |
| 热启动权重 | scratch/model_49999.pt（sha f1bc404f…f22319） |
| 动作 | data/processed/fight1_s3_000s_20s.npz |
| 训练 | 4096 envs、3000 iter、seed 0、num-steps-per-env 24，日志 tensorboard |
| 目标 iter | 52998（从 49999 续 3000 iter，与 M4/LAB20/LAB23 同口径） |
| 扰动 | nx_j60：力 (-300,0,0) N × 0.2 s = 60 N·s，第 8.0 s 起推，推后看 3 s，终止关 |
| 种子 | 101–120，每臂 3 遍 |
| 判据 | 倒地 = 骨盆比参考低 ≥0.08 m 或倾角差 ≥13°、连续 ≥10 帧；恢复 = 末尾 0.5 s 平均下移 <0.05 m 且倾角 <8° |
| 三型 | 擦边（偏离 ≤0.3 s 且 root <30 cm）/ 漂移（偏离 <0.5 s 但 root ≥30 cm）/ 退让（偏离 ≥0.5 s 且 root ≥30 cm） |

## 2. 七步

### 步骤 0：开工核对（1 min）

```bash
cd /home/overlume/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
git log --oneline -1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
sha256sum scratch/model_49999.pt
ls runs/20261122-003-lab23-dr-ablation/arms/fric/logs/g1_tracking/
```

预期：HEAD 是 `353e29d`（或更新）；显存已用 < 1 GB；sha 以 f1bc404f 开头、以 f22319 结尾；
LAB23 fric 的 ckpt 目录里有 model_52998.pt（f14 臂要复用它）。

### 步骤 1：写预测（5 min，先写再看数）

打开 `docs/learning/LAB24-prediction.md`，把 5 道题答完再继续。别在后面的步骤里回来改。

### 步骤 2：建 run 目录 + README 骨架（3 min）

```bash
RUN=runs/20261123-001-lab24-fric-scan
mkdir -p "$RUN"
cat > "$RUN/README.md" <<'EOF'
# LAB24 摩擦档强度扫描（nx_j60，20 种子 × 3 遍）

- 目的：LAB23 的 fric 臂（0.2–1.4）是 10% 失败率但漂移 81 cm；继续放宽摩擦上界（2.0 / 3.0）能否把漂移压回去。
- 假设：上界越高，策略越敢用接触力，被推后更容易原地找回；但 k 可能回升。
- 配置：热启动 model_49999（sha f1bc404f…f22319）、动作 fight1_s3_000s_20s.npz、终止关、8.0 s 起推 0.2 s、种子 101–120 × 3 遍。
- 臂：f14=0.2–1.4（复用 LAB23 fric ckpt）/ f20=0.2–2.0 / f30=0.2–3.0；com/bias 回默认。
- 命令：见 docs/learning/LAB24-fric-scan.md §2。可随时 Ctrl-C 暂停，重跑同一条命令自动续训。
- 结果：（跑完填：三臂三数表、判据是否满足、属于 §0.3 哪一种）
- 下一步：（填）
EOF
ls "$RUN"
```

预期：只列出 README.md，没有任何别的文件。

自检：`ls runs/*.md runs/*.json` 必须无输出（否则 RUN 写错了）。

### 步骤 3：训练（可暂停，≈3 h 20 min）

```bash
cd /home/overlume/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
LAB24_RUN=runs/20261123-001-lab24-fric-scan bash scripts/lab24_fric_scan.sh train
```

这条命令会：跳过 f14（复用 ckpt）→ 训练 f20（≈1 h 40）→ 训练 f30（≈1 h 40）。
每 500 iter 存一次 ckpt（model_50000 / 50500 / … / 52998）。

**暂停**：直接 Ctrl-C（最多丢 500 iter ≈16 min）。
**续训**：把同一条命令原样再执行一次，它会从该臂最新 ckpt 接着跑到 52998（只跑剩余差值）：

```bash
LAB24_RUN=runs/20261123-001-lab24-fric-scan bash scripts/lab24_fric_scan.sh train
```

**看进度**（随时，零 GPU）：

```bash
LAB24_RUN=runs/20261123-001-lab24-fric-scan bash scripts/lab24_fric_scan.sh status
```

预期输出形如：

```
arm    ckpt_iter    reps_done      ckpt
f14    52998        0/3            runs/20261122-003-lab23-dr-ablation/arms/fric/logs/.../model_52998.pt
f20    51500        0/3            runs/20261123-001-lab24-fric-scan/arms/f20/logs/.../model_51500.pt
f30    —            0/3            （未训练）
```

训练中途自检：

```bash
grep -n "foot_friction" -A3 runs/20261123-001-lab24-fric-scan/arms/f20/logs/g1_tracking/*/params/env.yaml
```

应看到 ranges 为 [0.2, 2.0]；同时核对 base_com 是 ±0.025/0.05/0.05、encoder_bias 是 ±0.01。

### 步骤 4：评估（可暂停，≈1 h 45 min）

```bash
LAB24_RUN=runs/20261123-001-lab24-fric-scan bash scripts/lab24_fric_scan.sh eval
```

流程：f14 三遍 → f20 三遍 → f30 三遍，每遍 20 种子（101–120），单次 ≈35 s。
**暂停/续跑**：Ctrl-C 后重跑同一条命令；已写完 summary.json 的种子、已有 recovery.json 的遍都会跳过，不会重复烧 GPU。

想先看 f14 自检结果（必须复现 LAB23 的 6/60、2/2/2）：

```bash
LAB24_RUN=runs/20261123-001-lab24-fric-scan LAB24_ARMS=f14 bash scripts/lab24_fric_scan.sh eval
```

### 步骤 5：出表 + 图 + 读数顺序（2 min）

```bash
LAB24_RUN=runs/20261123-001-lab24-fric-scan bash scripts/lab24_fric_scan.sh report
```

产物：`runs/20261123-001-lab24-fric-scan/lab24_scan.md|json`、`outputs/figures/lab24_fric_scan.png`
（三面板：倒地率 / 非倒地 root 误差 / 二者散点）。要改 md 标题或图标题，直接调 Python：

```bash
environment/.venv/bin/python scripts/lab24_report.py --run runs/20261123-001-lab24-fric-scan --figure   --title "LAB24 摩擦档强度扫描" --figure-title "LAB24: friction upper bound scan (nx_j60)"
```

**读数顺序（重要）**：

1. 先看自检行：f14 池化必须是 **6/60、逐遍 2/2/2**。不对 → 停，查 ckpt 与种子（见 §4）。
2. 再看三臂 k（逐遍，不看池化 p）：是否有一臂 ≤10%？
3. 再看同一臂的非倒地 root 中位：是否 ≤40 cm？**这才是「漂移压回去了」**。
4. 最后看三型计数（擦边/漂移/退让）：k 降但漂移没降 = 只是换了失败方式。
5. p 值只写逐遍 20 vs 20 的 Fisher（三位有效数字），不写池化 p。

### 步骤 6：文档同步（15 min，八处）

把实测数据填进本文件 §7，然后：

1. `runs/20261123-001-lab24-fric-scan/README.md` 填全（结果/判读/局限/文件清单）。
2. `docs/REPORT.md`：新增 §2.5.16（表 + 判读 + 图）插在 `## 3. 关键发现` 之前；§3 加一条；§4 加局限。
3. `docs/DECISIONS.md`：新增 D38（摩擦上界是否杠杆、判据、下一步）；D37 末尾加补注。
4. `docs/STATE.md`：头部「最后更新」与「当前阶段」；末尾新增「## LAB24 结果…当前事实以此为准」。
5. `docs/PLAN.md`：把 LAB24 行标 [x] 并写实测。
6. `docs/ARTIFACT_INDEX.md`：新增「## 2026-11-23（LAB24 摩擦档扫描）」节。
7. `README.md` 结果表加一行「摩擦档强度扫描」。
8. `docs/handoffs/HANDOFF.md`（仅本地）：最新提交、最近进展、下一步。

### 步骤 7：提交并 push（5 min）

```bash
git add README.md docs runs/20261123-001-lab24-fric-scan outputs/figures/lab24_fric_scan.png
git status --short     # 确认没有 train.log（0.9 GB/臂）被加进来
git commit -m "feat(lab24): 摩擦档强度扫描（0.2–2.0 / 0.2–3.0）——漂移是否随上界下降；D38"
```

push 用凭证脚本（别直接在 wsl -lc 的双引号里写 credential.helper）：

```bash
cat > /tmp/push_lab24.sh <<'EOF'
#!/usr/bin/env bash
set -e
cd /home/overlume/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
GIT_TERMINAL_PROMPT=0 git -c 'credential.helper=!"/mnt/c/Program Files/Git/mingw64/bin/git-credential-manager.exe"' push origin main
EOF
bash /tmp/push_lab24.sh
```

## 3. 通过判定（全满足才算做完）

1. f14 自检通过：池化 **6/60、2/2/2**（与 LAB23 一致）。
2. f20 / f30 都训练到 **iter 52998**（status 里 ckpt_iter = 52998）。
3. 每臂 **3 遍** recovery.json 都在，种子 101–120 齐全。
4. env.yaml 核对：f20 = [0.2, 2.0]、f30 = [0.2, 3.0]；com/bias 为默认值。
5. 表里三数齐全：k、非倒地 root 中位、三型计数。
6. 图 `outputs/figures/lab24_fric_scan.png` 生成、三面板正常。
7. 结论按 §0.3 三选一写死，不许出现「略有改善」这种模糊句。
8. 八处文档同步 + 提交 push 完成，工作树干净。

## 4. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 训练一秒就退出并说「已完成训练」 | 该臂 ckpt 已是 52998 | 正常，直接去评估 |
| Ctrl-C 后 status 里 ckpt_iter 是 51500 之类 | 上次被打断在中间 | 重跑同一条 train 命令，会自动续到 52998 |
| 续训报「热启动权重不存在」 | 指向的 ckpt 被删/挪走 | 找回 logs 目录；实在没有就删掉该臂目录重训 |
| 续训后 ckpt 名字不是 52998 | 手动改过 LAB24_TARGET_ITER | 别改，默认 52998 才对 |
| 提示 `LAB24_RUN 不能是 runs 本身` | RUN 写成 runs | 用完整编号目录 |
| 显存不足 / OOM | 有别的进程占卡 | nvidia-smi 看清楚，关掉旧进程再续训 |
| 某遍 recovery.json 在但种子不全 | 上次判分被中断 | 删掉该遍 recovery.json 再跑 eval（种子按 summary.json 跳过） |
| 图标题是方框 | matplotlib 默认字体没有中文 | 图标题用 ASCII；md 标题用 --title |

## 5. 五个坑

1. **RUN 写成 runs**：会把 README/表/图直接摊在 runs 根目录（2026-11-20 踩过）。脚本有护栏，你也要看目录名里有没有编号。
2. **把 pretrained 里的副本当训练产物**：热启动权重会被复制到 `logs/g1_tracking/pretrained/`，它不是本臂训练结果（2026-11-21 踩过，D34-a）。脚本已 grep 掉，但你核对 sha 时也要看路径。
3. **把三遍当 60 个独立样本**：重复测量会虚低 p。按 D35 报逐遍 20 vs 20。
4. **只看 k 就宣布「变强」**：LAB23 的 fric 是 10% 但漂移 81 cm。判据必须同时满足 k ≤10% 与 root ≤40 cm。
5. **train.log 进 git**：约 0.9 GB/臂。commit 前一定看 `git status --short`。

## 6. 分支预案

- **f20 或 f30 同时满足 k ≤10% 与 root ≤40 cm** → 「能站住又跟得住」的组合：写进 D38，LAB25 用该点 ±1 档确认 + ny_j60 复测 20 种子。
- **k 都 ≤10% 但 root 仍在 60–80 cm** → 摩擦上界不是杠杆，换轴：固定上界 1.4、把**下界**往 0.1 / 0.05 收（更滑的地面），或直接转 P7（观测加世界系根位置/速度）。
- **f30 训练崩（motion_end 暴跌、不收敛）** → 记事故，降为 0.2–2.5 再试一次；不要改评估口径。
- **f14 自检对不上 6/60** → 先查复用的是不是 LAB23 fric 的 ckpt（sha 603fe363…f9e945）与种子集，别急着怀疑判分。

## 7. 结果（跑完填）

- 待填：三臂三数表、自检是否通过、属于 §0.3 哪一种、结论一句、下一步。
