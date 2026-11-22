# LAB20b：加强 DR 的 ckpt 抗推能力（60 N·s 两向 × 20 种子 × 3 遍）

> 上一轮 LAB20 的结论是「加强 DR 没换来泛化，held-out 从 0.100 掉到 0.018」。但 DR 的本意
> 不是记住新动作，而是**在扰动下别摔**。这一课把 LAB12/LAB15/LAB16 那套扰动协议原样搬到
> LAB20 的权重上，只换 ckpt、不换别的，于是能和 M4 的数字直接对比。

## 0. 目标与已知

### 0.1 这一课要回答的问题

一句话：**训练期加强 DR，换来了受控外力下的稳定性吗？**

具体到数字：同样是 8.0 s 时对 torso_link 施加 60 N·s（300 N × 0.2 s）、20 个种子，
- −x 向：M4 是 5–6/20 倒地（LAB15 6/20；LAB16 三遍 6/6/5，翻转 4/20）；
- −y 向：M4 是 18–19/20（LAB15 19/20；LAB16 三遍 19/18/18，翻转 1/20）。
LAB20 的权重若更抗推，这两个数应该下降；若没变，就写「没有可分辨差异」。

### 0.2 已知数字（对照基线，全部来自已入库的 run）

| 项目 | 数值 | 来源 |
| --- | --- | --- |
| M4 ckpt | model_52998.pt，sha fc8f975b…8ceed9 | runs/20261117-001-m4-w15-finetune |
| LAB20 ckpt | model_52998.pt，sha e3747012…25b287 | runs/20261121-003-lab20-dr |
| −x 60 N·s（M4） | 6/20（LAB15）；三遍 6/6/5，翻 4/20（LAB16） | runs/20261120-001 / 002 |
| −y 60 N·s（M4） | 19/20（LAB15）；三遍 19/18/18，翻 1/20（LAB16） | 同上 |
| 5 种子基线（LAB12） | −x 1/5、−y 4/5 | runs/20261119-001-lab12-recovery |
| LAB20 vs M4 完成度 | train 0.999/0.999、heldout 0.100→0.018、unseen 0.025→0.024 | LAB20 / D34 |

### 0.3 为什么只测 60 N·s、只测 −x/−y

- 60 N·s 是**唯一两向都有分辨力**的档：−x 在 25–30%、−y 在 90–95%，一个降一个不降都看得见。
- 80 N·s 对 M4 已是 20/20（天花板），测它信息量低；要做也得先有 60 档的结论。
- ny_j45/j50 是过渡区（LAB17 已做），那是「找交点」的话题，不是「新 ckpt 有没有变强」的话题。
- 只换 ckpt 这一个因素，是本实验唯一能得出干净结论的原因。

### 0.4 成本

2 格 × 20 种子 × 3 遍 = **120 次评估**，单次约 35–45 s，合计 **70–90 min**。
脚本自己判分、算翻转率、画图，并生成与 M4 的对比表——你只需要跑一条命令、然后读三份产物。

## 1. 协议（必须与 LAB15/LAB16 一致，只有 ckpt 不同）

| 项 | 值 |
| --- | --- |
| checkpoint | runs/20261121-003-lab20-dr/logs/g1_tracking/2026-11-21_19-27-09_lab20-dr/model_52998.pt |
| sha256 | e3747012f3c7d35f9b1de4720a8420b6c1fb7f53c3c8e00c3ed2afcd2625b287 |
| 动作 | data/processed/fight1_s3_000s_20s.npz |
| 推力 | 8.0 s 起，持续 0.2 s，作用点 torso_link，60 N·s = 力 300 N |
| 方向/向量 | nx_j60 = (−300,0,0)；ny_j60 = (0,−300,0) |
| 终止 | 关（--ignore-terminations） |
| 种子 | 101–120（与 LAB15/LAB16 同一批） |
| 遍数 | 3（边界格 ≥3 遍，D28/D30） |
| 判据 | 骨盆比参考低 ≥0.08 m 或倾角差 ≥13°、连续 ≥10 帧 = 崩；末尾 0.5 s 平均下移 <0.05 m 且倾角差 <8° = 恢复 |

脚本已把上面全部写死，并会**校验 sha256**：不是这个 ckpt 就直接报错退出。

## 2. 七个步骤

### 步骤 0：开工前核对（2 分钟）

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
git status --short          # 期望：空
git log --oneline -1        # 期望：5445ed2 feat(lab20): ...
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
sha256sum runs/20261121-003-lab20-dr/logs/g1_tracking/2026-11-21_19-27-09_lab20-dr/model_52998.pt
```

期望 sha 以 **e3747012** 开头、**25b287** 结尾。**不要**用 pretrained/ 里的副本（那是 M4 权重，
sha 以 f1bc404f 开头）——挂错 ckpt 实验就白跑了。

### 步骤 1：先写预测（5 分钟）

打开 `docs/learning/LAB20b-prediction.md`，把 5 道题的答案写在每题下方。写预测是为了防止
「跑完看到数字再编解释」；判分时参考回答会写在你答案下面（AGENTS 规则）。

### 步骤 2：建 run 目录 + README 骨架（3 分钟）

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
RUN=runs/20261122-001-lab20b-perturb
mkdir -p "$RUN"
cat > "$RUN/README.md" <<'EOF'
# LAB20b：加强 DR 的 ckpt 抗推能力（60 N·s 两向 × 20 种子 × 3 遍）

- 目的：LAB20 的 held-out 归零（D34），检验加强 DR 是否换来受控外力下的稳定性。
- 假设：−x 60 N·s 的倒地数低于 M4 的 5–6/20；−y 已接近天花板，变化不会大。
- 配置：ckpt model_52998（sha e3747012…）｜动作 fight1_s3_000s_20s.npz｜8.0 s 起推 0.2 s｜60 N·s｜种子 101–120｜3 遍｜终止关。
- 对照：M4 的 LAB15/LAB16（−x 5–6/20 翻 4/20；−y 18–19/20 翻 1/20）。
- 结果：（跑完填：逐遍 k、翻转、Fisher p、配对改进/变差种子、结论）
- 下一步：（填）
EOF
ls -d "$RUN"
```

`RUN` 必须是**新建的一个子目录**，不能是 `runs` 本身（2026-11-20 踩过：产物摊在 runs
根目录、和 50 多个历史 run 混在一起）。自检：

```bash
ls runs/*.md runs/*.json 2>/dev/null    # 期望：没有任何输出
```

`20261122-001` 这个编号若已被占用，顺延到 `-002`。

### 步骤 3：跑三遍（70–90 分钟）

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
RUN=runs/20261122-001-lab20b-perturb
LAB20B_RUN_DIR="$RUN" bash scripts/lab20b_perturb.sh run 2>&1 | tee "$RUN/run.log"
```

这条命令做四件事：① 校验 ckpt sha；② 三遍各调一次 `scripts/lab12_recovery.sh run`（= LAB12
的同一套评估）；③ 逐遍判分；④ 合并翻转率（`fliprate.md` / `fliprate.json` + 图
`outputs/figures/lab20b_robust.png`）并生成与 M4 的对比 `vs_m4.md`。

跑的过程中你会看到 `== 第 N/3 遍` 与 `[i/40] nx_j60 seed 101 force=-300,0,0`。
**不要关窗口**（关掉会杀掉评估进程）。要走开就开 tmux：

```bash
tmux new -s lab20b
# 里面跑上面那条命令；Ctrl-b d 跳出；回来 tmux attach -t lab20b
```

### 步骤 4：看产物是否齐（2 分钟）

```bash
RUN=runs/20261122-001-lab20b-perturb
cat "$RUN/params.txt"
ls "$RUN"/rep*/recovery.json "$RUN"/fliprate.* "$RUN"/vs_m4.md outputs/figures/lab20b_robust.png
```

期望：`params.txt` 里 sha256 是 e3747012…25b287、cells 是 `nx_j60 ny_j60`、seeds 是 101–120、
reps 是 3；三个 `recovery.json` 都在。

### 步骤 5：按这个顺序读数（10 分钟）

1. `params.txt` 的 sha 与 seeds——先确认「实验是它自己」。
2. `fliprate.md` 的「每格 x 每遍」表——三遍 k 分别是多少。
3. 同文件的「翻转统计」——几个种子的判定会翻（≥3/20 必须写区间）。
4. `vs_m4.md` 的「每格倒地数」——和 M4 差多少、Fisher p 多大。
5. `vs_m4.md` 的「逐种子配对变化」——谁从不倒变倒、谁反过来（最硬的证据：同种子、只换权重）。
6. 图 `outputs/figures/lab20b_robust.png`——竖条纹就是翻转的种子。

### 步骤 6：同步八处文档（20 分钟）

按 AGENTS.md「交接与文档同步」，一个都不能漏：

1. `runs/20261122-001-lab20b-perturb/README.md`：把「结果/下一步」填全（逐遍 k、翻转、p、配对）。
2. `docs/REPORT.md`：在「## 3. 关键发现」之前插入 `### 2.5.13` 小节。
3. `docs/DECISIONS.md`：末尾加 `## D35`（决定：DR 加强在抗推上有没有收益、以后 DR 怎么选）。
4. `docs/STATE.md`：头部「最后更新」与「当前阶段」、末尾新节「LAB20b 结果…当前事实以此为准」。
5. `docs/PLAN.md`：找到 LAB20b 那一条，改成 `[x]` 并写一句实测。
6. `docs/ARTIFACT_INDEX.md`：新节 `## 2026-11-22（LAB20b 抗推对比）`。
7. `README.md`：结果表在「训练期 DR」行下面加一行「DR 抗推对比」。
8. `docs/learning/LAB20b-robust.md` 的 §7：填实测（表 + 结论句 + 与 M4 的对照）。

### 步骤 7：提交并推送（5 分钟）

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
git add -A
git status --short
git commit -m "feat(lab20b): DR 加强 ckpt 的 60 N·s 抗推对比（对照 M4）；D35"
GIT_TERMINAL_PROMPT=0 git -c 'credential.helper=!"/mnt/c/Program Files/Git/mingw64/bin/git-credential-manager.exe"' push origin main
```

`docs/handoffs/HANDOFF.md` 只本地、不入库，别 `git add` 它。

## 3. 「做完了」的判定（8 条，全满足才算过）

1. `params.txt` 的 sha256 = e3747012…25b287，seeds 是 101–120，reps=3。
2. 三个 `rep<N>/recovery.json` 都存在，每个含 2 格 × 20 种子 = 40 条。
3. `fliprate.md` 里两格的「逐遍 k」都写出来了（6 个数）。
4. `fliprate.md` 里两格的翻转数与 CI 都写出来了。
5. `vs_m4.md` 里有每格 Fisher p 和「配对改进/变差种子」清单。
6. 图 `outputs/figures/lab20b_robust.png` 存在，两个面板（nx_j60 / ny_j60）。
7. 八处文档都改了，且没有与脚本实际输出矛盾的句子（例如 k 写错、把 p 当「必然显著」）。
8. 已 commit + push，`git status` 干净。

## 4. 常见问题

| 现象 | 原因 | 怎么办 |
| --- | --- | --- |
| 脚本立刻报 ckpt sha 不符 | 用了 pretrained/ 下的副本，或路径写错 | 用 §1 的完整路径，先 sha256sum 核对 |
| 报 xxx 下没有 repN 目录 | 上一步没跑完/被打断 | 用 `bash scripts/lab20b_perturb.sh report` 补统计；缺哪遍单独补跑 |
| 报「会污染 runs 根目录」 | RUN 写成了 runs | 建 `runs/20261122-001-lab20b-perturb` 再用它的子目录 |
| 关掉终端后评估没了 | 前台进程随终端死掉 | 重跑缺的遍：`LAB20B_REPS=1 LAB20B_RUN_DIR="$RUN" LAB20B_CELLS="ny_j60" LAB20B_SEEDS="101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120" bash scripts/lab20b_perturb.sh run` |
| 图里中文变方块 | matplotlib 无中文字形 | 脚本已用 ASCII 图标题，别自己改中文 |
| fliprate.md 里 k 全是 0 | ckpt 挂错，或没加 --ignore-terminations | 只信脚本；核对 params.txt 的 sha |

## 5. 六个坑

1. **不要把 LAB15 的目录当 rep1**：LAB16 那套是「rep1 引用旧结果」，LAB20b 三遍都必须用新 ckpt 新跑
   （不同 ckpt 混着比 = 无效实验；fliprate.md 里也有 ckpt 一致性警告）。
2. **60 N·s 不是 60 N**：力 = 冲量 / 0.2 s = 300 N。命令里写错数量级，结论全错。
3. **别改判据**：阈值（0.08 m / 13° / 10 帧 / 末尾 0.5 s）是 LAB12 定的，改了就不能和 M4 比。
4. **三遍不是凑数**：它是量「同配置下判定会翻多少」，用来决定 k 要不要写区间（D28）。
5. **Fisher p 大不等于「没差别」**：20 个种子检验功效有限，p 大只能说「看不出来」。
6. **别边跑边改脚本**：三遍口径会不一致。

## 6. 分支预案（跑完先别急着写结论）

| 结果 | 解读 | 下一步 |
| --- | --- | --- |
| −x k ≤ 2/20 且 p<0.05 | 加强 DR 在 −x 上确有抗推收益 | D35 写「收益在鲁棒性而非泛化」；LAB20 负结论限定到「泛化指标」；考虑 LAB21 |
| −x k 落在 5–6 | 没有可分辨差异 | 写「20 种子下看不出差异」，附 CI；不要写「略有改善」 |
| −x k ≥ 9 或 −y 明显变差 | 加强 DR 反而更脆 | 记录「以鲁棒性换精度」的反例，先查 ckpt 与种子是否错位 |
| 两格翻转 ≥3/20 | 判定层不稳 | 按 D28 写区间，k 只给中位与极差 |
| 两向都不变 | DR 加强既不换泛化也不换抗推 | 结论：本强度 DR 无收益；未来 DR 先做单档消融（D34） |

## 7. 结果（跑完填）

（待填）
