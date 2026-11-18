# LAB08 — M4 规模训练：300 iter → 3000 iter（weight 1.5 固定）

> 目标读者：你（机械背景、Python 入门）。预计 2 小时（训练 1.7 h，可以挂着去干别的）。
> 前置：LAB07 已做完，D16 已定 **M4 用 weight = 1.5、std = 0.3**。
> 本 LAB 只回答一个问题：**参数选好之后，多训练还能不能继续降低根位置误差？**

---

## 1. 要回答什么

LAB07 的结论是「weight 越大位置越准，但边际递减、姿态付钱」，于是选了 1.5。
但四臂都只训练了 **300 iter**（相对官方 50000 iter 的 **0.6%**），这引出一个必须回答的问题：

> 300 iter 就已经把 S3 从 5/5 摔倒变成 0/5 不摔，根误差 25.7 cm。
> 那到底是「权重设对了、只是训练不够」，还是「权重设对了、也已经到顶」？

这两件事对未来所有实验的意义完全不同：

- 如果是前者 → M5 扰动实验必须在更充分的策略上做，否则测到的是「训练不足」而不是「抗扰能力」。
- 如果是后者 → 20~25 cm 就是这个配置族的地板，要继续降就得改观测/奖励结构，不是加预算。

## 2. 为什么是 3000 iter

| 预算 | env-steps | 占官方 50000 iter 的比例 | 4096 env 墙钟 |
|---|---|---|---|
| 300 iter（LAB06/LAB07） | 29.5 M | 0.6% | ≈ 11 min |
| **3000 iter（本 LAB）** | **295 M** | **6%** | **≈ 1.7 h** |
| 50000 iter（官方） | 4.9 G | 100% | ≈ 28 h |

选 3000 而不是 1000 或 10000：1000 只比 300 多 3 倍，落在 LAB07 已经看到的「边际递减」区间里，区分度不够；
10000 要 5.7 h，一旦卡住代价太大。3000 是「足够看出趋势、又不至于让你一晚白等」的点。
**这是规模轴，不是调参轴**——训练过程中不许改任何配置。

## 3. 三个假设与可区分预测

| 假设 | 机制 | 3000 iter 后应该看到 |
|---|---|---|
| H1 预算不足 | 300 iter 时策略还没把根位置误差压到位，梯度方向是对的 | root 误差明显下降（≥ 20%，即 ≤ 20 cm），R-MPKPE 不掉 |
| H2 配置到顶 | 25.7 cm 是这套观测/奖励能给出的地板，再多训练只是在噪声里晃 | root 误差与 300 iter **重叠**（差异 < 5%），R-MPKPE 持平 |
| H3 过拟合/退化 | 训练更久 → 策略更用力地贴参考轨迹，姿态与接触变差 | root 误差降但 R-MPKPE / joint_vel_rmse 明显变差（> 10%） |

三个假设在 root 误差上 H1 与 H2 相反；H3 与 H1 同向但姿态指标相反。所以**必须同时看 root 误差和 R-MPKPE**，只看一个会得出错误结论。

## 4. 先预测（开跑前写进 docs/learning/LAB08-prediction.md）

1. 3000 iter 后 root xy 误差中位数会落在多少 cm？给出区间和依据。
2. R-MPKPE 会变好、持平还是变差？为什么（提示：它是 root 相对量，根位置更准不自动等于姿态更准）。
3. 训练侧 error_anchor_pos 会从 0.2084 m 降到大约多少？它和评估侧 root 误差不是同一个量，差在哪？
4. 如果跑完发现 root 误差 24 cm、R-MPKPE 6.1，你的下一个实验是什么？（考决策链，没有标准答案）

## 5. 操作步骤（约 1.7 h + 10 min）

### 5.1 开跑前检查机器（D14）

```bash
free -h                      # available 应 > 3 GiB；不够就先关浏览器
nvidia-smi --query-gpu=temperature.gpu,memory.used --format=csv
```

### 5.2 把 config 改成 M4 配置（只改一行）

```bash
sed -n '210,214p' repos/mjlab_research/src/mjlab/tasks/tracking/tracking_env_cfg.py
sed -i '212s/weight=0.5,/weight=1.5,/' repos/mjlab_research/src/mjlab/tasks/tracking/tracking_env_cfg.py
sed -n '212p' repos/mjlab_research/src/mjlab/tasks/tracking/tracking_env_cfg.py   # 应显示 weight=1.5，std 仍 0.3
```

脚本会自己校验：相对 LAB06-A 快照必须**恰好 +1/-1 行**，且 210-214 行是 weight=1.5 与 "std": 0.3。改错会直接拒绝开跑。

### 5.3 训练（1.7 h）

```bash
nohup bash scripts/m4_scale_train.sh train > /tmp/m4_train.out 2>&1 &
tail -f /tmp/m4_train.out
```

产物落在 runs/YYYYMMDD-NNN-m4-w15-finetune/：command.txt、params.txt、cfg_snapshot.py、diff_A.txt、train.log、gpu_samples.csv、mem_before/after.txt、README.md。
墙钟用 /proc/uptime 记（D14：WSL2 的 time.time() 会跳，日志里的 Iteration time 会出现负数，别慌）。

**训练期间不要做的三件事**：别再开评估（抢显存）、别并发跑别的 bash 重任务（会让迭代时间失真）、别中途改 config（配置快照对不上）。

### 5.4 评估（5 个种子，约 5–10 min）

```bash
bash scripts/m4_scale_train.sh eval            # 默认种子 101 102 103 104 105
```

评估的是最终 ckpt model_52998.pt（49999 + 3000 - 1），不是中途的 save。落在 runs/YYYYMMDD-NNN-m4-eval/final/seed101…105/。

### 5.5 出对比表

```bash
bash scripts/m4_scale_train.sh report
```

生成 comparison.md、comparison.json、sweep.csv、sweep.png，两臂分别标 300（复用 LAB06 的 B 组）与 3000（本次）。

## 6. 读结果的三条硬规则

1. **先看分布再看中位**：5 个种子的 min–max 如果重叠，就是「不可分辨」，别拿 1 cm 的差讲故事。
2. **噪声底是 4%**：LAB05-A 43.34 vs LAB06-A 41.52（同配置同种子重跑）差 4.2%。小于 4% 的变化一律记为持平。
3. **摔与准同时报**：只报根误差降了、不报摔了几次和姿态代价，等于没报。M3-b 的两个反直觉教训就是这么来的。

## 7. 卡住对照

| 现象 | 可能原因 | 怎么办 |
|---|---|---|
| 报「相对 A 态恰好 +1/-1 行」失败 | 多改/漏改行 | 先 sed -n '210,214p'，或 git diff 看改了哪 |
| 迭代时间出现负数 | D14：WSL2 实时钟回跳 | 正常，忽略；真实墙钟看 README 里的 /proc/uptime 值 |
| 训练很慢（> 3 s/iter） | 内存/温度（LAB05 时 available 1.8 GiB、86 °C） | free -h 与 nvidia-smi 看；温度 > 85 °C 先降级再跑 |
| eval 某种子失败 | 显存或终止统计异常 | 看 seedNNN.log；其余种子继续，报告里缺的种子显示 n/a |
| report 找不到 300 iter 参照 | LAB06 的 eval 目录被移走 | 用 LAB06_RUNS_DIR= 指定目录 |

## 8. 通过标准

- [ ] runs/*-m4-w15-finetune/ 存在，diff_A.txt 恰好 1 行 + 改动，params.txt 里 iters=3000、num_envs=4096。
- [ ] train.log 最后一行迭代号是 52998。
- [ ] logs/g1_tracking/ 下有 model_50000/50500/…/52998.pt。
- [ ] eval 五个种子目录齐全，comparison.md 里两臂都有 5 个种子。
- [ ] 你能用一句话说明「这次实验证明了 H1/H2/H3 里的哪一个，证据是哪一行数字」。

## 9. 收尾（无论结论如何）

```bash
sed -i '212s/weight=1.5,/weight=0.5,/' repos/mjlab_research/src/mjlab/tasks/tracking/tracking_env_cfg.py
sha256sum repos/mjlab_research/src/mjlab/tasks/tracking/tracking_env_cfg.py
# 期望 f756b4d0e76a752a928cf41a6317f1147f8316319d8774c4c3769eb1df8b9a82
```

然后把 free -h、nvidia-smi、comparison.md、你的预测文档发我，我做判读并写 runs README 与 D17。

## 10. 自测记录（智能体）

2026-11-16 自测（`scratch/m4-selftest`，gitignored）：

- 训练：64 env × 40 iter（49999 -> 50038），save_interval 10，seed 0；退出码 0，真实墙钟 62 s，稳态 0.57-0.64 s/iter；ckpt 50000/50010/50020/50030/50038 齐全。
- 校验生效：config 为 1.5 时通过；改回 0.5 或改 std 会被 die 拒绝（LAB07 阶段已验）。
- eval：1 种子（101）能跑通并写出 `runs/*-m4-eval/final/seed101/summary.json`。
- report：两臂并表正常（300 = LAB06 的 B 组 5 种子，3000 = 自测 1 种子），`sweep.png` 正常出图；已把表头与横轴改为可覆盖，M4 传 `--xlabel "training iterations"`（DejaVu Sans 无中文字形，标题用英文）。
- 自测数字本身无意义（40 iter 策略完成度 0.34、root 56 cm），只验证流程。
- 自测结束后 config 已还原：sha256 = `f756b4d0e76a752a928cf41a6317f1147f8316319d8774c4c3769eb1df8b9a82`。

## 11. 结果（2026-11-17，用户执行）

`runs/20261117-001-m4-w15-finetune`（训练，6412 s）+ `runs/20261117-002-m4-eval`（5 种子评估）：

| 指标 | 300 iter | 3000 iter | 变化 |
| --- | --- | --- | --- |
| root xy 误差 cm | 25.72 | **3.90** | −84.8% |
| root p95 cm | 76.11 | 9.19 | −87.9% |
| MPKPE cm | 26.86 | 5.64 | −79.0% |
| R-MPKPE cm | 6.10 | 3.66 | −40.0% |
| joint_rmse rad | 0.1677 | 0.1168 | −30.4% |
| joint_vel_rmse | 1.28 | 1.07 | −16.4% |
| 左脚滑移 cm | 139.69 | 91.30 | −34.6% |
| 摔倒 | 0/5 | 0/5 | 持平 |

**假设判定：H1 成立（预算确实是瓶颈），H2、H3 否。**

### 预测判分

- Q1（约 10 cm）：方向对、幅度保守（实测 3.90，你的“迭代 ×10 → 减半”低估了 2.5 倍）。
- Q2（R-MPKPE 变差）：错。6.10 → 3.66。300 iter 时代姿态项根本没收敛，加预算让所有项一起变好；“位置吃姿态”只在**已收敛后继续加权**（LAB07 的 w2.5）才出现。
- Q3（error_anchor_pos → 0.05，且“不清楚两者差在哪”）：数值偏乐观（实测 0.114）；关键在于两者不同口径——训练侧是 3D 范数（含 z）、随机抽帧、含噪声动作与 reset 瞬态；评估侧是单条确定性 rollout 的 xy 误差。详见 `runs/20261117-002-m4-eval/README.md`。
- Q4：你的前提（“迭代次数不是瓶颈”）被本实验否掉；但排查链合理。

### 但这里有个转折

同一 ckpt 换到**没参与训练**的片段（`runs/20261117-003-generalization-probe/`，n=1 探索）：

| 评估片段 | 完成度 | reset | 最长存活段 |
| --- | --- | --- | --- |
| 训练片段 S3 000–20 s | 0.999 | 1（motion_end） | 19.96 s |
| 同片段未见窗口 S3 120–140 s | 0.101 | 24（ee_body_pos） | 2.02 s |
| 未见片段 S2 134 s | 0.024 | 50（ee_body_pos） | 0.48 s |

所以 3.90 cm 是**这个片段的专家**成绩，不是“会跟踪”。按 D17：引用时必须标注 in-distribution，M5 必须写成“对特化策略的扰动”。

### 你该带走的三个概念

1. **in-distribution vs held-out**：训练集上的好成绩可能只是记忆；判定方法是换一个没训过的片段/窗口。
2. **reset 频繁 = 指标失效**：一段 rollout 里机器人每 0.3 s 被重置一次，root 误差自然小，此时只看完成度与存活时间。
3. **同口径比较**：训练日志里的 `error_anchor_pos` 与评估脚本的 `root_err_cm` 不是同一个测量，任何“训练变好了吗”的判断都要用同一口径。
