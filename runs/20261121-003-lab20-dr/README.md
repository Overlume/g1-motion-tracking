# LAB20 训练期 DR 加强（vs M4 默认 DR）

- 目的：只改 **DR 强度**（摩擦 0.2–1.4 / 质心 ±4,±6,±6 cm / 编码器 ±0.02 rad），其余与 M4 一致，看三窗口完成度变化。
- 假设（跑前写死）：held-out 完成度从 0.101 提升（≥0.20 才算实质），训练窗口不低于 0.99，unseen 至少不变得更差。
- 结果：**假设全部不成立**——train 不动（0.999），heldout **0.100 → 0.018（−0.081，10/10 更差）**，unseen 持平（0.025 → 0.024）。

## 1. 协议（已核对）

| 项 | M4（对照） | LAB20 | 核对证据 |
| --- | --- | --- | --- |
| 起点权重 | scratch/model_49999.pt | 同一个 | 热启动副本 sha256 f1bc404f…f22319 |
| 动作 | data/processed/fight1_s3_000s_20s.npz | 同一个 | train.log 第 9 行 |
| 并行环境 / 迭代 / seed | 4096 / 49999→52998 / 0 | 同 | logs/g1_tracking/2026-11-21_19-27-09_lab20-dr/params/*.yaml |
| DR 摩擦 | 0.3–1.2 | **0.2–1.4** | env.yaml foot_friction.params.ranges: (0.2, 1.4)、shared_random: true |
| DR 质心 | ±2.5/±5/±5 cm | **±4/±6/±6 cm** | env.yaml base_com.params.ranges: {0: (-0.04,0.04), 1: (-0.06,0.06), 2: (-0.06,0.06)} |
| DR 编码器 | ±0.01 rad | **±0.02 rad** | env.yaml encoder_bias.params.bias_range: (-0.02, 0.02) |
| 训练 logger / save_interval | tensorboard / 500 | 同 | params/agent.yaml |
| 评估 | eval_tracking.py，终止开、启动 DR 关、999 步 | 同 | eval/*/summary.json 里 ignore_terminations: false、keep_startup_dr: false |

评估种子 **101–110**；M4 与 LAB20 **在同一轮、同一脚本、同一组种子上重跑**（配对比较，不拿 LAB09 的旧数字）。

## 2. 训练

- 命令：environment/.venv/bin/python -u scripts/lab20_dr_train.py --run-root runs/20261121-003-lab20-dr --mode full（脚本把 DR 三档写进 cfg 后调 mjlab 的 launch_training）
- 墙钟：**1:35:01（5701 s）**，比 M4 的 6412 s 快 11%（同迭代数；DR 只影响 startup，不增加每步开销）
- 迭代：49999 → 52998（+3000），落在 model_50000/50500/51000/51500/52000/52500/52998.pt
- 训练产物 ckpt：model_52998.pt，sha256 `e3747012f3c7d35f9b1de4720a8420b6c1fb7f53c3c8e00c3ed2afcd2625b287`
- train.log **840 MB**（整段 stdout，未跟踪；不需要时可删，指标已在 tfevents 里）

训练条件（DR 开）下的末尾指标对照（tfevents，iteration 52998）：

| 指标 | M4 | LAB20 | 读法 |
| --- | --- | --- | --- |
| Episode_Termination/motion_end | 9.50 | 9.92 | 跑完的 episode 比例差不多（甚至略高） |
| Episode_Termination/ee_body_pos | 0.208 | 0.292 | 末端跟踪超限略多 |
| Metrics/motion/error_anchor_pos | 0.114 | **0.212** | 锚点跟踪误差 **翻倍** |
| Episode_Reward/motion_global_root_pos | 0.701 | **0.172** | 根位置跟踪奖励掉到 1/4 |

## 3. 评估（60 次）

- 命令：LAB20_RUN=runs/20261121-003-lab20-dr bash scripts/lab20_eval.sh run
- 规模：2 策略 × 3 窗口 × 10 种子 = **60 次**；dr 半 30 次 09:23–09:43 ≈ 20 min（约 40 s/次）
- 产物：eval/lab20_eval.md（表）、eval/lab20_eval.json、eval/lab20_eval.png、eval/<policy>/<window>/seed<N>/summary.json

| 窗口 | M4（默认 DR） | LAB20（DR 加强） | 差 |
| --- | --- | --- | --- |
| 训练片段（S3 0 s） | 0.999 ± 0.000 | 0.999 ± 0.000 | **+0.000**（10/10 相同） |
| 未见窗口（S3 120 s） | 0.100 ± 0.002 | **0.018 ± 0.000** | **−0.081**（10/10 更差） |
| 未见片段（S2 134 s） | 0.025 ± 0.001 | 0.024 ± 0.006 | −0.001（8 差 2 好，全在 ±0.007 内） |

## 4. 判读

1. **DR 加强没有换来泛化**：heldout 从 0.100 掉到 0.018，降幅 81%，10 个种子无一例外。
2. **掉到的是「官方权重水平」**：同一口径下官方预训练权重 heldout = 0.016（LAB09）。LAB20 的 0.018 ≈ 官方 → M4 那 0.101 的「残余泛化」在 DR 加强后消失，说明它更可能是**训练窗口邻近帧的记忆外溢**，不是可迁移能力。
3. **专精没丢**：train 0.999，10 个种子逐一相同 → 不是「DR 太强把策略练废了」，而是「练稳了但没有变强」。
4. **训练条件指标变差**：锚点误差 0.114 → 0.212、root 位置奖励 0.701 → 0.172，而 episode 跑完比例反而略高 → 策略学会了「在扰动下别摔、把动作跑完」，代价是跟踪精度下降；这个权衡没有转化为新窗口上的能力。
5. **unseen 完全没有变化**（0.024 vs 0.025）→ 与「DR 加强不能创造新技能」一致。

## 5. 事故与修正（重要，方法论）

第一次跑评估时，scripts/lab20_eval.sh 用 ls .../model_*.pt | sort -V | tail -1 选 DR ckpt，pretrained/ 目录按字典序排在运行目录之后 → 选中了**热启动副本** pretrained/model_49999.pt（=官方权重），第一版表因此显示 train 0.381 / heldout 0.016 / unseen 0.014（正好是官方权重的成绩），会被误读成「DR 让一切崩掉」。

- 证据：第一版 eval.log 第 3 行写的是 DR ckpt: .../pretrained/model_49999.pt（sha f1bc404f…）。
- 修正：dr_ckpt() 加 grep -v "/pretrained/"，删掉 eval/dr 后用正确 ckpt 重跑 30 次（约 20 min）。
- 教训：**评估表的第一件事是核对 ckpt 的 sha**，不是看数字。讲义 §4 已把这条写进 FAQ。

## 6. 局限

- 只测了一种「加强档」（三档同时加强、强度有限），**没有单档消融**，不能判断是摩擦/质心/编码器哪一项造成 heldout 下降。
- 只训练了一个 seed（0），训练侧方差未知。
- heldout 只有 999 帧一个窗口（S3 120–140 s），unseen 只有 S2 一个片段。
- 没有测扰动鲁棒性：DR 加强的预期收益本该在受控外力下体现，这一项留给后续（用 LAB12 协议在同一 ckpt 上跑 60 N·s 两向即可，约 40 min）。

## 7. 文件清单

| 路径 | 内容 |
| --- | --- |
| README.md | 本文件 |
| train.log（840 MB，未跟踪） | 训练 stdout |
| logs/g1_tracking/pretrained/ | 热启动权重副本（不是训练产物） |
| logs/g1_tracking/2026-11-21_19-27-09_lab20-dr/ | ckpt、tfevents、params（env/agent）、git diff |
| eval/lab20_eval.md / .json / .png | 三窗口对比表 / 机读 / 分组柱图 |
| eval_summary.md / eval_summary.json（run 根目录） | 上表的入库副本（.gitignore 只放行 runs/<run>/*.md|json） |
| eval/<policy>/<window>/seed<N>/ | 60 份 summary.json + metrics.csv |
| eval.log | 第一版（错误 ckpt）评估日志，保留作事故证据 |

## 8. 复现

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
# 训练（≈1.6 h）
environment/.venv/bin/python -u scripts/lab20_dr_train.py --run-root runs/<新run名> --mode full
# 评估（≈40 min；M4 与 DR 都在同一脚本里重跑）
LAB20_RUN=runs/<新run名> bash scripts/lab20_eval.sh run
```

注意：--mode full 默认 3000 iter / 4096 envs，DR 三档默认就是本 lab 的加强值；要跑对照臂加 --default-dr。
