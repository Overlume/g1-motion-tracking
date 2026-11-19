# LAB12 — 关掉终止之后：那 31 次"站着被判罚"到底倒没倒

> 前置：LAB11（姿态审计）。你已经知道 80 次扰动评估里，**survived 39 / collapse 10 / upright_trip 31**，
> 而 collapse 全部出现在 +y（侧向），upright_trip 则散在 +x 120、−x 60、−x 120、+y 120、−y 60、−y 120 六格。
> 你也已经看到限制：姿态审计只能看**复位前 0.4 s**的那一帧姿态，因为 M5 的评估开着 termination，
> 推后 0.14–1.22 s（中位 0.22 s）就被复位了，**之后再发生什么，数据里没有**。

## 0. 一句话

这一次不加新扰动、不训练，只做一件事：**把那 6 个"站着被判罚"的格，关掉 termination 重跑，推后看 3 s**，
用"有没有真的倒下"给 M5 的 31 次 upright_trip 定性。成本约 20 min，只占 GPU 一小会儿。

## 1. 为什么必须单独做这一步

M5 的结论目前有两层，可信度不同：

| 说法 | 证据 | 可信度 |
| --- | --- | --- |
| **阈值排序** +y ≪ −x ≈ −y ≪ +x（至少差 2 倍） | 12 格 × 5 种子的终止统计 | 高（只看"是否触发判罚"，与倒地与否无关） |
| **终止 = 失稳倒地** | LAB11 姿态审计：10 次 collapse 在复位前已达"比参考低 8 cm 或歪 13°" | 只对 +y 的 10 次成立 |
| **upright_trip = 站着但被判定出局** | 复位前姿态仍接近参考 | 中：只说明"复位那一刻还站着"，不说明"后来站住了" |

第三行就是缺口。评估脚本在终止时 break 并复位，机器人被"掐断"在推后 0.2 s 左右，
我们既不能说它倒了，也不能说它稳住了。D21 因此只允许把 upright_trip 写成"误差判罚，结局未知"。

LAB12 用 --ignore-terminations 把这个缺口补上。

## 2. 实验设置

| 项 | 值 |
| --- | --- |
| 策略 | 自己的 M4 策略 model_52998.pt（4096 env × 3000 iter，weight 1.5） |
| 动作 | data/processed/fight1_s3_000s_20s.npz（训练目标片段，in-distribution） |
| 格（6 个） | px_j120、nx_j60、nx_j120、py_j120、ny_j60、ny_j120（M5 里出现 upright_trip 的全部格） |
| 种子 | 101–105（与 M5 / LAB11 完全相同的种子） |
| 冲量 | j<N> → 世界系力 N/0.2 s，持续 0.2 s，施加在 torso_link，t = 8.0 s 起 |
| termination | **关闭**（--ignore-terminations） |
| 评估长度 | 不设上限，跑满 999 步（19.98 s）或直到骨盆真的低于 0.15 m 连续 5 步 |
| 判分窗口 | 8.0–11.2 s（推力 0.2 s + 之后 3 s） |

**注意：termination 关掉不等于不会结束。** eval_tracking.py 里还有一层独立的 fall 判据
（骨盆高度 < 0.15 m 连续 5 步 → 打印 FALL 并 break）。真的平躺下去时数据会在倒地那一刻结束，
t_end 就是摔倒时刻——这层我们**故意保留**，它是"确定倒了"的硬证据。

## 3. 三类判定怎么算

判分器 scripts/lab12_recovery_report.py，阈值与 LAB11 的姿态审计同源：

- **偏离帧**：骨盆比参考低 ≥ 0.08 m，**或**躯干倾角比参考歪 ≥ 13°（参考 = 同一帧的动作参考姿态）
- **崩溃**：8.0–11.2 s 内**连续** ≥ 10 帧（0.2 s）偏离
- **恢复**：数据最后 0.5 s 的平均下移 < 0.05 m 且平均倾角差 < 8°

| 判定 | 含义 | 条件 |
| --- | --- | --- |
| stable | 推完就稳住，没有连续 0.2 s 明显偏离 | 崩溃不成立 |
| transient | 崩了一会儿又站回来了 | 崩溃成立 + 末尾恢复 |
| down | 崩了没起来（含直接触发 fall 判据） | fall 命中，或崩溃成立 + 末尾未恢复 |

**为什么末尾要取"可用数据的最后 0.5 s"而不是固定 11.2 s**：fall 命中会 break，数据在 9 s 左右就断了，
拿 11.2 s 的窗口去取平均只会得到 nan。这一条我在自测里踩过一次，已修。

## 4. 操作步骤

先写预测（第 5 节）到 docs/learning/LAB12-prediction.md，然后：

~~~bash
cd /home/overlume/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh

# 1) 跑 30 次评估（6 格 x 5 种子，约 20 min）。脚本自己建 runs/YYYYMMDD-NNN-lab12-recovery/
bash scripts/lab12_recovery.sh run

# 2) 判分（run 里已自动调用过一次；想重算就单独跑）
bash scripts/lab12_recovery.sh report

# 3) 只看汇总表
bash scripts/lab12_recovery.sh show
~~~

结果落在 runs/YYYYMMDD-NNN-lab12-recovery/：

| 文件 | 内容 |
| --- | --- |
| params.txt | ckpt / sha256 / motion / 格 / 种子 / 推力窗口 / termination=OFF |
| <cell>/seedNNN/metrics.csv | 逐帧：time_s、pelvis_z、ref_pelvis_z、tilt_deg、ref_tilt_deg、root_err_cm… |
| <cell>/seedNNN/summary.json | fall_detected、reset_count 等 |
| recovery.md / recovery.json | 判分结果（汇总表 + 30 行逐次表） |
| wallclock_s.txt | 总墙钟（用 /proc/uptime，避免 D14 的时钟回跳） |

想先试一格再跑全套：

~~~bash
LAB12_CELLS="py_j120" LAB12_SEEDS="101" bash scripts/lab12_recovery.sh run
~~~

## 5. 先预测再实测（写进 docs/learning/LAB12-prediction.md）

对着 LAB11 的表填，**不要改答案**：

**Q1.** 30 次里 stable / transient / down 各多少次？（三类之和 = 30）

**Q2.** 六个格里，哪一格最可能出现 transient（崩了又恢复）？给一个格名，并说理由。

**Q3.** py_j120 与 nx_j120 这两格，谁的 t_end（数据结束时刻：倒下或跑完）更早？大约差多少秒？

**Q4.** 这一步能不能检验"侧向支撑窄，所以 +y 更脆"这个解释？如果能，需要看到什么结果；如果不能，缺的是什么？

**Q5（选做）.** 如果 6 格全部 down 且 t_end 都落在 8.2–9.5 s，M5 那张"阈值排序"表还有多少信息量？

## 6. 跑完按这个顺序读

1. **先看 recovery.md 汇总表**：三类计数。stable 多 → D21 偏保守，upright_trip 多数确实站住了；
   down 多 → 31 次里多数只是"没撑到复位那一刻"的延迟倒地，M5 的"抗扰"表述要再收紧。
2. **再看逐次表的 t_end 与 fall 列**：fall=是 是硬证据（骨盆 < 0.15 m）；
   t_end ≈ 19.98 s 才是"跑满全程、真稳"。
3. **看方向差异有没有消失**：M5 排序是 +y ≪ −x ≈ −y ≪ +x。若六格全 down，排序仍成立，
   但含义从"哪个方向能扛住"变成"哪个方向倒得更快"，看 t_end 中位数差多少。
4. **最后看 transient 的存在性**：只要出现一次"崩了又站回来"，就说明策略有恢复能力，
   M5 的"没有踉跄后站稳"要限定为"在 0.2 s 复位窗口内看不到恢复"。

## 7. 常见坑

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 报 --perturb-force: expected one argument | 负号开头的值被 argparse 当选项 | 脚本已用等号形式，手敲也要带等号 |
| report 报"窗口内没有数据" | 该 rollout 在 8.0 s 前就结束了 | 正常不会出现，检查推力时刻是否被改 |
| recovery.md 里 end_drop 是 nan | 老版本用固定 11.2 s 取窗口 | 用当前版本重跑 report |
| 重跑 run 又新建了一个目录 | run 子命令总是新编号 | 想追加/重算用 LAB12_RUN_DIR=runs/... bash scripts/lab12_recovery.sh run |
| metrics.csv 只有 5–6 秒数据 | 该 rollout 推后不久就 fall 了（正常） | 这正是 down 的证据，不是 bug |
| 别拿这些 run 的 root_err 做对比 | 数据在倒地时刻就断了，分母不一样 | 只看三类判定与 t_end |

## 8. 通过标准

- [ ] 30 个 rollout 全部产出 metrics.csv + summary.json（6 格 × 5 种子）
- [ ] recovery.md 汇总表三类计数之和 = 30
- [ ] 每个 down 都标了 fall 是/否，且 fall=是 的 rollout 骨盆最低值 < 0.15 m
- [ ] 能回答 Q1–Q5，且 Q1 的误差能说清错在哪一步（把 down 估多了还是 transient 估多了）

## 9. 自测记录（智能体，非本次结果）

写脚本时用 2 个格各 1 个种子跑通全流程（scratch/lab12-selftest，gitignored）：

| 格 | 种子 | 判定 | fall | t_end (s) | 骨盆最低 (m) | 最大倾角差 | 末尾下移 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| py_j120 | 101 | down | 是 | 8.96 | 0.094 | 172.5° | +0.371 m |
| nx_j120 | 101 | down | 是 | 9.18 | 0.089 | 149.7° | +0.474 m |

两例都在推后约 1 s 内真的平躺下去。**这只是 2 个样本，不能当结论**，但足以说明：
关掉 termination 后，"站着被判罚"的格很可能并不真的站得住。你的 Q1 请认真估。
