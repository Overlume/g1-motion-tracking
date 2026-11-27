# supermimic-dsh — Unitree G1 动作模仿复现（纯仿真）

目标：在本地 WSL + RTX 3060 6GB 上，从零复现"人形机器人全身动作模仿"的最小可用闭环，并加入受控外力扰动实验。最终产出一个可解释、可重复、可交接的工程目录，而不是一次性 demo。

> 公开仓库：https://github.com/Overlume/g1-motion-tracking （代码 + 文档 + 小幅结果文件；不含权重与动作数据）
>
> 项目指令来源：`<外部 brief 文档>`（用户 brief，2026-11-11 读入）。
> 第一轮核验结论见 `docs/DECISIONS.md` 与 `docs/STATE.md`。

## 结果（截至 2026-11-27，完整版见 [docs/REPORT.md](docs/REPORT.md)）

| 阶段 | 内容 | 关键数字 |
| --- | --- | --- |
| 基线 | 官方预训练策略 | dance1 完成度 1.00、MPKPE 10.74 cm；格斗 S3 **5/5 摔倒**（中位 11.66 s） |
| 调参 | 根位置奖励权重 0.5→2.5（300 iter） | root 41.52 → 22.95 cm（−44.7%），但滑移 +42.5%、R-MPKPE U 形（1.0 最好） |
| 规模 | 300 → 3000 iter（weight 1.5） | root 25.72 → **3.90 cm（−84.8%）**，R-MPKPE 6.10 → 3.66，完成度 0.999 |
| 泛化 | 同一 ckpt 换窗口 | 训练片段 0.999 → held-out 窗口 **0.101**（最长存活 2.02 s） |
| 扰动 | torso_link 世界系 ±x/±y；冲量 40/45/60/80/90/120 N·s（力 200–600 N × 0.2 s），各 5 种子 | 终止阈值：+y 0.81 m/s 起出现倒地、0.90 m/s 多数倒地；结局（终止关、推后 3 s）：120 N·s（600 N）四方向 5 次里 4–5 次真倒地；冲量—倒地率单调；**20 种子后两向边界不同（D26）**：60 N·s 时 −y 19/20（95%，CI 75–100%）vs −x 6/20（30%，12–54%），Fisher p = 3.9e-5；80 N·s 两向都 20/20（下界 0.83）；5 种子的「80 起全倒」作废；同种子 3 遍测翻转率（LAB16，D28）：−x 60 翻 4/20、−y 60 翻 1/20 → 边界格改写 5–6/20 与 18–19/20；**−y 过渡区（LAB17，D30）**：45 N·s 三遍 11/11/12（58%，32–81%，翻转 3/20）、50 N·s 15/15/15（75%，51–91%，0/20）→ 50% 交点在 40–45 N·s 之间（插值 ≈44.3），「过渡在 45–60」作废；**−x 第二分点（LAB18，D31）**：70 N·s 三遍 14/11/10（中位 55%，包络 27–88%，翻转 4/20，全部 rep1 down→后两遍 transient）→ −x 的 50% 交点 ≈68 N·s（65–70），与 −y 的 ≈44 相差 ≈23–24 N·s；**四向同口径（LAB19，D32）**：60 N·s 时 +x **0/20（全 stable）**、−x 25–30%、−y 90–95%、+y **20/20**；80 N·s 时 +x **0/20**（19 transient 全恢复）、其余 20/20 → 边界顺序 **+y < −y < −x < +x**，「80 起全倒」对 +x 作废；**补 +x 100 N·s（LAB19b，D33）**：0/20（0%，CP 上界 16.8%）、12–13/20 崩过又恢复、三分类翻转 6/20 → +x 的 down 边界收窄到 **100–120**，且 80→100 的 stable 份额非单调（1/20 → 7–8/20）；**判定审计（A1，D36）**：k=0 有两种含义——+x 80 N·s 的 57 次 transient 偏离仅 0.2 s、末尾 root 误差 4.5 cm（擦边），而 DR 加强权重的 −x 60 N·s 是 **60/60 退让**（偏离 0.72–0.90 s、root 93 cm），详见 §2.5.14 |
| 训练期 DR | 三档 startup DR 同时加强（摩擦 0.2–1.4 / 质心 ±4,±6,±6 cm / 编码器 ±0.02 rad），起点/动作/seed/迭代同 M4（+3000 iter） | **负结果**：train 0.999 不变；held-out **0.100 → 0.018**（10/10 更差，等于官方权重的 0.016）；unseen 0.025 → 0.024 持平 → held-out 完成度应读作「窗口记忆」而非泛化（LAB20，D34） |
| DR 抗推对比 | 把 LAB12 协议搬到 DR 加强权重（60 N·s 两向 × 20 种子 × 3 遍 = 120 次，仅换 ckpt） | **-x 失败率 30% → 0%**（0/0/0，p=0.020，6 个失败种子全部转站住），**-y 无收益**（20/19/20 vs 19/18/18，p=1，seed102 系统性变差）；但 -x 的 20/20 全为 transient、骨盆从未触地、末尾 root 误差 18.7 → **92.9 cm** → DR 的收益是「退让式不倒地」，不是「不丢跟踪」（LAB20b，D35） |
| DR 单档消融 | 三档加强拆成三个单因素臂（fric 摩擦 0.2–1.4 / com 质心 ±4,6,6 cm / bias 编码器 ±0.02 rad），训练口径同 M4 | **只有摩擦档降 −x 失败率**：nx_j60 逐遍 **2/2/2**（10%，池化 p=0.019）、com 13/15/15（72%）、bias 19/19/19（95%），M4 6/6/5、LAB20 0/0/0；但 fric 的 54/54 非倒地全是**漂移型**（被推走 81 cm，骨盆 0.418 m 未触地）；三臂 held-out **全 0.018** → 崩塌与档位无关（LAB23，D37） |
| 摩擦档强度扫描 | 只放宽摩擦上界：f14 0.2–1.4（复用 LAB23 fric ckpt）/ f20 0.2–2.0 / f30 0.2–3.0，com/bias 回默认；nx_j60 × 20 种子 × 3 遍 | **非单调**：逐遍 f14 **2/2/1**（8%）、f20 **10/14/15**（65%）、f30 **0/0/0**（0%）；非倒地 root 中位 f14 82.3 / f20 68.7 / f30 **18.7 cm**（≈M4 的 18.2、p75 38.6、max 61.8）→ 只有 3.0 档同时满足判据，但 f30 仍 21/60 漂 ≥30 cm、单 seed 三遍极差 49.5 cm，不能写「漂移解决」（LAB24，D38） |
| 摩擦上界收敛 | 每档只换训练 seed（f20/f30 各复训 seed 1，seed0 复用 LAB24）＋ f30 零训练扩测 −y 45/60 N·s | **f20 的 65% 不复现**（seed1 5/5/4 = 23%，p=1.0 vs M4；但 root 68.7 → 71.4、非倒地 46/46 全漂移）；**f30 的 0/60 只部分复现**（seed1 1/3/3 = 12%，p=0.09–0.70）；**3.0 档确实改 −y**：45 N·s 从 M4 的 34/60 降到 **0/60**（p<1.5e-4，但 49/60 退让、root 87 cm），60 N·s 仍 20/20 → 「−y 无收益」只在饱和点成立（LAB25，D39） |
| −y 过渡区补点 | f30（摩擦 0.2–3.0）零训练补 ny_j50/ny_j55，20 种子 × 3 遍 | **ny_j50 5/5/4（14/60，23%，vs M4 75%，p=0.004）、ny_j55 20/19/19（58/60，97%）→ 50% 交点 ≈44.4 → ≈51.8（+7.4 N·s）**；代价是非倒地 root 中位 87 → 111 → 121 cm、退让占比 82% → 100% → 3.0 档是「延迟 7 N·s」而非「抬高天花板」（LAB26，D40） |
| 参考质量 | LAB22 对 4 段参考逐帧 FK 查脚底 | **训练参考本身不落地**：s3_000s 穿地 max **59.6 mm**、p99 38.6、接触仅 **34%**、最长双足离地 **3.82 s**、脚底中位悬空 **+2.8 cm**；4 段里 3 段 FAIL（仅 held-out 段 16.7 mm 合格）；同流水线 dance1 只飘 0.8 cm → 归因重定向高度标定；策略以骨盆 **−1.62 cm** 被动补偿 → M4 的 3.90 cm root 误差含 ≥1.6 cm 参考不可行成分（LAB22，D41） |
| 观测加世界系根位姿 | actor 160 → 166（+根世界位置 3 维、+世界系速度 3 维），其余同 M4 | **负结果**：train 0.999 不变；held-out **0.100 → 0.018**、存活 1.98 s → **0.34 s**（`ee_body_pos` 终止 61 次、不是摔倒）；unseen 0.025 → 0.038 但误差更差；第一层权重列范数 位置 0.12/0.12/0.22、速度 1.14/0.92/1.47（老维度中位 1.68）→ 模型只用了世界系速度（LAB21，D42） |
| 多片段联合训练 | 训练加第二段动作：dance1 0–20 s（地面校正后，穿地 32.9 → 3.5 mm），其余同 M4 | **只涨训练动作家族**：dance1 的另一窗口 0.061 → **0.108**（+0.046，9/10 更好），但 held-out 0.099 → 0.097、unseen 0.025 → 0.026 持平；train 完成度 0.999 不变而 MPKPE +33%、root +56% → 是「多记一段」不是泛化（LAB27，D43） |
| 窗口距离曲线 | 同一段动作（dance1）的 6 个 20 s 窗口（0–120 s）× M4/LAB27 × 10 种子，零训练 | **曲线非单调**：LAB27 = 0.999 / 0.244 / 0.055 / 0.108 / 0.309 / 0.101，5/6 窗口优于 M4（4 个 10/10 一致），但 40–60 s 反向 −0.060；M4 基线自身 0.061–0.484（0 s 窗最慢 4.01 rad/s → 完成度被难度污染）；首段对齐 MPKPE 才可比、内容新颖度 vs Δ ρ = −0.886（n=6 探索性）→ **记忆没有可测的时间半径**（LAB28，D44） |
| 可重复性 | 同配置同种子重跑 3 遍（−y 60 / 80 N·s × 5 种子） | 倒地数 **4/4/4 与 5/5/5**，倒地种子集合与唯一 transient 种子三遍完全不变 → 重跑噪声（判定层）= 0；数值层有位级抖动（骨盆最低 ±1.1 cm、末尾 root ±3 cm、最长偏离 ±0.34 s）；故 20% 差别是种子抽样而非重跑噪声，5/5 的 p(down) 下界仍为 0.48（D25） |

> 结论一句话：**3.90 cm 是“这个 20 s 窗口的专家成绩”，不是“学会了动作模仿”**；本仓库把两者分开报，并给了受控扰动下的失稳边界。

![受控外力扰动：四个方向的失稳边界](outputs/figures/m5_perturbation.png)

![根位置奖励权重扫描：误差下降但姿态先好后坏](outputs/figures/weight_sweep_lab07.png)

## 现在从哪里看

1. `docs/STATE.md` — 当前阶段、已验证事实、下一步最小动作。
2. `docs/PLAN.md` — 里程碑与验收标准。
3. `docs/DECISIONS.md` — 已做的路线/许可/环境决定及理由。
4. `docs/SOURCES.md` — 所有外部来源与核验日期。
5. `docs/ARTIFACT_INDEX.md` — 仓库、数据、模型、日志的实际位置。

## 目录

```
repos/        第三方仓库（只读，不改动其内部约定）
  mjlab_research/    mujocolab/mjlab v1.6.0（MuJoCo Warp 训练框架）
  supermimic/        HybridRobotics/supermimic（可微世界模型方法，进阶对比）
src/          本项目自己的代码（转换、评估、扰动）
scripts/      可重复执行的脚本（环境、数据、运行）
configs/      实验配置快照
data/         raw / processed / manifests（大文件不入 Git）
runs/         每次实验一个目录，含命令、配置、指标、视频
outputs/      videos / figures / reports
environment/  项目内隔离的 Python 环境与缓存
docs/         状态、计划、决策、来源、学习记录、交接
scratch/      临时探查（pdf、下载样本、实验性脚本）
```

## 快速开始（2026-11-14 实测可用）

```bash
# 1) 环境（项目内隔离：venv 在 environment/.venv，uv/Python 也下载到 environment/）
bash scripts/setup_env.sh

# 2) 回放官方预训练 checkpoint（单台）
MUJOCO_GL=egl environment/.venv/bin/python scripts/replay_policy.py \
  --checkpoint-file scratch/model_49999.pt \
  --motion-file data/processed/lafan_dance1_subject1.npz \
  --video-length 600 --num-envs 1 --width 960 --height 540

# 3) 4 台并排回放（取景默认自动装下所有 env）
MUJOCO_GL=egl environment/.venv/bin/python scripts/replay_policy.py \
  --checkpoint-file scratch/model_49999.pt \
  --motion-file data/processed/fight1_s3_000s_20s.npz \
  --video-length 400 --num-envs 4 --width 960 --height 540 --tag demo4

# 4) 量化评估（输出 runs/<dir>/summary.json + metrics.csv）
MUJOCO_GL=egl environment/.venv/bin/python scripts/eval_tracking.py \
  --motion-file data/processed/fight1_s3_000s_20s.npz \
  --seed 101 --keep-dr --ignore-terminations \
  --out runs/YYYYMMDD-001-eval
```

### 取景参数（2026-11-14 新增，旧命令不用改）

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `--camera` | `follow` | follow=相机跟着所有机器人的中心并自适应距离；fit=静态装下网格；auto=mjlab 原来的跟随相机 |
| `--camera-margin` | 1.3 | 取景余量，越大相机越远 |
| `--camera-elevation` | -16 | 相机仰角（度） |
| `--camera-azimuth` | 45 | 相机方位角（度），45=斜对棋盘格 |
| `--camera-min-distance` | 5.2 | follow 模式距离下限，越大越不容易被摔倒的机器人切边 |

背景：mjlab 默认 `viewer.max_extra_envs=2`，只渲染「主 env + 2 个邻居」，所以 4 个 env 只显示 3 台机器人（与分辨率无关）；`--camera auto` 走 tracking 模式跟随单台，env 散开后容易出画。修复说明见 `docs/STATE.md` 的「取景修复」一节。

## 文档地图

- `docs/COMPLETION_CRITERIA.md` — 项目「完成」的判定条件、必跑实验、不在范围内的事项
- `docs/GITHUB_UPLOAD_PLAN.md` — 上传清单、`.gitignore`、提交划分、上传前检查
- `docs/learning/` — 带练 LAB（用户亲手做的实验记录）
- `docs/handoffs/HANDOFF.md` — **唯一**交接文档（**仅本地**，已写入 `.gitignore`；换会话只带这一份即可接手）
