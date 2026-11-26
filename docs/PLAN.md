# PLAN

按阶段推进；每个里程碑有明确验收标准，未达成不进入下一步。

## M0 阶段 0：核验与环境基线（完成 2026-11-11）

- [x] 硬件/WSL/GPU/软件只读核查
- [x] SuperMimic 身份、仓库、代码完整度、数据与许可核验
- [x] mjlab 基线能力与 G1 任务核验
- [x] 项目内隔离环境可 import mjlab / torch 可见 GPU
- [x] 最小 GPU 冒烟：环境创建 + 若干步 step，记录显存与耗时（runs/20261111-001-smoke）
- [x] 路线决定（用户授权智能体代选：mjlab 基线优先）
验收：`docs/STATE.md` 中"已完成"小节全部有证据；冒烟日志在 `runs/`。

## M1 阶段 1：G1 仿真与现成策略回放（完成 2026-11-12）

- [x] 跑通 `mjlab demo`（或等价 play）看到 G1 动作
- [x] 记录：显存峰值、num_envs、帧率、是否 CUDA graph 生效（runs/20261112-003-train-scale）
- [x] 明确区分：MuJoCo 运动学播放 vs 策略闭环驱动（回放走 checkpoint + 物理闭环；ghost 才参考姿态）
- [x] 量化评估脚本与 seed=0 基线（runs/20261112-001-eval：MPKPE 10.74 cm、R-MPKPE 3.42 cm、无跌倒）
验收：视频/截图 + `runs/` 记录 + 显存实测数字。

## M2 阶段 2：参考动作与转换

- [x] 选定动作来源：LAFAN1（先 dance1），后续 GMR 重定向；不用 BONES-SEED（D4）
- [x] LAFAN1 BVH -> GMR -> CSV -> mjlab npz 流水线打通（2026-11-13，`scripts/gmr_bvh_to_csv.py` + `scripts/csv_to_npz_local.py`）
- [x] 首个自制动作 `data/processed/fight1_s2_134s_20s.npz` 回放 + 量化评估（runs/20261113-001；2026-11-13 16:20 更正：runs/20261113-002 的 11.7 cm 作废，当时脚本忽略 dones，测的是 0.24 s 循环）
- [x] 跨 subject 一致性（LAB02：fight1_subject3；v2 重测见 runs/20261113-004，termination 开时 3 次 reset、10.58 s 之后的段不可用）
- [x] 评估协议 v2：1 env + 逐帧 done/reset 记录 + 分段 drift_rate + --ignore-terminations（runs/20261113-004~009）
- [ ] 校验补充项：脚底接触/穿地、重定向物理可行性
验收：转换前后对比图 + 转换日志 + 关节名逐项核对表。

## M3 阶段 3：训练理解与最小训练

- [x] 冒烟训练：512 env × 100 iter（runs/20261114-006，用户亲手，稳态中位 0.89 s、首次 11.84 s、显存峰值 505 MiB）
- [x] 证明参数确实更新（loss/策略变化）、检查点可回放（model_0→model_99 的 actor MLP 权重 mean|dw|≈7.3e-3；ckpt 已被 replay_policy 读回）
- [x] 用户亲手改一个有明确意义的参数，先预测后对比：2026-11-15 完成 LAB05（A/B 各 300 iter + 5 种子评估，`runs/20261115-002-m3b-eval`），三道预测命中 1/3；结论见该 run 的 README 与 D13
验收：训练曲线 + 检查点 + 用户练习记录（`docs/learning/`）。

## M4 阶段 4：武术类动作跟踪

- [ ] 选一个双足支撑、慢速、幅度适中的动作（如冲拳/太极起势）
- [ ] 训练到可完成，记录完成率/跟踪误差/摔倒率
验收：对比视频 + 指标表。

## M5 阶段 5：受控扰动

- [ ] 定义扰动协议（对象、坐标系、方向、大小、持续时间、动作阶段）
- [ ] 实现扰动注入（力/冲量，先小后大）
- [ ] 基线 vs 改进策略同条件对比，记录恢复时间与摔倒率
验收：协议文档 + 可重复脚本 + 表格。

## M6 阶段 6：复现总结

- [ ] 汇总所有 `runs/`，写复现报告与局限


### M3 训练侧改动清单（2026-11-13 拟定，每条都要同一套验收）

- [ ] 微调基线：从 scratch/model_49999.pt 出发，用 fight npz 做数百次迭代，记录 4096 env 的耗时/显存（已有 1.76-1.90 s/iter、1871 MiB）。
- [ ] 全局项：motion_global_root_pos 权重 0.5 -> 1.0、std 0.3 -> 0.15；对照 run 只改这一项。
- [ ] 观测补全局：把根相对世界的水平位置/朝向（或根速度）加入 policy 观测，解决看不见自己漂移的问题。
- [ ] 终止项补水平漂移：在 anchor_pos 系列里加水平分量，避免训练时静默漂移。
- [ ] 验收：同一片段（S3 + dance1）、同一 seed、同一评估脚本，对比 root_err_mean、drift_rate、R-MPKPE、摔倒次数。


### 取景与种子分布补充（2026-11-14）

- [x] 4 env 取景修复（`scripts/camera_fit.py` + `--camera fit`）：实测 4 台全部入镜（`outputs/videos/model_49999/960x540/cam_fit_final`）
- [x] S3 摔倒分布 N=5（`runs/20261114-001~005`）：5/5 摔倒，中位 11.66 s，范围 5.54–18.56 s
- [ ] 用户按原命令复验取景
- [x] LAB04：用户亲手起一次训练（512 env × 100 iter），先预测再实测 —— 2026-11-14 完成，runs/20261114-006（LAB04 文档与脚本已就绪：docs/learning/LAB04-smoke-train.md、scripts/lab04_smoke_train.sh；2026-11-14 用 64 env × 2 iter 自测脚本通过）
- [x] M3-b 微调对照（LAB05）：从 `scratch/model_49999.pt` 续训，单参数改动（global root pos 权重 0.5 -> 1.0、std 0.3 -> 0.15），同动作 S3、同 seed 0、各 300 iter，再各 5 种子评估。讲义 docs/learning/LAB05-ab-finetune.md，脚本 scripts/lab05_ab_finetune.sh（2026-11-14 就绪，自测通过）。**2026-11-15 完成**：两组 5/5 不摔、完成度 0.999，但 B root_err 43.34 -> 52.72 cm 更差


### 交接状态（2026-11-14 15:20）

- [x] 相机视角定稿（用户 15:19 确认"可用，不要再去改动"）：默认 follow / margin 1.3 / elevation -16 / azimuth 45 / min-distance 5.2
- [x] S3 五种子摔倒分布（5/5 摔倒，中位 11.66 s）
- [x] **已过**：LAB04 冒烟训练（2026-11-14，`runs/20261114-006`）→ M3-b 单参数对照（2026-11-15，`runs/20261115-002`）
- [x] **LAB06 只改 weight 的对照**（2026-11-16 完成，`runs/20261116-003`）：w1.5/std0.3 相对 w0.5/std0.3 的 root 误差 41.52 -> 25.72 cm（-38.1%，5 种子分布不重叠），R-MPKPE -2.4%、joint_rmse +0.5%，两组 0/5 摔；H1 成立、H2/H3 不成立（D15）。
- [x] **LAB07 权重扫描**（2026-11-16 完成，`runs/20261116-006-lab07-eval`）：root 41.52 / 30.86 / 25.72 / 22.95 cm（w0.5/1.0/1.5/2.5），相对上一档 -25.7% / -16.7% / -10.8%（单调下降、边际递减）；R-MPKPE 6.25 / 5.98 / 6.10 / 6.45（1.0 触底、2.5 出界）；四臂 0/5 摔。M4 定为 weight 1.5（D16）。
       - [x] **M4 规模训练**（2026-11-17 完成，`runs/20261117-001-m4-w15-finetune` + `runs/20261117-002-m4-eval`）：300 → 3000 iter 把 root xy 25.72 → 3.90 cm（−84.8%）、R-MPKPE 6.10 → 3.66、0/5 摔；H1 成立。**但泛化探针（`runs/20261117-003-generalization-probe`）显示换片段 0.3–2 s 即终止，属特化（D17）**
       - [x] **LAB09 泛化探针**（2026-11-17 完成，`runs/20261117-004-lab09-generalization`）：m4 train 0.999/3.88 cm vs held-out 0.101（2.02 s）vs S2 0.025；官方 train 0.333/82.83 cm、held-out 0.016 —— 3.90 cm 确认为 in-distribution 特化（D17）；终止开/关口径见 D18
       - [x] **M5 受控扰动**（2026-11-18 完成，`runs/20261117-005-m5-perturb`）：+y ≤0.90 m/s 崩、−x/−y 0.90–1.80、+x 1.80–3.60；未终止格后 0–1 s 13.2–54.4 cm（基线 3.88）；口径见 D19。2026-11-18 细化 +y（`runs/20261118-001-m5-threshold-refine`）：0.75 m/s 崩、0.90 m/s 终止，口径见 D20；2026-11-18 视频核对 4 例（`runs/20261118-002-m5-video-check`）确认终止格=跪坐倒地，`fall_detected` 只认平躺（D20 修订）；2026-11-18 零 GPU 姿态审计 80 次（`runs/20261118-003-posture-audit`）：39 跑完 / 10 姿态崩（全在 +y）/ 31 站立误差判罚（D21）
       - [x] **M6 报告与提交**（2026-11-18 完成）：`docs/REPORT.md`、`configs/perturb_protocol_v1.json`、`outputs/figures/`、README 结果表与结果图、`LICENSE` + `THIRD_PARTY_NOTICES.md`、8 次提交并已 push 到 Overlume/g1-motion-tracking（2026-11-18）
       - [x] **上传**（2026-11-18）：public 仓库 https://github.com/Overlume/g1-motion-tracking ，分支 `main`，作者 Overlume <overlume@qq.com>
- 完整交接：`docs/handoffs/HANDOFF.md`（**唯一**交接文档，**仅本地**，不入库；2026-11-20 合并旧交接后持续原地更新，D29）；当前事实以 `docs/STATE.md` + `docs/DECISIONS.md` 为准
- [x] **LAB11 姿态审计复现**（2026-11-18 完成）：用户 `scratch/lab11-repro/posture_audit.csv` 与 canonical 逐字节一致；80 次 rollout 三类 39/10/31，collapse 全在 +y；四道预测仅 Q1 的 survived 命中，判分见 `docs/learning/LAB11-prediction.md`
- [x] **LAB12 恢复实验**（2026-11-19 完成，`runs/20261119-001-lab12-recovery`）：6 格 × 5 种子 = 30 次，`--ignore-terminations` 跑推后 3 s → **2 稳 / 5 摔后恢复 / 23 倒地**（22 次 fall 命中）；31 次 upright_trip 里 74% 是延迟倒地。+x 600 N 2 稳 + 2 倒 + 1 临界，其余三方向 600 N 全倒；300 N 时 −x 4/5 恢复 vs −y 4/5 倒地 → “被判罚快”≠“站不住”（D22）。合并后 80 次扰动最终 46 没倒 / 33 倒。
- [x] **LAB13 冲量中点扫描 + LAB13b 真中间档补测**（2026-11-19，`runs/20261119-002-lab13-mid-impulse`、`runs/20261119-001-lab13b-true-mid`）：首轮实际 40/45 N·s（−x 全恢复、−y 倒 3/5）；补测 80/90 N·s（400/450 N）20 次全倒 → 六点 −x = 0/0/1/5/5/5、−y = 0/3/4/5/5/5（冲量 40/45/60/80/90/120），**单调**；5/5 从 80 N·s 起，边界夹在 60–80 N·s。单位口径与旧“非单调”更正见 D24。
- [x] **LAB14 重复实验测噪声底**（2026-11-19 完成，`runs/20261119-001-lab14-replicate`）：−y 60/80 N·s × 种子 101–105 × 3 遍 = 30 次；倒地数 4/4/4 与 5/5/5，倒地种子集合与唯一 transient 种子（102）三遍完全不变 → **判定层重跑噪声 = 0**；数值层有位级抖动（骨盆最低 ±1.1 cm、末尾 root ±3 cm、最长偏离 ±0.34 s）→ 20% 差别是种子抽样而非重跑噪声，要收紧只能加种子（D25）。用户预测判分 3.0/5（`docs/learning/LAB14-prediction.md`）
- [x] **LAB15 边界两格 20 种子**（2026-11-20 完成，`runs/20261120-001-lab15-boundary20`）：4 格 × 种子 101–120 = 80 次，39.8 min；60 N·s 时 −x 6/20（30%，12–54%）、−y **19/20**（95%，75–100%），Fisher p = 3.9e-5；80 N·s 两向 20/20（下界 0.83）。→ 两向边界不同（−x 过渡 60→80，−y 过渡 45→60），「5/5 从 80 起」作废（D26）；101–105 子集里 nx_j60 seed102 由 transient 翻成 down（D27）。用户预测判分 2.0/5
- [x] **LAB16 边界判定翻转率**（2026-11-20 完成，`runs/20261120-002-lab16-fliprate`）：`nx_j60` 三遍 6/6/5（翻转 4/20，20%）、`ny_j60` 19/18/18（1/20）；LAB15 的 6/20 修正为 5–6/20；区间端点 Fisher p = 1.0e-5 起，两向差异不撤（D28）。边界格此后默认 ≥3 遍
- [x] **LAB17 −y 过渡区**（2026-11-20 完成，`runs/20261120-003-lab17-y-transition`）：`ny_j45` 三遍 **11/11/12**（58%，CI 32–81%，翻转 3/20：102/109/115）、`ny_j50` **15/15/15**（75%，51–91%，0/20）→ −y 的 **50% 交点在 40–45 N·s 之间（插值 ≈44.3）**，过渡带改写为 40→50（D30）；过渡区中心翻转 3/20 高于 60 N·s 的 1/20 → 「边界格 ≥3 遍」扩到过渡区全部格。用户预测判分 2.2/5（`docs/learning/LAB17-prediction.md`）
- [x] **LAB18 −x 70 补点**（2026-11-21 完成，`runs/20261120-004-lab18-x70`）：`nx_j70` 三遍 **14/11/10**（中位 55%，遍包络 [27%,88%]，翻转 4/20；101–105 子集 5/5→3/5→3/5）→ **−x 的 50% 交点在 65–70 N·s 之间（中位插值 ≈68）**，与 −y 的 ≈44 相差 ≈23–24 N·s（D31）；4 个翻转全部是 rep1 down→后两遍 transient，记为首遍效应待验证。用户预测判分 2.5/5（`docs/learning/LAB18-prediction.md`）
- [x] **LAB19 四向对称性**（2026-11-21 完成，`runs/20261121-001-lab19-symmetry`）：+x/+y 的 60/80 各 20 种子 × 3 遍 = 240 次（2 h 4 min）。60 N·s：+x **0/20（20/20 stable）**、−x 5–6/20、−y 18–19/20、+y **20/20**；80 N·s：+x **0/20**（1 stable + 19 transient）、其余 20/20；四格翻转 0/20（合计 0/80）→ 边界顺序 **+y < −y < −x < +x**，「80 起全倒」对 +x 作废（D32）。用户预测判分 2.2/5
- [x] **LAB19b +x 100 N·s 补点**（2026-11-21 完成，`runs/20261121-002-lab19b-px100`）：`px_j100`（500 N）× 种子 101–120 × 3 遍 = 60 次（37 min 23 s）。三遍 k = **0/0/0**（0/20，CP [0%,16.8%]），stable 8/8/7、transient 12/12/13；二进制翻转 0/20、三分类翻转 **6/20**（全 S↔T）→ **+x 的 down 边界 >100 N·s，落在 100–120**（120 = 3/5，CI 14.7–94.7%；50% 交点只能写「约 110–120」）；stable 份额非单调（60 20/20 → 80 1/20 → 100 7–8/20）记观察（D33）

### 后续 LAB 路线（LAB17–LAB22，2026-11-20 拟定）

教学约定：一次只推进一个 lab；讲义写在 `docs/learning/LABxx-*.md`，用户亲手操作，智能体只写讲义、验收、纠错。

- [x] **LAB17 −y 过渡区**（2026-11-20 完成，`runs/20261120-003-lab17-y-transition`）：`ny_j45` 三遍 11/11/12（58%，[32%,81%]，翻转 3/20）、`ny_j50` 15/15/15（75%，[51%,91%]，0/20）→ **−y 的 50% 交点在 40–45 N·s 之间（线性插值 ≈44.3）**，D26 的「过渡 45→60」收窄为 40→50（D30）；LAB13 的「45 = 3/5」在 101–105 子集复现。讲义 `docs/learning/LAB17-y-transition.md`。
- [x] **LAB18 −x 70 补点**（2026-11-21 完成，`runs/20261120-004-lab18-x70`）：`nx_j70` 三遍 14/11/10（中位 55%，[27%,88%]，翻转 4/20）→ 交点在 65–70 N·s（中位插值 ≈68），与 −y 的 ≈44 相差 ≈23–24 N·s（D31）；4 个翻转全部是 rep1 down→后两遍 transient。原始设计：目标：−x 的过渡带从「60→80」两点变成 60/70/80 三点，估斜率与 50% 交点（两点插值 ≈66 N·s），并与 −y 的 ≈44 N·s 同口径对照；顺带检验「过渡区中心更不稳」是否也适用 −x。讲义 `docs/learning/LAB18-x70.md`、预测 `docs/learning/LAB18-prediction.md`（用户亲手跑，智能体只验收）。
- [x] **LAB19 方向对称性**（2026-11-21 完成，`runs/20261121-001-lab19-symmetry`）：四向 60/80 共 240 次，边界顺序 +y < −y < −x < +x，+x 在 80 仍 0/20（D32）。原始设计：`px_j60/px_j80`、`py_j60/py_j80` 各 20 种子 × 3 遍 = 240 次 ≈2 h。目标：把「−x 与 −y 边界不同」扩成四向对照，回答「推的方向 vs 推的大小」哪个主导。一次跑满 3 遍（用户 2026-11-21 要求，不探路）。讲义 `docs/learning/LAB19-symmetry.md`、预测 `docs/learning/LAB19-prediction.md`。
- [x] **LAB19b +x 100 N·s 补点**（2026-11-21 完成，`runs/20261121-002-lab19b-px100`）：补 `px_j100` 一格，60 次 37 min 23 s。结果：k = 0/0/0，+x 的 down 边界收窄到 **100–120**（>100），四向排序不变（D33）。原始设计：把 +x 的 down 边界从「80–120」缩成「80–100」或「100–120」，判断 +x 是陡转还是缓升；100 是唯一能把 80–120 对半切的点。讲义 `docs/learning/LAB19b-px100.md`、预测 `docs/learning/LAB19b-prediction.md`。
- [x] **LAB20 训练期 DR（2026-11-22 完成，负结果，D34）**：讲义 docs/learning/LAB20-dr.md + 预测 docs/learning/LAB20-prediction.md。已核实 mjlab tracking 默认就开着三项 startup DR（摩擦 0.3–1.2 / 质心 ±2.5,±5,±5 cm / 编码器 ±0.01 rad），所以本 lab = 把 DR 调强到 0.2–1.4 / ±4,±6,±6 / ±0.02，其余与 M4 一致。工具：scripts/lab20_dr_train.py（训练）、scripts/lab20_eval.sh + scripts/lab20_report.py（2 策略 × 3 窗口 × 10 种子）。预算 ≈ 2.7 h（smoke 5 min + 训练 1.9 h + 评估 40 min）。判据：heldout 是否提升、unseen 是否跟着提升、train 掉多少。  **实测：训练 1:35:01；train 0.999 / heldout 0.100 → 0.018（10/10 更差）/ unseen 0.025 → 0.024。判分 1.9/5。**
- [x] **LAB20b 加强 DR ckpt 的抗推对比（2026-11-22 完成，`runs/20261122-001-lab20b-perturb`，D35）**：把 LAB12 协议（60 N·s 两向 × 20 种子 101–120 × 3 遍 = 120 次，**1 h 13 min 31 s**）搬到 LAB20 的 `model_52998.pt`（sha e3747012…25b287），只换 ckpt。实测：**-x 三遍 k = 0/0/0**（M4 6/20；翻转 0/20；Fisher p = 0.0202；6 个失败种子全部转站住）；**-y 三遍 k = 20/19/20**（M4 19/20；翻转 1/20；p = 1；seed102 系统性变差）。机制：-x 的 20/20 全是 transient、骨盆从未触地（最低 0.447–0.466 m），但末尾 root 误差中位 92.9 cm（M4 非倒地 18.7 cm）→ DR 的收益是「退让式不倒」，不是「不丢跟踪」，且只在 -x 有效（D35）。工具 `scripts/lab20b_perturb.sh` / `scripts/lab20b_compare.py`；讲义 `docs/learning/LAB20b-robust.md`、预测判分 2.5/8 ≈ 1.6/5。
- [x] **LAB21 观测加世界系根位置/速度（2026-11-25 完成，runs/20261125-003-lab21-world-obs，D42）**：只改 actor 观测——末尾追加 root_pos_w（机器人根世界坐标，scale 0.5、噪声 ±0.1 m）与 root_vel_w（世界系线速度，scale 0.5、噪声 ±0.1 m/s），actor 160 → 166 维；critic / 奖励 / 终止 / DR / 超参 / 迭代数 / 种子全同 M4。训练：热启动 model_49999.pt（旧格式 ckpt 自动迁移并按 160→166 扩展第一层与优化器动量）、3000 iter / 4096 envs、目标终点 52998，**可中断续训**（重跑同一条命令）。评估：3 窗口（S3 0 s / S3 120 s / S2 134 s）× 10 种子（101–110）= 30 次，基线直接用 LAB20 已跑好的 M4（`runs/20261121-003-lab20-dr/eval/m4`），不重跑。判据（先定）：held-out 完成度 Δ ≥ +0.05 且 ≥7/10 同向、train 不掉 = **A 有效**；held-out 涨但 unseen 不动 = **B 只换记忆方式**；|Δ| < 0.01 或变差 = **C 假设被否**。成本：训练 ≈1 h 35 + 评估 ≈20 min ≈ **2 h**。工具 `scripts/obs_world.py`（观测项定义，训练/评估共用）、`scripts/lab21_obs_train.py`（训练+续训）、`scripts/lab21_eval.sh`（评估+出表）、`scripts/lab21_report.py`（配对表+图）、`scripts/lab21_push.sh`；`scripts/eval_tracking.py` 新增 `--obs-world-root` 开关（默认行为不变）。讲义 `docs/learning/LAB21-world-root-obs.md`、预测 `LAB21-prediction.md`。**实测：训练 1 h 44、评估 18.8 min；train 0.999 / held-out 0.100 → 0.018（存活 1.98 s → 0.34 s，61 次 ee_body_pos 终止、非摔倒）/ unseen 0.025 → 0.038（误差变差）；判分 4.5/8 ≈ 2.8/5。机制：位置三维权重范数 0.12/0.12/0.22（老维度中位 1.68）、速度 1.14/0.92/1.47；新维度归一化 std 4.50（数据 0.31/0.58），count 继承 5.21e9。**
- [x] **A1 判定口径审计（2026-11-22 完成，`runs/20261122-002-transient-audit`，D36）**：把已有 recovery.json 回算「偏离时长 + 末尾 root 误差」，把 transient 分擦边型/退让型。实测：+x 80 的 57 次 transient 全是擦边（偏离 0.2 s、root 4.5 cm）→ D32 维持；DR 的 −x 60 是 **60/60 全退让**（root 93.4 cm vs M4 18.2 cm）→ D35 补量化；+x 100 为中间态（20.7 cm）。工具 `scripts/lab_transient_audit.py` + `scripts/lab_transient_audit_plot.py`。
- [x] **LAB22 参考动作脚底接触/穿地校验（2026-11-25 完成，runs/20261125-002-lab22-foot-contact，D41）**：M2 遗留（PLAN M2 未勾项）。用 G1 MJCF 对 4 段 npz 逐帧 FK（脚底 7 块碰撞 box 的最低角点 z），输出穿地深度（max/p99）、接触比例（任一足/双足/腾空）、最长悬空、骨盆高度；Part B 用已有 metrics.csv 的 pelvis_z 与 ref_pelvis_z 算策略骨盆 offset。判据：穿地 max ≤20 mm 且 p99 ≤20 mm = OK、20–50 = WARN、>50 = FAIL。成本：CPU ≈82 s + <1 s，零 GPU。工具 `scripts/lab22_foot_audit.py` + `scripts/lab22_policy_offset.py`；讲义 `docs/learning/LAB22-foot-contact.md`、预测 `LAB22-prediction.md`。**实测**：4 段里 3 段 FAIL——s3_000s（训练）**59.6 mm**（p99 38.6、>20 mm 帧 46/999、接触仅 **34%**、双足 13%、最长悬空 **3.82 s**、脚底 p50 **+2.8 cm**）、s2_134s 59.0 mm、dance1 53.3 mm（但接触 90%、p50 +0.8 cm）；只有 held-out 段 s3_120s 合格（16.7 mm）。Part B：M4 骨盆比参考**低 1.70–1.76 cm**（均值 −1.62，|Δ|>2 cm 占 47–48%）→ 策略「往下够地面」而非悬空。判分 1.5/8 ≈ 0.9/5。结论：root 误差 3.90 cm 含 ≥1.6 cm 参考不可行成分，完成度 0.999 只能读作「没摔、跑完」（D41）。
- [x] **LAB23 DR 单档消融（2026-11-23 完成）**：把 LAB20 的三档加强拆成三个单因素臂（`fric` 摩擦 0.2–1.4 / `com` 质心 ±4/6/6 cm / `bias` 编码器 ±0.02 rad，其余两项回默认），训练口径同 M4/LAB20（热启动 model_49999、3000 iter、4096 envs、seed 0）。评估按 D35/D36 的三数口径：nx_j60 × 20 种子 × 3 遍 + ny_j60 × 1 遍 + held-out S3 120 s × 10 种子。工具 `scripts/lab23_ablate.sh`（run/report/show，可中断续跑）+ `scripts/lab23_report.py`。成本：3 臂 ≈7 h 30 min（训练 4 h 45 + 扰动 2 h 27 + held-out 21 min）；两臂版 ≈5 h。讲义 `docs/learning/LAB23-dr-ablation.md`、预测 `LAB23-prediction.md`。目标：回答「哪一档造成 held-out 崩塌」「哪一档换来退让式不倒地」。
- [x] **LAB24 摩擦档强度扫描（2026-11-24，D38）**：只沿摩擦一档加深——f14 = 0.2–1.4（复用 LAB23 fric ckpt，不重训）、f20 = 0.2–2.0、f30 = 0.2–3.0；com/bias 回 mjlab 默认。训练口径同 M4/LAB20/LAB23；评估 nx_j60 × 20 种子 × 3 遍。**实测（runs/20261123-001-lab24-fric-scan）：非单调——f14 2/2/1（5/60，8%）、f20 10/14/15（39/60，65%）、f30 0/0/0（0/60）**；非倒地 root 中位 f14 82.3 / f20 68.7 / f30 **18.7 cm**（≈M4 18.2，p75 38.6、max 61.8）→ 只有 3.0 档同时满足 D37 判据，但仍 21/60 漂 ≥30 cm、单 seed 三遍极差 49.5 cm。f14 复跑 5/60 vs LAB23 6/60 = 又一次 D28 翻转（seed119）；f20 逐遍 10→14→15 → 过渡区 3 遍不够。工具 `scripts/lab24_fric_scan.sh`（含断点续训，f30 实跑中用过）+ `scripts/lab24_report.py`；讲义 `docs/learning/LAB24-fric-scan.md`、预测 `LAB24-prediction.md`（判分 2.5/8）。见 REPORT §2.5.16、D38。
- [x] **LAB25 摩擦上界收敛（2026-11-24 实测，D39）**：回答 D38 的两个未定项——(a) f20（0.2–2.0）的 65% 是否复现（重训训练 seed 1）；(b) f30（0.2–3.0）是否也改 −y（复用 LAB24 ckpt 测 ny_j60 / ny_j45，零训练）。四臂：f30ny60 / f30ny45（eval-only）+ f20s1 / f30s1（训练 seed 1，3000 iter、4096 envs、热启动 model_49999）→ nx_j60 × 20 种子 × 3 遍 = 240 次评估。工具 `scripts/lab25_dr_seed.sh`（run/train/eval/report/status，可断点续训；f30 ckpt 用 sha 1119f986…455ba0 校验）+ `scripts/lab25_report.py`（跨训练 seed 表 + −y 表 + 双面板图）。判定线：f20 seed1 每遍 ≥9/20 = 同向、≤4/20 = seed0 孤立；f30 seed1 ≤2/20 且 root 中位 ≤40 = 复现；ny_j60 ≤10/20 = 也改了 −y。成本 ≈5 h 50（训练 3 h 20 + 评估 2 h 25）。讲义 `docs/learning/LAB25-converge.md`、预测 `LAB25-prediction.md`。
  **实测**（`runs/20261124-001-lab25-converge`）：f20 seed1 **5/5/4**（14/60，23%，p=1.0 vs M4；root 71.4、非倒地全漂移）
  → 65% 属 seed0 离群；f30 seed1 **1/3/3**（7/60，12%，p=0.09–0.70）→ 0/60 只部分复现；f30 的 −y：ny_j60 **20/20/20**、
  ny_j45 **0/0/0**（M4 34/60，p<1.5e-4；49/60 退让、root 87.2）→ 「−y 无收益」只在 60 N·s 饱和点成立。
  实测（`runs/20261122-003-lab23-dr-ablation`，三臂训练 1 h 40–41 min/臂、评估 230 次 ≈2 h 15 min）：nx_j60 逐遍 fric **2/2/2**（10%，池化 p=0.019）、com 13/15/15（72%）、bias 19/19/19（95%），M4 6/6/5、LAB20 0/0/0；fric 的 54/54 非倒地全是**漂移型**（root 中位 81.2 cm）、com 17/17 漂移、bias 2 漂移 1 退让；三臂 held-out **全 0.018** → 崩塌与档位无关（D37）。下一步：摩擦档强度扫描。

- [x] **LAB26 −y 50/55 N·s 补点（2026-11-25 完成，D40）**：补齐 f30（摩擦 0.2–3.0，复用 LAB24 ckpt sha 1119f986…455ba0）在 −y 的过渡区——ny_j50（0,−250,0）与 ny_j55（0,−275,0），20 种子 101–120 × 3 遍 = 120 次 ≈1 h 10，零训练。目标：把 LAB25 的端点（45 = 0/60、60 = 60/60）变成 4 点曲线，定位 3.0 档的 −y 50% 交点并给出非倒地跟踪代价（root 中位 / 退让占比）。工具 `scripts/lab26_ymid.sh`（run/report/status/show，可断点续跑，启动校验 ckpt sha）+ `scripts/lab26_report.py`（表 1 曲线与交点插值、表 2 root/退让/漂移、逐遍 Fisher）。讲义 `docs/learning/LAB26-y-f30-mid.md`、预测 `docs/learning/LAB26-prediction.md`（本次未作答，直接给参考回答）。  **实测（runs/20261125-001-lab26-y-f30-mid，63.0 min）：ny_j50 5/5/4（14/60，23%，CP 13.4–36.0%，vs M4 75% p=0.0039/0.0039/0.0012）、ny_j55 20/19/19（58/60，97%，CP 88.5–99.6%）→ 交点 M4 ≈44.4 → f30 ≈51.8（+7.4 N·s）**；非倒地 root 中位 87.2 / 110.8 / 120.8 cm（45/50/55），退让占比 82% → 100%；ny_j50 翻转 5/20、ny_j55 1/20。→ 3.0 档是「延迟 7 N·s」而非「抬高天花板」（55 已 97%、60 仍 100%）。见 D40。

- [x] **LAB27 多片段联合训练（2026-11-25 讲义与工具就绪，2026-11-26 实测完成，D43；`docs/learning/LAB27-multiclip.md`）**：训练从 1 段变 2 段——S3 0–20 s（原样）+ dance1 0–20 s（新片段，先做地面高度校正过 D41 闸），其余与 M4 完全一致（热启动 model_49999、3000 iter、4096 envs、seed 0、目标 52998、DR 默认、160 维观测），回答「单片段训练是不是 held-out 0.100 的成因」。三步：Part A `scripts/lab27_ground_fix.py` 对 dance1 的 0–20 s 与 60–80 s 做「接触帧地面钳制 + 腾空帧线性插值」（实测 before 32.9/9.9 mm → after 3.5/5.9 mm，p99 0.8/2.5 mm，闸门 PASS）；Part B `scripts/lab27_multiclip_train.py` 把两段打成 packed 目录（motion_ends = [999 1999]，因为 mjlab 启动校验只接受单个存在路径，list 会被 ValueError 拦下）再训练 ≈1 h 45，支持中断续训；Part C `scripts/lab27_eval.sh` + `scripts/lab27_report.py` 跑 4 窗口（train S3 0 s / heldout S3 120 s / unseen S2 134 s / dance dance1 60–80 s）× 10 种子 × 2 策略 = 80 次 ≈50 min。判据：(A) held-out ≥0.15 且 ≥7/10 同向 → 数据是瓶颈之一；(B) 只涨训练片段 → 只会多记一段；(C) held-out <0.05 → 异质片段损害专精。预测 `docs/learning/LAB27-prediction.md`。成本 ≈2 h 40。
  **实测**（runs/20261125-004-lab27-multiclip，训练 1 h 43 min 47 s、评估 80 次 ≈50 min）：train 0.999 → 0.999（但 MPKPE 5.65 → **7.51 cm**、root 3.86 → **6.04 cm**）；held-out 0.099 → **0.097**（Δ −0.003，10/10 平）；unseen 0.025 → **0.026**；dance1 60–80 s 0.061 → **0.108**（Δ +0.046，9/10 更好）→ **属 (B) 只涨训练动作家族**，代价是训练片段精度。地面校正 32.9/9.9 → 3.5/5.9 mm 均过闸。下一步 LAB28（同动作距离曲线）。
- [ ] **LAB28 窗口距离曲线（2026-11-26 讲义与工具就绪）**：零训练，只换被评估的动作窗口——dance1 的 0/20/40/60/80/100 s 各 20 s（6 段互不重叠，必须含 0 s 作为「距离 0」上限锚点；4 段新生成都要过 D41 闸 max ≤20 mm 且 p99 ≤20 mm），用 M4 与 LAB27 两个 ckpt × 10 种子（101–110）= 120 次 ≈75 min，回答「记忆的作用半径」：完成度随「距训练窗口的时间距离」怎么衰减。判据：(A) 单调衰减且远端收敛到 M4 水平 → 有半径（加数据要沿时间铺满）；(B) 只有 0 s 高、20 s 就掉到 M4 → 纯局部记忆（改方法，别再加段）；(C) 平台/不单调 → 距离不是主因（先查参考/相位）。工具 `scripts/lab28_window_curve.sh`（build/run/report/show；按 summary.json 跳过、拒绝 RUN=runs、校验 LAB27 ckpt sha 8745f96e…125dac）+ `scripts/lab28_report.py`（两张表 + 逐种子配对 + 自动描述曲线形状 + 双面板图）。讲义 `docs/learning/LAB28-window-curve.md`、预测 `docs/learning/LAB28-prediction.md`。成本 ≈80 min。

顺序理由：先把评估类（17–19）做完闭环扰动结论，再动训练侧（20–21，贵且会换 ckpt），最后补 M2 的验证缺口（22）。
