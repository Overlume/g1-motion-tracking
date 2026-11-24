# ARTIFACT_INDEX

| 路径 | 内容 | 来源/版本 | 大小 | 备注 |
|---|---|---|---|---|
| `repos/mjlab_research/` | mjlab v1.6.0 源码 | github.com/mujocolab/mjlab | ~95MB | 只读，不改其约定 |
| `repos/supermimic/` | SuperMimic 源码 | HybridRobotics/supermimic @abc124c1 | ~1.3MB | 无数据/检查点/配置 |
| `scratch/supermimic.pdf` | 论文 PDF | arXiv 2610.28480v2 | 3.5MB | 未提取正文 |
| `scratch/demo_motion.npz` | mjlab demo 动作（旧格式） | GCS mjlab_beta | 11.8MB | fps=50，6574 帧，29 关节，30 body |
| `environment/.venv/` | 项目内 Python 3.13 环境 | uv | 待测 | 见 `scripts/setup_env.sh` |

## 运行记录

- runs/20261111-001-smoke/README.md — G1 tracking 环境 GPU 冒烟（8/64/512 envs，显存与耗时）；日志 environment/logs/smoke_*env*.log。

## 脚本

- scripts/setup_env.sh — 项目内 uv + venv 安装。
- scripts/convert_demo_motion.py — GCS 旧格式 npz → v1.6.0 MotionLoader 格式。
- scripts/smoke_env.py — 最小 GPU 冒烟（NUM_ENVS/STEPS/MOTION 环境变量）。
- scripts/smoke_mem.sh — 带 nvidia-smi 采样的冒烟封装。

- runs/20261111-002-replay/README.md — 官方预训练 checkpoint 回放 + 视频；产物 outputs/videos/model_49999/（mp4/gif/frames）；日志 environment/logs/replay_demo.log。
- scripts/replay_policy.py — 加载官方 checkpoint 回放并录 mp4（无 viewer）。
- scripts/extract_frames.sh — 从 mp4 按时间抽帧，用于人工核对。
- runs/20261112-001-eval/README.md — 量化跟踪评估（MPKPE/R-MPKPE/关节误差/跌倒/倾角/脚滑），metrics.csv + summary.json。

## 2026-11-12

- runs/20261112-001-eval/ — 官方预训练策略 seed=0 量化评估（metrics.csv、summary.json、repro/summary.json、README）
- runs/20261112-002-train-smoke/ — 训练模式冒烟（512 envs，10 iter，model_0.pt / model_9.pt）
- runs/20261112-003-train-scale/ — num_envs 512/1024/4096 的耗时与显存扫描
- scripts/eval_tracking.py — 跟踪评估（MPKPE、R-MPKPE、关节 RMSE、跌倒、倾角、脚滑）
- scripts/smoke_train.sh — 训练冒烟 + nvidia-smi 显存采样
- scripts/eval_repro.sh — 同种子两次评估并对比
- outputs/videos/model_49999/960x540/ — 960x540 回放视频、gif、抽帧
- environment/logs/replay_960x540.log、eval_tracking.log、eval_tracking_repro.log、train_smoke_512env.log、train_scale_1024env.log、train_scale_4096env.log
- docs/learning/LAB01-first-run.md — 用户手做的第一个实验（回放、显存采样、改参数、读代码）

## 2026-11-13 阶段 2

- scripts/gmr_bvh_to_csv.py —— LAFAN1 BVH -> G1 pkl/csv（GMR，支持 --start-frame/--max-frames）
- scripts/csv_to_npz_local.py —— CSV -> mjlab npz（复用 mjlab MotionLoader，本地保存，不传 W&B）
- data/raw/lafan1/ —— LAFAN1 zip + 解压的 77 个 bvh（CC BY-NC-ND 4.0，勿公开再分发）
- data/processed/gmr/ —— 重定向中间产物（pkl/csv）
- data/processed/fight1_s2_134s_20s.npz —— fight1_subject2 的 134.6-154.6 s 片段（600 帧 30fps -> 999 帧 50fps）
- runs/20261113-001-retarget/ —— 本次重定向与回放的实验记录
- docs/learning/LAB02-retarget-own-clip.md —— 用户手做：fight1_subject3 重定向 + 转换 + 评估
- runs/20261113-002-eval-fight/README.md —— fight 片段量化评估（与 dance 对照表）


## 2026-11-13 16:20 新增（诊断与修复）

- scripts/replay_policy.py —— 重写：1 env 默认、逐帧 RESET 日志、episode_log.json、--ignore-terminations
- scripts/eval_tracking.py —— 重写 v2：分段统计、reset 记录、motion_frame_furthest、drift_rate_cm_s
- runs/20261113-004-eval-s3-v2/ —— S3（termination 开）：3 次 reset、completion 0.5295
- runs/20261113-005-eval-s3-cont/ —— S3（关 termination）：13.92 s 摔倒
- runs/20261113-006-eval-s2-v2/ —— S2（termination 开）：每 12.5 步 reset 一次（0.24 s 循环）
- runs/20261113-007-eval-dance1-v2/ —— dance1（v2 协议）：completion 0.9998、root_err 9.78 cm
- runs/20261113-008-continuous-render/ —— 修复后的连续录像（S2/S3，0 reset）
- runs/20261113-009-eval-s2-cont/ —— S2（关 termination）：4.28 s 摔倒、root_err 末值 289.7 cm
- outputs/videos/model_49999/960x540/fight_s2_cont/、fight_s3_cont/ —— 连续录像 + contact_2fps.png + episode_log.json
- docs/learning/LAB03-verify-no-reset.md —— 用户手做：验证 reset 已记录、录像不再重复


## 2026-11-14 新增（LAB03 复盘与可复现性）

- scripts/motion_stats.py —— npz 运动学统计（路径长度、净位移、平均/最大速度、x/y 范围、逐秒分解）
- docs/learning/LAB03-verify-no-reset.md —— 已补“我的记录”：用户 6 问作答 + 批改 + 新发现
- runs/20261113-010/011-eval-s3-repro/ —— 同 seed=0、DR 开，跑两次：fall None / 781
- runs/20261113-012/013-eval-s3-nodr/ —— 关 DR，跑两次：fall 424 / 380
- runs/20261113-014/015-eval-dance1-nodr/ —— dance1 1500 步 ×2：root_err 7.730 / 7.693（差 0.5%）
- outputs/videos/model_49999/640x360/lab3_fourenv/frame0.png —— 4 env 场景下开头只有 3 台入镜（相机取景）

## 2026-11-14（取景修复 + S3 种子分布）

- scripts/camera_fit.py：env 网格布局复现 + 自适应自由相机（v2 公式）
- scripts/replay_policy.py：新增 --camera(follow/fit/auto)、--camera-margin、--camera-elevation、--camera-azimuth、--camera-min-distance；默认 follow
- outputs/videos/model_49999/960x540/cam_default_dance1/：follow 相机 + 4 env 取景验证（dance1，400 步，frames/t0.2,t4.0.png + episode_log.json）
- outputs/videos/model_49999/960x540/cam_follow_final/：S3 摔倒场景下的取景限制证据（靠近相机的一台会被下缘切掉）
- runs/20261114-001-s3-seed101 … 005-s3-seed105：S3 五种子摔倒分布（各含 summary.json）
- runs/20261114-001-s3-seed101/README.md：五种子汇总表与命令
- docs/COMPLETION_CRITERIA.md：项目结束边界
- docs/GITHUB_UPLOAD_PLAN.md：GitHub 上传清单与提交计划
- environment/requirements.txt：uv pip freeze（142 行）

## 2026-11-14 15:20 交接（历史）

- `docs/handoffs/HANDOFF.md`（仅本地，`.gitignore` 已排除，不随仓库分发）— **唯一**交接文档；2026-11-20 把 `20261114-1520` / `20261118-2055` / `20261119-1500` / `20261120-1120` 四份旧交接合并成这一份并删除旧文件（D29）
- outputs/videos/model_49999/960x540/cam_default_dance1/：相机定稿证据（follow，4 env）

## 2026-11-17

- runs/20261117-001-m4-w15-finetune/ — M4 规模训练（4096 env × 3000 iter，weight 1.5，model_52998.pt，真实墙钟 6412 s）
- runs/20261117-002-m4-eval/ — 300 iter vs 3000 iter 评估（5 种子，comparison.md/sweep.png）
- runs/20261117-003-generalization-probe/ — 泛化探针（n=1：held-out S3 窗口 0.101、未见 S2 0.024、官方对照 0.016）
- data/processed/fight1_s3_120s_20s.npz — S3 片段 120–140 s 的 held-out 窗口（999 帧 @50 fps，sha256 4aac9984…），由 `data/raw/lafan1/bvh/fight1_subject3.bvh --start-frame 3600 --max-frames 600` 经冻结流水线生成；中间产物 `data/processed/gmr/fight1_s3_120s_20s.pkl|.csv`。LAFAN1 CC BY-NC-ND，**不公开分发**。
- runs/20261117-004-lab09-generalization/ — 泛化探针正式版（6 组合 × 5 种子，generalization.md/json）
- runs/20261117-005-m5-perturb/ — M5 受控扰动（12 格 × 5 种子，m5.md/json/png）
- configs/perturb_protocol_v1.json — M5 扰动协议 v1
- outputs/figures/{m5_perturbation,weight_sweep_lab07,m4_scale_300_vs_3000}.png + evidence/ — 报告用图与证据
- docs/REPORT.md — 项目报告 v1.0
- LICENSE — 纯 MIT；THIRD_PARTY_NOTICES.md — 第三方组件与数据许可
- runs/20261118-001-m5-threshold-refine/ — M5 阈值细化（+y × 4 档 × 5 种子，m5.md/json、README.md）
- scripts/m5_combined_plot.py — 合并多 run 的扰动结果出图
- runs/20261118-002-m5-video-check/ — M5 终止格视频核对（4 例，含复现一致性表与结论）
- outputs/figures/evidence/m5_fall_check_*.jpg — 跪坐倒地（3 例）与存活对照（1 例）的抽帧拼图
- runs/20261118-003-posture-audit/ — 零 GPU 姿态审计：80 次 rollout 三分法（39/10/31）、posture_audit.csv 逐次明细
- scripts/m5_posture_audit.py — 姿态审计脚本（复位前 0.4 s 判据 + 4 例视频自检）
- docs/learning/LAB11-posture-audit.md — LAB11 讲义（终止 ≠ 倒地，零 GPU 统计）
- scripts/lab12_recovery.sh — LAB12 恢复实验（run/report/show；6 格 × 5 种子，--ignore-terminations，推后 3 s）
- scripts/lab12_recovery_report.py — LAB12 判分器（stable / transient / down，输出 recovery.md/json）
- docs/learning/LAB12-recovery.md — LAB12 讲义（缺口、判据、四道预测题、读数顺序、坑表）
- runs/20261119-001-lab12-recovery/ — LAB12 恢复实验（6 格 × 5 种子，termination 关，推后 3 s；2 稳 / 5 恢复 / 23 倒地）
- scripts/lab12_plot.py — LAB12 分类堆叠图（outputs/figures/lab12_recovery.png）
- outputs/figures/lab12_recovery.png — 报告用图：每格 5 种子的结局构成
- docs/learning/LAB12-prediction.md — 用户四题 + 实测答案 + 判分（0.5/5）与错误模式
- scripts/lab13_mid_impulse.sh — LAB13 薄封装（LAB12 评估+判分，tag lab13-mid-impulse，格 nx_j40/nx_j45/ny_j40/ny_j45）
- docs/learning/LAB13-mid-impulse.md — LAB13 讲义（为什么选 400/450、5 道预测题、读数顺序、坑表）
- runs/20261119-002-lab13-mid-impulse/ — LAB13 冲量中点扫描（−x/−y × 400/450 N·s × 5 种子；证明非单调，D23）
- docs/learning/LAB13-prediction.md — 用户五题 + 判分（1.5/5）与“默认单调”错误模式
- docs/handoffs/HANDOFF.md — 交接（唯一文件；含 LAB16 结论与 D28，2026-11-20 合并）


## 2026-11-19（LAB14 就绪）

- docs/learning/LAB14-replication.md — LAB14 讲义（同配置重跑 3 遍测噪声底；5 道预测题、读数顺序、判读规则、通过标准）
- docs/learning/LAB14-prediction.md — LAB14 预测模板（待用户填写）
- scripts/lab14_replicate.sh — LAB14 重复实验（run/report/show；每遍写 <run>/rep<N>/，复用 LAB12 评估与判分）
- scripts/lab14_compare.py — LAB14 汇总比较（每格×每遍计数、逐种子矩阵、噪声判定、与 LAB13 基线对照）

## 2026-11-24（LAB24 摩擦档扫描）

- `runs/20261123-001-lab24-fric-scan/` — 三臂（f14 复用 LAB23 fric / f20 0.2–2.0 / f30 0.2–3.0）× nx_j60 20 种子 × 3 遍 = 180 次评估；README + `lab24_scan.md`（三数表、三型、Fisher p）+ `lab24_scan.json`；162 MB。
- `outputs/figures/lab24_fric_scan.png` — 倒地率 / 非倒地 root 中位 / root-vs-偏离散点（三面板）。
- `scripts/lab24_fric_scan.sh`（run/train/eval/report/status，断点续训）、`scripts/lab24_report.py`（三臂三数 + 基线对比 + 图）。
- 结论：摩擦上界非单调（8% → 65% → 0%）；f30 同时满足 D37 判据（k 0/60、root 中位 18.7）但仍 21/60 漂 ≥30 cm。REPORT §2.5.16、D38。

## 2026-11-23（LAB23 DR 单档消融）

- runs/20261122-003-lab23-dr-ablation/ — 三臂（fric/com/bias）× nx_j60 20 种子 × 3 遍 + ny_j60 × 1 遍 + held-out 10 种子；
  lab23_ablation.md/json（三数表 + Fisher + 逐种子配对）、arm.log、train.log（0.9 GB/臂，不入库）
- outputs/figures/lab23_dr_ablation.png — 倒地率 / 非倒地 root 误差 / held-out 三面板
- scripts/lab23_ablate.sh — run/report/show；每臂训练 + 扰动 + held-out，幂等可续跑
- scripts/lab23_report.py — 三数口径汇总（k、root 中位、擦边/漂移/退让计数）+ Fisher 逐遍 p + 基线对照（M4/LAB20）
- docs/DECISIONS.md — D37（逐档报三数；held-out 崩塌与档位无关；三型扩展）；D36 补注
- docs/REPORT.md — §2.5.15 + §3 第 11 条 + §4 第 4/9/10 条改写 + 第 11 条新增
- docs/learning/LAB23-dr-ablation.md、docs/learning/LAB23-prediction.md（判分 1.5/8 ≈ 0.9/5）

## 2026-11-22（A1 判定口径审计）

- runs/20261122-002-transient-audit/ — 回算 LAB16/17/19/19b/20b 的 recovery.json：audit.md/json（每格 k、transient 偏离/root 中位、退让型/擦边型计数）+ README（方法、阈值理由、结论、局限、复现）
- outputs/figures/lab_transient_audit.png — 非倒地 rollout 的散点（x = 最长连续偏离 s，y = 末尾 root 误差 cm，虚线 = 退让判据 0.5 s / 30 cm）
- scripts/lab_transient_audit.py、scripts/lab_transient_audit_plot.py — 派生指标审计 + 配图；REGISTRY 里登记了 5 个 run 与 10 个格
- docs/DECISIONS.md — D36（transient 分两型；报鲁棒性必须三数齐全）
- docs/REPORT.md — §2.5.14 + §2.5.10/§2.5.13 补注 + §3 第 10 条 + §4 第 4 条改写 / 第 10 条扩写

## 2026-11-22（LAB20b 抗推对比结果）

- runs/20261122-001-lab20b-perturb/ — LAB20 权重 × LAB12 协议：2 格 × 20 种子 × 3 遍 = 120 次，**1 h 13 min 31 s**；
  -x 三遍 k = 0/0/0（p=0.0202，6 个失败种子配对转为站住）、-y = 20/19/20（p=1，seed102 系统性变差）；机制见 D35（README + fliprate.md/json + vs_m4.md + rep<N>/）
- outputs/figures/lab20b_robust.png — 逐种子 × 逐遍判定热图（nx_j60 全 transient、ny_j60 全 down 只剩 seed116 一格）
- scripts/lab20b_perturb.sh、scripts/lab20b_compare.py — 三遍 + 判分 + 翻转 + 图；Fisher + 逐种子配对对比；scripts/lab16_report.py 新增 `--intro`
- docs/DECISIONS.md — D35（DR 收益 = 退让式不倒地；鲁棒性拆成失败率 + 扰动后误差）；D34 补注
- docs/REPORT.md — §2.5.13 + §3 第 9 条 + §4 第 9/10 条 + §7 第 1 条
- docs/learning/LAB20b-robust.md（讲义 §7 实测）、docs/learning/LAB20b-prediction.md（判分 2.5/8 ≈ 1.6/5）

## 2026-11-22（LAB20 训练期 DR 结果）

- runs/20261121-003-lab20-dr/ — 训练期 DR 加强（摩擦 0.2–1.4 / 质心 ±4,±6,±6 cm / 编码器 ±0.02 rad，其余同 M4）+ 三窗口配对评估；README §5 记录「ckpt 选错」事故
- runs/20261121-003-lab20-dr/eval/lab20_eval.md|json|png — M4 vs LAB20 × 3 窗口 × 10 种子（60 次）表、机读、分组柱图
- scripts/lab20_dr_train.py — 训练入口（改 cfg 后走 launch_training；--mode smoke|full、--default-dr 对照臂）
- scripts/lab20_eval.sh — 2 策略 × 3 窗口 × N 种子评估封装（dr ckpt 排除 pretrained/）
- scripts/lab20_report.py — 完成度配对表 + 图（--figure）
- docs/learning/LAB20-dr.md — LAB20 讲义（含 ckpt sha 核对与「pretrained 陷阱」FAQ）
- docs/learning/LAB20-prediction.md — 5 题 + 结果 + 判分 3/8（1.9/5）与参考回答
- docs/DECISIONS.md — D34：DR 加强负结果、held-out 降级为「窗口记忆」指标、评估必须核对 ckpt sha

## 2026-11-21（LAB19b 结果）

- `runs/20261121-002-lab19b-px100/`：+x 100 N·s（`px_j100` = 500 N × 0.2 s）× 种子 101–120 × 3 遍 = 60 次判定（37 min 23 s）；README + `run.log` + `fliprate.md/json`
- `outputs/figures/lab19b_px100.png`：单格 3 行 × 20 列判定矩阵（无 down，104/107/109/110/112/113 有 S↔T 条纹）
- `docs/REPORT.md` §2.5.11：+x 边界收窄到 100–120、stable 份额非单调、三分类翻转 6/20
- `docs/DECISIONS.md` D33：边界区间写法、不补 px_j110、新观察只作记录、down 与 S/T 两种翻转都要报
- `docs/learning/LAB19b-px100.md`（讲义 §7）、`docs/learning/LAB19b-prediction.md`（参考回答；本次未判分）
- `README.md` 结果表、`docs/STATE.md`、`docs/PLAN.md` 已同步

## 2026-11-21（LAB19 结果）

- `runs/20261121-001-lab19-symmetry/`：+x/+y 60/80 × 种子 101–120 × 3 遍 = 240 次判定（2 h 4 min）；README + `run.log` + `fliprate.md/json`
- `outputs/figures/lab19_symmetry.png`：四向 4 面板判定矩阵（三遍全同，无竖条纹）
- `docs/REPORT.md` §2.5.10：四向边界顺序 +y < −y < −x < +x
- `docs/DECISIONS.md` D32：四向排序、「80 起全倒」对 +x 作废、翻转 0/80 的对照
- `docs/learning/LAB19-symmetry.md`（讲义 §7）、`docs/learning/LAB19-prediction.md`（判分 2.2/5 + 参考回答）
- `README.md` 结果表、`docs/STATE.md`、`docs/PLAN.md` 已同步

## 2026-11-21（LAB18 结果）

- `runs/20261120-004-lab18-x70/`：−x 70 N·s × 种子 101–120 × 3 遍 = 60 次判定（三遍 14/11/10，翻转 4/20）；README + `fliprate.md/json`
- `outputs/figures/lab18_x70.png`：种子 × 遍判定矩阵（可见 102/104/112/113 的 rep1 竖条纹）
- `docs/REPORT.md` §2.5.9：−x 第二分点与交点 65–70（≈68）
- `docs/DECISIONS.md` D31：交点定位、两向差 ≈23–24、翻转单向化记入局限
- `docs/learning/LAB18-x70.md`（讲义）、`docs/learning/LAB18-prediction.md`（判分 2.5/5）
- 脚本沿用 `scripts/lab16_fliprate.sh` + `scripts/lab16_report.py`（--title / --figure-title）

## 2026-11-20（LAB18 就绪）

- docs/learning/LAB18-x70.md — LAB18 讲义（`nx_j70`，60 次 ≈32 min；为什么是 70 不是 55/65/75、三种结果的判读、分支预案）
- docs/learning/LAB18-prediction.md — LAB18 预测模板（5 题，待用户填写）

## 2026-11-20（LAB17 −y 过渡区）

- runs/20261120-003-lab17-y-transition/ — 2 格 × 20 种子 × 3 遍 = 120 次；`ny_j45` 三遍 11/11/12（58%，[32%,81%]，翻转 3/20：102/109/115）、`ny_j50` 15/15/15（75%，[51%,91%]，0/20）→ −y 的 50% 交点在 40–45 N·s（README + fliprate.md/json + rep1..3）
- outputs/figures/lab17_ythmids.png — 种子 × 遍判定矩阵（可见 `ny_j45` 的 102/109/115 竖条纹）
- docs/DECISIONS.md — D30：−y 50% 交点 40–45；过渡区中心翻转 3/20，把「≥3 遍」扩到过渡区全部格；40 N·s 的 0/5 必须带 5 种子上界
- docs/REPORT.md — §2.5.8 −y 过渡区；§2.5.6 结论 1、§2.5.4 更正块、§3 发现 5 补 LAB17 收窄
- docs/learning/LAB17-prediction.md — 判分 2.2/5
- docs/learning/LAB17-y-transition.md — 讲义（含「RUN 不能是 runs 本身」警告与 runs 根目录自检）
- scripts/lab12_recovery.sh、scripts/lab16_fliprate.sh — 加「拒绝 runs 根目录」护栏（LAB17 事故后）
- scripts/lab16_report.py — 新增 `--title` / `--figure-title`（图标题用 ASCII，避免中文字形缺失）

## 2026-11-20（LAB16 判定翻转率）

- runs/20261120-002-lab16-fliprate/ — 2 格 × 20 种子 × 3 遍 = 120 次判定（新跑 80 次，42.8 min）；`nx_j60` 翻转 4/20、`ny_j60` 1/20（README + fliprate.md/json + rep2/rep3 + run.log）
- scripts/lab16_fliprate.sh — LAB16 run/report/show（rep1 引用 LAB15；新跑 rep2/rep3，复用 LAB12 的评估与判分）
- scripts/lab16_report.py — 翻转率统计（每格×每遍、逐种子矩阵、二进制/三分类翻转 + Clopper–Pearson CI、判定规则、种子×遍矩阵图）
- docs/learning/LAB16-fliprate.md — LAB16 讲义（三档判定规则、读数顺序、通过标准、五个坑）
- docs/learning/LAB16-prediction.md — LAB16 预测模板（待用户填写）
- docs/DECISIONS.md — D28：边界格写区间 + 核心集；翻转分「判据擦边」与「rollout 双稳态」；边界格默认 ≥3 遍
- docs/REPORT.md — §2.5.7 边界判定翻转率
- outputs/figures/lab16_flipmatrix.png — 种子 × 遍判定矩阵

## 2026-11-20（LAB15 结果）

- runs/20261120-001-lab15-boundary20/ — 4 格 × 20 种子 = 80 次；60 N·s 时 −x 6/20 vs −y 19/20（p=3.9e-5）；80 档两向 20/20（README + boundary20.md/json + recovery）
- docs/DECISIONS.md — D26：两向边界不同（−y 过渡 45→60，−x 60→80）；D27：判据附近的判定会翻转，边界格 k/N 带 ±1 种子
- docs/REPORT.md — §2.5.6 新增；§2.5.4 结论 2/3 更正
- docs/learning/LAB15-prediction.md — 判分 2.0/5
- outputs/figures/m5_impulse_sweep.png — 60/80 换成 20 种子点

## 2026-11-20（LAB15 就绪 + lab13b 补跑）

- scripts/lab15_boundary20.sh — LAB15 薄封装（run/report/show；4 格 × 20 种子 = 80 次）
- scripts/lab15_report.py — LAB15 统计（k/N + Clopper–Pearson 95% CI + Fisher 精确检验 + 图）
- docs/learning/LAB15-boundary20.md — LAB15 讲义（为什么 20 种子、读数顺序、通过标准、五个坑）
- docs/learning/LAB15-prediction.md — LAB15 预测模板（待用户填写）
- runs/20261119-001-lab13b-true-mid/ — 补跑 20 次（20/20 down；原始逐次文件仅本地，1.7 MB）+ README §6 数值漂移表
- docs/DECISIONS.md — D25 补充条：倒地/翻滚类样本的单次数值不可复现，引用时报 3 遍中位数

## 2026-11-19（LAB14 可重复性）

- runs/20261119-001-lab14-replicate/ — 2 格 × 5 种子 × 3 遍 = 30 次重复评估；判定层零变化，数值层有位级抖动（README + replication.md/json + rep<N>/）
- docs/DECISIONS.md — D25：重跑噪声判定层为 0；“再跑一遍”不能替代加种子；边界写成倒地率 + N
- docs/REPORT.md — §2.5.5 可重复性节
- docs/learning/LAB14-prediction.md — 判分 3.0/5

## 2026-11-19（LAB13 更正 + LAB13b 补测）

- runs/20261119-001-lab13b-true-mid/ — LAB13b 补测：80/90 N·s（400/450 N）× −x/−y × 5 种子 = 20 次，20/20 倒（README + recovery）
- docs/DECISIONS.md — D23 追加更正 + 新增 D24：扰动一律写“冲量 N·s（力 N × 0.2 s）”；旧“非单调/凹坑”作废
- docs/REPORT.md — §2.5.4 重写（六点单调表；5/5 从 80 N·s 起，边界 60–80 N·s）；§3 发现 5 更新
- README.md — 扰动结果行改为冲量口径 + 单调结论
- docs/learning/LAB13-mid-impulse.md — 顶部更正注 + §9 补测读数
- docs/learning/LAB13-prediction.md — 按 80/90 N·s 重判 2.5/5（旧 1.5/5 作废）
- docs/learning/LAB14-*.md / scripts/lab14_*.* — 重跑格改为 −y 60 / 80 N·s
