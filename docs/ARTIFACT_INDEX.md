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

## 2026-11-14 15:20 交接

- docs/handoffs/（仅本地，`.gitignore` 已排除，不随仓库分发）：最新 `20261118-2055-handoff.md`（M0–M6 收口、M5 三段、D21、下一步恢复实验）；历史 `20261114-1520-handoff.md`
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
