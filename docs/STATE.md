# STATE

最后更新：2026-11-18 (+08:00)
当前阶段：M0–M6 全部完成并已公开；M5 阈值细化与终止格视频核对已完成（2026-11-18）。
入口提示：本文件后面的小节按时间累积；**当前事实以本文件末尾最新小节为准**（`docs/handoffs/` 已于 2026-11-18 移出公开仓库）。

## 当前阶段完成标准

- 官方身份、代码开放度、数据来源、许可、硬件可行性均有可引用证据。✅
- 本地隔离环境可运行，且能用至少一种方式证明 MuJoCo Warp 在本机 GPU 上工作。✅（20261111-001-smoke）
- 路线已决定。✅（D2：mjlab 基线优先；用户授权智能体代选）

## 已验证成果（含证据）

1. **SuperMimic 身份**：arXiv 2610.28480v2，Qiayuan Liao 等，UC Berkeley，CoRL 2026；项目页 https://supermimic.github.io/；仓库 https://github.com/HybridRobotics/supermimic（Apache-2.0）。证据：docs/SOURCES.md。
2. **代码完整度**：仅 src + tests + pyproject + uv.lock；无 configs/数据/检查点/文档；默认训练 4096 envs；运动数据走 W&B registry（默认 supermimic/wandb-registry-motions/lafan_run1_subject5）。证据：repos/supermimic/README.md、repos/supermimic/src/supermimic/scripts/train.py、repos/supermimic/src/supermimic/runners/supermimic_runner.py。
3. **mjlab 基线**：mjlab v1.6.0（Apache-2.0），含 Mjlab-Tracking-Flat-Unitree-G1，G1 跟踪 14 个 body、脚底摩擦、ee_body_pos 终止条件；mjlab demo 会从 GCS 下载预训练 checkpoint 与 LAFAN dance1 动作（num_envs=8）。证据：repos/mjlab_research/src/mjlab/scripts/gcs.py、repos/mjlab_research/src/mjlab/tasks/tracking/config/g1/env_cfgs.py。
4. **动作格式**：v1.6.0 MotionLoader 需要 root_pos_w/root_quat_w(wxyz)/root_lin_vel_w/root_ang_vel_w/joint_pos/joint_vel/joint_names；GCS demo npz 是旧格式（body_pos_w 等，29 关节/30 body/6574 帧/50fps/11.8MB）。已写 scripts/convert_demo_motion.py 转换，产物 data/processed/lafan_dance1_subject1.npz（6574 帧，root_pos z 0.288-0.842）。证据：repos/mjlab_research/src/mjlab/tasks/tracking/mdp/commands.py:52-110。
5. **数据许可**：BONES-SEED 为专有许可，仅 Academic User / Qualifying Startup；HF 上 gated=auto，g1.tar.gz 23.5GB。决定不使用（D4）。
6. **硬件**：RTX 3060 Laptop 6GB（compute 8.6，驱动 616.64/CUDA 13.4）、20 核、WSL 内存 7.6GB、磁盘可用 918GB；WSL GPU 直通正常。
7. **项目内隔离环境**：scripts/setup_env.sh（uv 在 environment/bin/uv，缓存/解释器都在项目内）；Python 3.13.18 + torch 2.15.0+cu132（cuda True）+ mujoco 3.16.0 + mujoco-warp 3.16.0 + warp-lang 1.19.0，142 包，.venv 6.3GB。日志 environment/logs/setup_env.log。

## 冒烟实测（20261111-001-smoke，play 模式，warm）

| num_envs | create (s) | ms/iter | env-steps/s | nvidia-smi 峰值 |
|---|---|---|---|---|
| 8 | 7.73 | 17.76 | 450 | ~215 MiB |
| 64 | 7.87 | 19.86 | 3,223 | — |
| 512 | 8.08 | 28.83 | 17,760 | 283 MiB |

结论：6GB 显存不是障碍；每步耗时几乎与 env 数无关（固定开销主导）。首次冷启动 create 64s（kernel 编译），之后 ~8s。torch 显存统计无效（mujoco-warp 自带 mempool），须用 nvidia-smi。

## 未验证 / 未知

- **训练模式**（play=False，含 replay buffer / RSL-RL PPO 更新）的可行 num_envs、每迭代耗时、CPU 内存占用。
- WSL 7.6GB 内存是否够 512+ envs 的观测/回放缓冲。
- SuperMimic 的 W&B registry 是否公开可下载。
- LAFAN1 原始 BVH 的许可细则（若后续需要更多动作）。

## 运行中任务

- 无。

## 阻碍与失败尝试

- GitHub/HF 页面直接抓取多次超时或被 403（改用 curl + raw/API 成功）。
- supermimic 首次 git clone 报 TLS 中断，重试成功。
- 冒烟首次在 300s 预算内未打完（kernel 编译慢），非失败，需更长预算重跑。
- UNC 路径上 tools.edit 偶发 GetFileSecurityW EIO，改用 WSL 内 python 改文件。
- 后台 bash job 运行期间再发起别的 ops_bash 会触发 dsh-ops 执行树链接竞争：报 EXECUTOR_UNAVAILABLE / link awk.exe ... No command was executed；等后台 job 结束后自动恢复。教训：后台任务运行时不并发跑其他 bash 命令。

## M1 进展（20261111-002-replay）

- 官方预训练 checkpoint（model_49999.pt，6.77MB，sha256 校验通过）在 play 模式回放成功，4 envs，301 步。
- EGL 离屏渲染可用：视频 outputs/videos/model_49999/rl-video-step-0.mp4（320x240/50fps/300 帧）与 replay.gif。
- 观察：G1 在棋盘地面完成舞蹈动作（t=2.0s 展臂，t=4.0s 收臂），说明 checkpoint 加载、动作命令驱动闭环、渲染链路都通。

## M1 中间状态（数字已被下方"M1 完成"取代，仅作历史）

- 官方预训练 checkpoint（model_49999.pt）回放成功：960x540 视频 outputs/videos/model_49999/960x540/rl-video-step-0.mp4（600 步 = 12.02 s）。
- 量化评估完成（runs/20261112-001-eval）：全部 6574 步（131.48 s）跟踪，未跌倒；MPKPE 均值 7.36 cm / p95 14.12 cm；R-MPKPE 均值 3.36 cm；关节位置 RMSE 0.108 rad；pelvis 倾角误差均值 2.49°。
- 已知缺陷：eval 脚本未固定随机种子（两次运行 MPKPE 10.2 vs 7.36 cm）；脚滑指标用踝 link 位移，转身时高估。

## 下一步最小动作（M1 剩余）

1. eval 固定种子（cfg.seed），并改进脚滑指标（用足底接触点/接触传感器）。
2. 训练模式实测（play=False，512 envs 迭代耗时 / CPU 内存），为 M3 铺垫。
3. 读 SuperMimic 的 world model 代码，写一页对比笔记（M6 的素材）。

## M1 完成（2026-11-12）—— 当前事实以此为准

- 阶段 1 验收全部达成：回放、量化评估、训练模式实测。
- 官方预训练 checkpoint（scratch/model_49999.pt）在 lafan_dance1_subject1.npz（6574 帧 / 131.48 s）上跑完全程，无跌倒，track_completion=1.0。
- 指标（seed=0，runs/20261112-001-eval）：MPKPE 均值 10.738 cm（p95 17.432）、R-MPKPE 均值 3.42 cm（p95 5.58）、关节位置 RMSE 0.1084 rad、关节速度 RMSE 0.9504、 pelvis 倾角误差均值 2.564°、脚滑左 448.4 / 右 546.6 cm。
- 可复现性：同种子两次，MPKPE 10.738 vs 10.753，关节 RMSE 完全一致，脚滑差约 0.5%。早期未固定种子的 7.36/10.2 cm 作废。
- 训练模式实测：512 env 0.82 s/iter，1024 env 1.03 s（653 MiB），4096 env 1.76-1.90 s（1871 MiB）；6GB 可开官方默认 4096 envs。检查点 20MB 正常保存（runs/20261112-002/003）。
- 视频：outputs/videos/model_49999/960x540/rl-video-step-0.mp4（960x540，600 步 = 12.02 s）。

## 下一步最小动作（2026-11-13 更新）

1. **用户手做 LAB02**（docs/learning/LAB02-retarget-own-clip.md）：自己把另一个 LAFAN1 片段（fight1_subject3 前 600 帧）重定向、转 npz、跑评估，先预测再对结果。
2. 补 LAB01 的三次耗时数字（1/2/4 envs）。
3. M3：用 fight npz 做最小训练（4096 envs，几百次迭代），用户亲手改一个参数并先预测后对比 R-MPKPE。

## M2 进展（2026-11-13）—— 动作来源换成 LAFAN1 + GMR

- LAFAN1 原始数据已下载：data/raw/lafan1/lafan1.zip（144,051,503 B，sha256 ea918082...65a977），解压 77 个 bvh 到 data/raw/lafan1/bvh/（30 fps）。许可 CC BY-NC-ND 4.0，个人非商业本地可用，重定向产物不公开。
- GMR v0.2.0 已装在项目内 environment/gmr-venv（torch 2.15.0+cpu、mujoco 4.0.1），支持 bvh_lafan1 -> unitree_g1。
- 新脚本：scripts/gmr_bvh_to_csv.py（BVH -> pkl/csv，支持 --start-frame/--max-frames）、scripts/csv_to_npz_local.py（CSV -> mjlab npz，复用 mjlab MotionLoader，不传 W&B）。
- 首个自制动作：data/processed/fight1_s2_134s_20s.npz（fight1_subject2 的 134.6-154.6 s 最激烈窗口，600 帧@30fps -> 999 帧@50fps）。
- 校验：关节顺序与 mjlab G1 完全一致（order match True），关节限位越界 0 个，重定向 600 帧耗时 6.1 s（约 20 ms/帧）。
- 官方预训练策略在 fight 片段（runs/20261113-002-eval-fight，seed 0）：完成率 1.0、无跌倒，但 R-MPKPE 3.42 -> 8.18 cm、关节 RMSE 0.108 -> 0.301 rad、倾角误差 2.56 -> 7.55 度、pelvis 平均高度 0.743 -> 0.655 m。结论：平衡没崩，姿态跟踪明显退化。
  - 2026-11-13 16:20 更正：这一条作废。当时 replay/eval 脚本忽略 dones，episode 被 ee_body_pos 终止后环境自动 reset、动作从第 0 帧重播，24 步（0.48 s）就再次触发，实际评估的是 0.24 s 循环（reset 79 次、motion_frame_furthest=13、completion 0.013）。真实行为见下方 M2 更正一节：4.28 s 摔倒、root_err 末值 289.7 cm。
- 回放视频：outputs/videos/model_49999/960x540/fight1_134s/rl-video-step-0.mp4（1000 步 = 20.0 s）。

## 未验证 / 未知（2026-11-13 更新）

- GMR 重定向的物理可行性（人体 1.75 m 缩比到 G1、脚部接触假设）未单独验证，可能贡献一部分跟踪误差。
- 只有 fight1_subject2 一个片段；跨 subject（fight1_subject3/5）是否一致未测。
- 用户 LAB01 的三次耗时（1/2/4 envs）仍未回报。

## 带练模式调整（2026-11-12，用户反馈）

- 用户明确表示"全是你在操作，我好像只是在看你的结果"。后续每个单元先给用户可独立执行的 LAB（命令、预期输出、卡住处理），用户做完回报再继续；代操作只用于用户明确不做的部分。
- LAB 放 docs/learning/LABnn-*.md，用户回报结果写进 docs/learning/ 学习记录。


## M2 更正与实测诊断（2026-11-13 16:20）—— 当前事实以此为准

### 一、录像“约 3 秒有效后重复”的根因（已定位并修复）

- 旧版 scripts/replay_policy.py 的循环体只有 env.step(actions)，完全忽略 dones；旧版 scripts/eval_tracking.py 虽然把 done 写进 csv，也从不处理。
- mjlab 的 ManagerBasedRlEnv 在 termination 为真时自动 reset，并把 motion 相位清零（play 配置 sampling_mode 为 start）；录像器 video_recorder.py:131-132 又一定录满 video_length 帧才停 → 机器人回到起始姿态、参考继续推进，看起来就是动作反复从头播放。
- 4 env 同录时各 env 的 reset 时刻不同，所以画面里不同机器人各演各的（反复出现的起始张臂姿态）。
- 修复：scripts/replay_policy.py 改为 1 env 默认、逐帧打印 RESET（含 step、t、触发项、回到的 motion frame）、写 episode_log.json、新增 --ignore-terminations；scripts/eval_tracking.py v2 记录 motion_frame/done/fired、输出 reset 次数与触发项、按段统计。

### 二、修复后的实测（1 env，seed 0，官方预训练 checkpoint）

| 动作 | 帧数 | reset | completion | MPKPE cm | R-MPKPE cm | root_err cm | 关节 RMSE rad | 结果 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dance1 | 6574 | 1 | 0.9998 | 10.74 | 3.42 | 9.78 | 0.108 | ok 131.48 s |
| S3-terminations-on | 999 | 3 | 0.5295 | 95.27 | 9.73 | 94.42 | 0.236 | ok 19.98 s |
| S3-terminations-off | 999 | 0 | 0.6977 | 111.67 | 12.73 | 109.15 | 0.254 | 13.92 s fall |
| S2-terminations-off | 999 | 0 | 0.2152 | 137.56 | 27.26 | 132.16 | 0.397 | 4.28 s fall |

- S2 / S3 传 --ignore-terminations 后不再被打断，看到的就是策略真实行为：S2 在 4.28 s 摔倒，S3 在 13.92 s 摔倒。
- 参考动作自身速度（由 npz 的 root_pos_w 计算，dt=0.02）：S2 路径 34.42 m、净位移 2.45 m、平均 1.72 m/s、峰值 5.79 m/s；S3 路径 18.01 m、平均 0.90 m/s、峰值 2.56 m/s；dance1 路径 79.33 m、平均 0.60 m/s、峰值 2.15 m/s。

### 三、漂移归因（每条都有实测支撑）

1. 不是脚滑：漂移增量与脚滑的相关系数 corr = 0.066；S3 全程脚滑 187/259 cm，而根位移误差达 339 cm。
2. 不是 reset 污染：漂移从 step 0 起单调增长（0.2 -> 45.6 cm 只用 2 s）；同一个脚本测 dance1 的 root_err 只有 9.78 cm、drift_rate 0.01 cm/s。
3. 是根速度跟不上 + 没有全局位置反馈：观测里只有相对 anchor（motion_anchor_pos_b/ori_b），全局项只有奖励权重 0.5 / std 0.3（tracking_env_cfg.py:210-219），策略结构上看不到自己在世界里的位置与朝向 → 一旦落后就补不回来（S3 yaw 误差 |mean| 36.5 度、最大 ±333 度）。
4. 速度是分界线：0.60 m/s（dance1）root_err 9.78 cm；0.90 m/s（S3）段内漂移 14.31 cm/s、误差成批冲到 338.9 cm；1.72 m/s（S2，峰值 5.79）4.28 s 摔倒。

### 四、修复后的连续录像

- outputs/videos/model_49999/960x540/fight_s2_cont/rl-video-step-0.mp4（999 帧 / 19.98 s，0 reset，4.28 s 摔倒后在地上挣扎到结束）
- outputs/videos/model_49999/960x540/fight_s3_cont/rl-video-step-0.mp4（999 帧 / 19.98 s，0 reset，13.92 s 摔倒）
- 两个目录都含 episode_log.json（ignore_terminations: true、reset_count: 0）与 contact_2fps.png（4x5 抽帧总览）。

## 下一步最小动作（2026-11-13 16:20 更新）

1. 用户手做 LAB03（docs/learning/LAB03-verify-no-reset.md）：自己用修复后的脚本复测 S2/S3，先读 episode_log.json 与 segments，判断录像是否还会重复并解释原因。
2. M3 训练侧（每条都要同一套验收）：提高 motion_global_root_pos 权重 / 收紧 std；给观测补全局根位置或根速度；加水平漂移 termination；用 S3 与 dance1 复测，对比 root_err 与 R-MPKPE。


## LAB03 复盘与可复现性排查（2026-11-14 13:25）

- 用户 LAB03 作答已整理进 docs/learning/LAB03-verify-no-reset.md 的“我的记录”，共 6 问：
  - 对：连续录像判据（reset_count=0）、ee_body_pos 的四个刚体、sampling_mode=start 的解释、预测会摔倒、drift_rate 自己拟合成功。
  - 半对（已批改）：把 ignore_terminations 说成“检测到终止后不 reset 而是等录像完成” —— 实际是把 termination_manager.compute 换成恒 False；把旧 completion 1.0 归因于“不 reset” —— 实际是旧脚本忽略 dones、继续数步数（006 也 reset 了 79 次）。
  - 未完成：参考动作路径长度/速度（已补 scripts/motion_stats.py，待用户自己跑）。
- 用户的 lab3_check 日志显示 reset 在 6.68 s（step 334）与 7.90 s（step 395），与我们记录的 10.56 s 不一致 → 由此发现可复现性问题。
- 可复现性实测：dance1 1500 步两次差 0.5%（7.730 vs 7.693 cm）；S3 同 seed 两次差异很大（摔倒步 380 / 424 / 652 / 695 / 781，一次不摔）。结论见 D11。
- 脚本改动：scripts/replay_policy.py 与 scripts/eval_tracking.py 新增 --seed（默认 0）与 --keep-dr（默认关闭 base_com / encoder_bias / foot_friction 启动随机化）；episode_log 与 summary.json 记录 seed、keep_startup_dr。
- 4 env “各演各的”根因（回答用户问题）：每个 env 用自己的随机流采样上述启动随机化，再加上动作混沌，不是四个不同 seed；画面只剩 3 台是相机取景固定、640x360 有一台出画（frame0.png 开头即 3 台）。
- 新 run：runs/20261113-010/011（S3，DR 开，复现测试）、012/013（S3，DR 关）、014/015（dance1，DR 关，1500 步）。
- 当前最需要补的数字：S3 与 dance1 的多 seed 分布（N=5），用它替换 D10 里的单次摔倒时间。


## 取景修复与 S3 多 seed 分布（2026-11-14 14:10）—— 当前事实以此为准

### 一、4 env 只显示 3 台机器人：根因与修复

- 根因（读码确认，不是推测）：`repos/mjlab_research/src/mjlab/viewer/offscreen_renderer.py:186-217` 的 `_get_env_ids()` 只把 `cfg.env_idx` 和最近 `max_extra_envs` 个邻居提交给 `mjv_addGeoms`；`ViewerConfig.max_extra_envs` 默认 2（`repos/mjlab_research/src/mjlab/viewer/viewer_config.py:38`）→ 任何分辨率、任何 num_envs 下**最多只渲染 3 个 env**。第 4 台不是"出画"，是根本没进渲染列表。
- 更正：本文件 2026-11-14 13:25 段末"画面只剩 3 台是相机取景固定、640x360 有一台出画"作废。
- 第二个问题：默认相机 `origin_type=AUTO` 走 TRACKING 模式（`offscreen_renderer.py:232-260`），`cfg.lookat` 被忽略，相机固定 distance=5 跟着 env 0 的 pelvis；env 之间一旦漂移散开就装不下。
- 修复（两件事一起做）：
  1. `scripts/replay_policy.py`：`num_envs > 1` 时设 `viewer.max_extra_envs = num_envs - 1`；
  2. 新增 `scripts/camera_fit.py`：复现 `terrain_entity.py:368-381` 的网格布局，按 hfov/vfov 与仰角分别求最小距离，切到 `WORLD` 自由相机；`--camera auto` 可回退 mjlab 原行为。
- 新参数与兼容性（旧命令不用改）：`--camera {fit,auto}` 默认 `fit`；`--camera-margin` 默认 1.5；`--camera-elevation` 默认 -25；`--camera-azimuth` 默认 90。原有参数含义全部不变。
- 实测（`outputs/videos/model_49999/960x540/cam_fit_final`，num_envs=4、960x540、400 步、S3、`--ignore-terminations`）：
  `camera={"mode":"fit","lookat":[0,0,0.65],"distance":5.109,"elevation":-25,"azimuth":90,"half_x":1.8,"half_y":1.8,"max_extra_envs":3}`；
  抽帧 t=0.2 / 3.0 / 6.0 s 均可见 4 台机器人；t=6 s 有 1 台因漂移贴近画面下缘（物理漂移，不是渲染裁剪）。
- 过程（可复盘的两次迭代）：v1 公式用 `min(vfov,hfov)` 统一求距离 → 4 env 得 6.37 m，机器人只占画面高度约 20%（`cam_fit_test`）；v2 按水平/竖直分别算 + `elevation -25` + `margin 1.5` → 5.109 m（`cam_fit_final`）。

### 二、S3 多 seed 分布（N=5，DR 开，999 帧，`--ignore-terminations`）

| seed | run | 摔倒 | 摔倒步 | 摔倒时间 s | 完成度 | MPKPE cm | R-MPKPE cm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | runs/20261114-001-s3-seed101 | 是 | 928 | 18.56 | 0.9309 | 153.4 | 22.7 |
| 102 | runs/20261114-002-s3-seed102 | 是 | 277 | 5.54 | 0.2793 | 91.6 | 34.1 |
| 103 | runs/20261114-003-s3-seed103 | 是 | 583 | 11.66 | 0.5856 | 125.1 | 11.0 |
| 104 | runs/20261114-004-s3-seed104 | 是 | 510 | 10.20 | 0.5125 | 91.1 | 12.7 |
| 105 | runs/20261114-005-s3-seed105 | 是 | 607 | 12.14 | 0.6096 | 155.5 | 12.9 |

- 摔/不摔：**5/5 摔倒**（0% 存活）。
- 摔倒时间：中位数 **11.66 s**，范围 5.54–18.56 s（排序 5.54 / 10.20 / 11.66 / 12.14 / 18.56）。
- 口径：这 5 次跑的是"真实摔倒"（关掉 termination 一路跑到 pelvis_z<0.15 持续 5 步）；与之前记录的"首次 termination 步"（380/424/652/695）不是同一个量，不可混用。
- 教学点：seed101 摔倒最晚（18.56 s）却 MPKPE 均值最高（153.4 cm）——因为它漂移累积时间最长；R-MPKPE 最高的是 seed102（34.1 cm），它 5.54 s 就被格斗动作的快速位移甩开。两者一起看才说明"漂移"和"姿态"是两件事。

## 下一步最小动作（2026-11-14 14:10）

1. 用户按**原命令**复验取景（不传任何新参数）：`--num-envs 4 --width 960 --height 540`，确认 4 台入镜。
2. 用户回答 1 个问题：seed101 的 MPKPE 均值最高（153 cm），为什么它反而是最晚摔倒的？
3. M3：写 LAB04，让用户亲手起一次训练（建议 512 env、100 iter），先预测 iter 时间与显存峰值再实测。


## 取景第二版：follow 相机（2026-11-14 15:10）—— 取代上面 14:10 的 fit 默认

- 用户反馈：4 台都能入镜了，但视角不好（俯视、机器人小、地平线在画面中间）。
- 做法改变：默认 `--camera follow`（不再默认静态 fit）。每帧用 `entity.data.root_link_pos_w`（世界系根位置）算四台机器人质心放进 `cam.lookat`，距离按当前实际散布 + `--camera-min-distance` 自适应，相邻帧做 0.75/0.25 指数平滑（避免抖动）。
  - 踩坑：第一版用 `sim.data.qpos[:, :2]` 当世界坐标，结果每个 world 的 qpos 是**相对各自 origin** 的，质心永远在网格中心、完全不跟随（`distance_max` 卡在 4.5 就是证据）。改用 `root_link_pos_w` 后才真正跟随。
- 默认参数（都可用命令行覆盖）：`--camera-margin 1.3`、`--camera-elevation -16`、`--camera-azimuth 45`、`--camera-min-distance 5.2`；`--camera fit`（静态）与 `--camera auto`（mjlab 跟随）保留。
- 实测（dance1，4 env，960x540，400 步，`cam_default_dance1`）：四台全程在画面内，跟随使机身约占画面高度 28%（≈150 px / 540 px），低机位带地平线；`follow` 距离在 4.75–5.2 m 之间自适应。
- 已知限制：S3 这类会摔倒并滑出 2 m 格子的动作，摔到靠近相机那一台仍会被画面下缘切掉（`cam_follow_final` 可见）。要更保险就 `--camera-min-distance 6.5`，代价是机器人更小。
- 新参数兼容性：旧的 `--num-envs 4 --width 960 --height 540` 不用改动，默认即 follow；旧命令行为只会更稳（都渲染 + 跟随）。


## M3-b A/B 结果（2026-11-16）—— 当前事实以此为准

- 用户跑完 LAB05：A（原奖励）`runs/20261114-007-m3b-a-finetune`，B（weight 1.0 / std 0.15）`runs/20261115-001-m3b-b-finetune`，各 4096 env x 300 iter、seed 0、从 `scratch/model_49999.pt` 续训到 50298；评估 `runs/20261115-002-m3b-eval`（5 种子 101-105，终止检测开，启动 DR 关）。
- 两个反直觉结果：
  1. 微调 300 iter（29.5M env-steps ≈ 官方 50k iter 的 0.6%）把 S3 从“5/5 摔倒、中位 11.66 s”变成“0/5 摔倒、完成度 0.999”——两组都是。但这不等于学会：root xy 误差 43/53 cm、MPKPE 44/53 cm，只有根相对姿态 R-MPKPE 5.7/6.0 cm 还算好。
  2. 把 std 0.3 -> 0.15 让根位置跟踪**变差**：root_err 中位 43.34 -> 52.72 cm（两分布 5 种子互不重叠），drift 0.06 -> 0.70 cm/s，训练侧 error_anchor_pos 0.386 -> 0.508 m（两套独立测量同向）。
- 机制：奖励核 `exp(-d^2/std^2)` 在 d≈0.45 m 处的梯度幅值随 std 收窄从 ≈1.05 掉到 ≈4.9e-3，乘权重后 ≈0.52 vs ≈0.0049（差约 100 倍）——核一窄，该项在 0.3 m 外基本失效，权重 x2 补不回来。
- 口径差异（写报告要带）：训练 `episode_length_s = 10.0`（`repos/mjlab_research/src/mjlab/tasks/tracking/tracking_env_cfg.py:315`，最多 500 步），episode 主要由 `motion_end` 截断；评估跑满 999 步 / 19.98 s。A/B 同协议可比，但训练 `Mean episode length`（186/188）不能和评估的 999 步对照。
- 配置已还原：`tracking_env_cfg.py` sha256 = `f756b4d0e76a752a928cf41a6317f1147f8316319d8774c4c3769eb1df8b9a82`（2026-11-16 核对，weight=0.5 / std=0.3）。
- 细节与判读：`runs/20261115-002-m3b-eval/README.md`；决定记录见 D13。
- 下一步（二选一，等用户定）：① LAB06 只改 weight（0.5 -> 1.5，std 保持 0.3）的同尺寸 A/B，约 25 分钟，清掉 LAB05 同时改两项的混淆；② M4 规模测试：A 配置 3000 iter（约 1.8 h），看 root 误差是否随预算下降。


## LAB06 就绪（2026-11-16 15:50）

- 用户 2026-11-16 选定方案 A：先补“只改 weight”的对照，再谈规模。
- 讲义 `docs/learning/LAB06-weight-only-ab.md`（三种假设 -> 可区分预测、4 道预测题、成本、机器状态注意），脚本 `scripts/lab06_weight_only.sh`（train A|B / eval / report；B 组校验恰好 +1/-1 行且 std 仍为 0.3；新增 gpu_samples.csv 每 2 s 采样温度/功耗/显存）。
- 报告脚本 `scripts/lab05_ab_report.py` 已参数化 `--title/--labels`，LAB05 默认输出不变。
- 自测：2026-11-16 用 64 env x 6 iter 跑通 A（77 s）、B（51 s）、eval 2 种子、report；config 已还原，sha256 = f756b4d0…b9a82。
- 用户补充的机器状态（LAB05 A/B 训练期间）：内存 7.6 GiB 总/1.8 GiB 可用、swap 已用 1.5 GiB；显存 1865/6144 MiB；GPU 72% 利用率、82/93 W、**86 °C**。结论：显存不是瓶颈，瓶颈在 CPU/内存侧与散热；跨天比迭代耗时无效，须同时记录温度与内存（脚本已自动记录）。


## M3-b LAB06 结果（2026-11-16 17:30）

- 用户 2026-11-16 跑完 LAB06（只改 weight 0.5 -> 1.5，std 保持 0.3）：训练 `runs/20261116-001-lab06-a-finetune` / `runs/20261116-002-lab06-b-finetune`（4096 env x 300 iter、seed 0、S3、从 scratch/model_49999.pt 续训、49999 -> 50298），评估 `runs/20261116-003-lab06-eval`（5 种子 101–105）。
- 结果：root 误差 41.52 (39.82–42.59) -> **25.72 (24.63–26.47) cm，-38.1%，分布不重叠**；MPKPE 42.65 -> 26.86；R-MPKPE 6.250 -> 6.103（-2.4%）；joint_rmse 0.1668 -> 0.1677（+0.5%）；yaw 误差 24.28 -> 17.48 deg；代价：tilt 误差 +14.2%、左脚滑移 +24.8%、joint_vel_rmse +5.1%。两组均 0/5 摔倒、完成度 0.999。
- 训练侧：error_anchor_pos 0.392 -> 0.2084 m，episode length 193.8 -> 199.6，迭代耗时中位 1.93 -> 2.00 s。
- 与 LAB05 合看：LAB05-A 43.34 / LAB06-A 41.52（同配置重跑 → **噪声底 ≈4%**）/ LAB05-B 52.72 / LAB06-B 25.72。据此把 LAB05 的退步归因 std 收窄、LAB06 的进步归因 weight 加大。
- 假设判定：H1（核宽塌缩）成立；H2（观测缺失）、H3（权重过载）不成立。
- 用户预测：Q1 方向对幅度低估（-38% vs 预测 5–15%）、Q2 错、Q3 可复现性对而数字错、Q4 对。
- 新决定：D14（WSL2 时钟跳变与计时口径）、D15（std 固定 0.3，只加 weight）。
- 环境问题：B 组日志里 `Iteration time` 出现 -5.7 ~ -6.2 s。原因是本机 WSL2 的 CLOCK_REALTIME 周期性 ±7.5 s 跳变（实测 +7.43/-7.94 s，周期约 5.5 s，10 s 窗口净慢约 10%；`/proc/uptime` 与 perf_counter 正常）。不影响训练与指标，只影响耗时读数。用户可自行修复（sudo hwclock -s + timedatectl set-ntp false）。
- 配置已还原并核对：sha256 = `f756b4d0e76a752a928cf41a6317f1147f8316319d8774c4c3769eb1df8b9a82`（weight=0.5 / std=0.3）。
- 下一步（待用户定）：① **M4 规模训练**（weight 1.5、3000 iter，约 1.7 h + 10 min 评估）——推荐；② LAB07（weight 2.5，300 iter，约 35 min）先确认 1.5 是否饱和。


## LAB07 就绪（2026-11-16 19:10）

- 用户在 LAB06 之后要求“再做严谨一点的对照”：把 weight 从 0.5 / 1.5 两点扩成 **0.5 / 1.0 / 1.5 / 2.5 四点**（std 全部固定 0.3），之后再进 M4。
- 设计：四点才能看出曲线形状（单调下降 / 先降后平 / U 形），直接决定 M4 用哪个权重；同时 w1.0/std0.3 可与 LAB05-B（w1.0/std0.15）直接对照，把 LAB05 的混淆彻底算清。
- 只需两次新训练（w1.0、w2.5），w0.5 与 w1.5 复用 `runs/20261116-001/002` 的训练与 `runs/20261116-003-lab06-eval` 的评估。成本约 35 min。
- 新产物：讲义 `docs/learning/LAB07-weight-sweep.md`（曲线三形状表、4 道预测题、操作步骤、读图顺序、坑表、通过标准、还原）、脚本 `scripts/lab07_weight_sweep.sh`（train <weight> / eval / report；校验“与 LAB06-A 快照恰好 +1/-1 行、改的行含 weight=、weight 等于传入值、std 仍 0.3”）、汇总器 `scripts/lab07_sweep_report.py`（多臂表 + 曲线表 + 逐种子表 + sweep.csv + sweep.png，matplotlib 3.11.2）。
- 自测：2026-11-16 用 64 env x 6 iter 跑通 train 1.0 / train 2.5 / eval 2 种子 / report 四臂汇总（selftest 数值无意义，只验证管线），config 已还原，sha256 = f756b4d0…b9a82。
- 已知边界：每个臂只训练 1 个 seed，训练侧噪声未量化；评估噪声底 ≈4%，相邻两档差 <4% 只能写“不可分辨”。这些已写进汇总器的“读数提醒”。


## LAB07 结果：权重扫描（2026-11-16 20:05）

- 用户跑完 LAB07（四点 0.5 / 1.0 / 1.5 / 2.5，std 固定 0.3）：新训练 `runs/20261116-004-lab07-w10-finetune`（墙钟 629 s）、`runs/20261116-005-lab07-w25-finetune`（643 s），评估 `runs/20261116-006-lab07-eval`（5 种子 × 4 臂）；w0.5 / w1.5 复用 LAB06 的 run 与 eval。config 已还原，sha256 = f756b4d0…b9a82。
- 结果：root xy 误差 41.52 / **30.86** / **25.72** / **22.95** cm（w0.5 / w1.0 / w1.5 / w2.5），相对上一档 −25.7% / −16.7% / −10.8%（**单调下降、边际递减，地板 ~20 cm**）；R-MPKPE 6.25 / **5.98** / 6.10 / 6.45（**1.0 触底，2.5 超出 4% 噪声**）；joint_vel_rmse 1.22 / 1.24 / 1.28 / 1.33；左脚滑移 111.9 / 99.6 / 139.7 / 159.5；tilt 4.24 / 4.81 / 4.84 / **4.04**（2.5 最好）；yaw 24.28 / 17.99 / 17.48 / 19.88。四臂全部 0/5 摔倒、完成度 0.999。
- 训练侧 error_anchor_pos 0.392 / 0.2738 / 0.2084 / 0.1695 m（随权重单调下降）；episode length 193.8 / 184.5 / 199.6 / 166.9（2.5 明显变短）。
- 用户预测：Q1 形状与机制方向对（但“最先受影响的是朝向”不准）、Q2 错（tilt 不是灵敏探针，2.5 反而最好）、Q3 对（−25.7% > −16.7%）、Q4 决策规则对。
- 新决定：D16——**M4 用 weight 1.5**；不做 3.5 档；tilt 不作为代价探针（用 R-MPKPE / joint_vel_rmse / 滑移）。
- 判读细节与 5 条局限：`runs/20261116-006-lab07-eval/README.md`。
- 下一步：M4（weight 1.5、3000 iter、4096 env、seed 0、S3，约 1.7 h + 10 min 评估）。

## M4 就绪（2026-11-16 20:15）

- 新建 `docs/learning/LAB08-m4-scale.md`（H1 预算不足 / H2 配置到顶 / H3 过拟合退化 + 4 道预测题 + 四步命令 + 三条读数硬规则 + 卡住对照 + 通过标准）。
- 新建 `scripts/m4_scale_train.sh`：`train`（3000 iter、save 500、校验 weight=1.5 且相对 LAB06-A 恰好 +1/-1 行、墙钟用 /proc/uptime）、`eval`（5 种子评估 model_52998.pt）、`report`（复用 lab07_sweep_report.py，臂标 300 / 3000）。
- `scripts/lab07_sweep_report.py` 增 `--xlabel`（表头/横轴/标题可覆盖，M4 用 "training iterations"），LAB07 默认行为不变。
- 自测：64 env × 40 iter 62 s 通过、ckpt 齐全、eval 单种子通、report 两臂并表出图；config 已还原 sha `f756b4d0…b9a82`。
- 待用户执行（约 1.7 h + 10 min）：写 `docs/learning/LAB08-prediction.md` -> 改 weight=1.5 -> `train` -> `eval` -> `report`。

## M4 结果：300 vs 3000 iter（2026-11-17）

- 训练 `runs/20261117-001-m4-w15-finetune`（4096 env × 3000 iter，49999 → 52998，seed 0，weight 1.5，真实墙钟 6412 s，1.99 s/iter 中位，142/3000 负值迭代 = D14）。
- 评估 `runs/20261117-002-m4-eval`（5 种子）：root xy 25.72 → 3.90 cm（−84.8%）、p95 76.11 → 9.19、R-MPKPE 6.10 → 3.66、MPKPE 26.86 → 5.64、joint_rmse 0.1677 → 0.1168、yaw 17.48 → 11.36、滑移 139.7 → 91.3；tilt 持平（噪声内）；两臂 0/5 摔。H1 成立、H2/H3 否。
- 泛化探针 `runs/20261117-003-generalization-probe`（探索性 n=1）：同 ckpt 换未见窗口 `fight1_s3_120s_20s.npz` 完成度 0.101（24 次 ee_body_pos reset、最长 2.02 s）、换未见片段 S2 0.024；官方 ckpt 在未见窗口 0.016。→ 3.90 cm 是 in-distribution 特化，不是泛化。
- 新数据（冻结流水线生成，LAFAN1 不公开分发）：`data/processed/fight1_s3_120s_20s.npz`（999 帧，sha256 4aac9984…）、`data/processed/gmr/fight1_s3_120s_20s.csv|.pkl`。
- 新决定 **D17**：报提升必须标注 in-distribution / held-out；换片段评估不得用 root 误差给频繁 reset 的 rollout 排序；M5 须写成“对特化策略的扰动”。
- 用户预测判分：Q1 方向对幅度保守（预测 10 cm / 实测 3.90）、Q2 错（R-MPKPE 反而 −40%）、Q3 口径混淆（训练侧 error_anchor_pos 是 3D + 抽帧 + 噪声，实测 0.114 而非 0.05）、Q4 前提错但排查链合理。
- 下一步：LAB09 把泛化探针正式化（5 种子 × 2 片段 × 2 策略）；之后 M5 扰动协议（双策略）。

## LAB09 就绪（2026-11-17）

- 讲义 `docs/learning/LAB09-generalization.md`：三片段（train / 同片段未见窗口 / 未见片段）× 两策略（官方 / M4 model_52998），讲清 in-distribution 与 held-out 的区别。
- 脚本 `scripts/lab09_generalization_probe.sh`（run / report，默认 6 组合 × 5 种子 = 30 次评估 ≈ 20 min）+ `scripts/lab09_report.py`（D17 reset-guard：中位 reset > 5 的格把 root / pose 误差写成 n/a†）。
- 自测：2 组合 × 1 种子通过；report 正确把官方-heldout 那格标为 n/a†（完成度 0.016、reset 70、终止全 ee_body_pos）。

## LAB09 结果：泛化探针（2026-11-17）

- `runs/20261117-004-lab09-generalization`（6 组合 × 5 种子，1136 s）：
  - m4 策略：train 完成度 0.999 / root 3.88 cm；held-out 窗口 0.101（最长 2.02 s）；未见 S2 0.025（0.50 s）。
  - 官方策略：train 0.333 / root 82.83 cm（最长 6.66 s）；held-out 窗口 0.016（0.32 s）；未见 S2 0.013（0.26 s）。
  - 30 条 rollout 全部无摔倒判定，终止原因只有 `ee_body_pos` / `motion_end`。
- 结论：M4 的 3.90 cm 是 in-distribution 特化（换窗口 −90% 完成度）；两条策略在未见片段上都只有 0.3–0.5 s 存活。
- 口径澄清（D18）：11-14 的“官方 S3 5/5 摔”是终止关；“0/5 摔”是终止开、被 ee_body_pos 先行重置，两者是同一失败的不同阶段。
- 用户预测：Q1 数值对（借用了探索数据）、Q2 对、Q3 对、Q4 方向对但缺 held-out 数字与 in-distribution 标注。
- 下一步：M5 受控扰动（对象 = 特化策略 model_52998，官方作对照；世界系 ±x/±y × 3 档冲量 × 5 种子）。

## M5 就绪（2026-11-17）

- 讲义 `docs/learning/LAB10-m5-perturbation.md`；脚本 `scripts/m5_perturb.sh`（run/report）+ `scripts/m5_report.py`。
- 设计：torso_link 世界系恒力 0.2 s（8.00–8.20 s），档位 j30/j60/j120 = 30/60/120 N·s = Δv 0.90/1.80/3.60 m/s（整机 33.34 kg，g1.xml 30 body 求和）；4 方向 × 3 档 × 5 种子 = 60 次评估 ≈ 40 min；基线复用 LAB09 `m4_train`。
- `scripts/eval_tracking.py` 追加 4 个默认关闭的扰动参数（v2.1，走 `write_external_wrench_to_sim`）；回归验证：不加参数时 seed101 复现 LAB09 的 3.98 cm / 0.999 / 1 reset。
- 单种子标定：12 N·s 无感、60 N·s 后 0–1 s 28.6 cm 后恢复、120 N·s 在 8.18 s 触发 reset（完成度 0.591）→ 三档正好跨在阈值两侧。
- 脚本自测：2 组合 × 1 种子通过，报告 † 标记（扰动窗口内 reset 时误差不可比）生效。

## M5 结果：受控外力扰动（2026-11-18）

- `runs/20261117-005-m5-perturb`（12 格 × 5 种子 = 60 次评估，基线 LAB09 m4_train）：
- 终止阈值：+y ≤0.90 m/s（j30 就崩）；−x、−y 落在 0.90–1.80 m/s；+x 落在 1.80–3.60 m/s（j60 抗住、j120 崩）。
- 恢复质量（未终止格）：+x j30 后 0–1 s 13.2 cm → 1–2 s 5.3；+x j60 28.1 → 9.5；−x j30 38.3 → 25.3（慢）；−y j30 54.4 → 34.7（最慢）。
- 终止全部发生在 8.14–8.62 s，即推力窗口结束后 0–0.4 s（≤20 步），没有“踉跄后站稳”的过程。
- 口径见 D19：终止格完成度 0.57–0.59 是预算伪影；抗扰结论只对特化策略 + 该片段 8.0 s 成立。
- 工具 bug：首轮 −x 因 argparse 把 `-150,0,0` 当选项而整组失败，已修（`--perturb-force="$vec"` + `M5_RUN_DIR` 追加），补齐后重跑 report。
- 下一步：M6（docs/REPORT.md + configs/perturb_protocol_v1.json + outputs/figures + 8 次提交）。

## M6 完成：报告与 8 次提交（2026-11-18）

- `docs/REPORT.md`（11.6 KB）：一句话结论 + 设置与口径 + 5 组结果表（官方基线 / 权重扫描 / 规模 / 泛化 / 扰动）+ 6 条关键发现 + 6 条局限 + 复现命令 + 许可。
- `configs/perturb_protocol_v1.json`：M5 协议（body=torso_link、世界系、0.2 s 窗口、30/60/120 N·s、4 方向、种子 101–105、评估口径）。
- `outputs/figures/`：`m5_perturbation.png`、`weight_sweep_lab07.png`、`m4_scale_300_vs_3000.png`；`evidence/` 内 2 个 gif（≤1.7 MB）+ 3 张关键帧 png。
- `README.md`：新增一屏结果表（基线 / 调参 / 规模 / 泛化 / 扰动）+ 结论一句话。
- `LICENSE`（纯 MIT，copyright 2026 Overlume）+ `THIRD_PARTY_NOTICES.md`（LAFAN1 CC BY-NC-ND 不分发、mjlab Apache-2.0、GMR MIT、权重不入库）。
- `.gitignore`：`environment/*` 只放行 requirements.txt；`runs/*` 白名单放行 md/json/csv/png（不含 metrics.csv、logs、ckpt）；`outputs/videos/` 整体排除。
- 8 次提交（分支 main，已推送 origin）：60f2d86 chore → b26960b feat(eval) → 8e5c6c2 feat(retarget) → 3c1ee54 docs(lab) → 4e9fe32 feat(train) → 4e19846 feat(m4) → fd38cbe feat(m5) → 9fa1f3c docs(report)。
- 入库自检：140 个文件、`.git` 4.8 MB、无 `*.pt/*.npz/*.onnx/*.mp4`、无 metrics.csv、无 `/home/<user>`（runs 内绝对路径已改相对；文档内改为 `<workspace>`）。
- 已推送：https://github.com/Overlume/g1-motion-tracking （2026-11-18，git push --force-with-lease origin main；远端原有 GitHub 自动生成的 LICENSE 单独一次 initial commit，已由本项目 8 次提交整体替换）。提交作者 Overlume <overlume@qq.com>，历史用 git rebase --root -x "git commit --amend --no-edit --reset-author --date=<原日期>" 重写，保留原 author date。WSL 内无 git 凭据，push 时借用 Windows Git Credential Manager（credential.helper=!"/mnt/c/Program Files/Git/mingw64/bin/git-credential-manager.exe"）。
- 下一步：可选补 M5 阈值细化（+y 方向 0.45/0.60/0.75 m/s）或训练期 DR；报告与仓库已公开。

## M5 阈值细化结果（2026-11-18）

- run：`runs/20261118-001-m5-threshold-refine`（+y × {j15,j20,j25,j27} × 5 种子 = 20 次评估，墙钟 800 s，≈40 s/次）。
- 结果：0.45 m/s 0–1 s 13.48 cm、0.60 → 17.77、**0.75 → 43.05（峰值 74）**、0.81 → 53.53（峰值 105，1/5 在 10.46 s 终止）、0.90 → 5/5 终止 @8.62 s。
- 两级阈值：**0.75 m/s 跟踪崩（不终止）**，**0.81–0.90 之间开始终止**；0.81→0.90 之间是断崖，不是斜坡。
- 关键口径：本次 20 格 `fall_detected` 全为 False，终止由 `anchor_pos`（j30 seed101 @8.64 s）/ `ee_body_pos`（j60 seed101 @8.48 s）触发，**不是倒地检测**；与 `runs/20261114-001…005` 的官方策略真倒地（`fall_detected=True`）不同类，不可混写（D20）。
- 脚本：`scripts/m5_perturb.sh` 支持任意 `j<N>` 档位（`M5_TIERS`）；`scripts/m5_report.py` 加 `--tiers/--dirs`；新增 `scripts/m5_combined_plot.py` 合并粗扫+细化出图（`outputs/figures/m5_perturbation.png` 已重画）。
- 下一步：−x/−y 补 j40/j45 收窄“≥2 倍”；终止格抽 2–3 个回视频确认是否真倒地；抗扰结论若要可比较需训练期扰动。

## M5 终止格视频核对（2026-11-18）

- run：`runs/20261118-002-m5-video-check`。给 `scripts/replay_policy.py` 加 `--perturb-body/--perturb-force/--perturb-start-s/--perturb-duration-s`，按同一 ckpt（`model_52998`）、seed、力争重放 4 例（3 终止 + 1 存活）。
- 复现一致性：录像 reset 时刻与评估差 ≤1 帧——`py_j30_s101` 8.64 vs 8.62（`anchor_pos`）、`py_j60_s101` 8.48 vs 8.48（`ee_body_pos`）、`py_j27_s103` 10.48 vs 10.46（`ee_body_pos`）。
- 观察：3 个终止格在终止前 0.04–0.09 s 已跪坐/坐倒/手撑地；存活对照 `py_j27_s101` 蹲到 0.506 m、倾角 31°、踉跄后站回。
- 数值：4 例 `pelvis_z` 最低 0.425/0.546/0.524/0.506 m ≫ `FALL_HEIGHT=0.15`（`scripts/eval_tracking.py:41-42`）→ `fall_detected` 全 False。
- 结论：**终止 ≈ 跪坐倒地**；`fall_detected` 只识别平躺，不能当“是否摔倒”的判据（D20 修订）。
- 证据：`outputs/figures/evidence/m5_fall_check_{py_j30_s101,py_j60_s101,py_j27_s103,py_j27_s101}.jpg`。
