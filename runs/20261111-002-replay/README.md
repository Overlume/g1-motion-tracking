# run 20261111-002-replay — 官方预训练 checkpoint 回放（M1）

- 目的：确认"现成策略 + 参考动作"在本机能跑通并产生可看的 G1 动作视频；区分官方预训练结果与本人训练结果。
- 运行位置：本地 WSL2，RTX 3060 6GB，MUJOCO_GL=egl 离屏渲染。
- 代码：repos/mjlab_research v1.6.0；脚本 scripts/replay_policy.py（自写，跳过 play.py 的 viewer 阻塞）。
- 检查点：**官方预训练** scratch/model_49999.pt，来自 https://storage.googleapis.com/mjlab_beta/model_49999.pt，6,769,287 bytes，sha256 f1bc404f...19（与 gcs.py 记录一致）。
- 动作：data/processed/lafan_dance1_subject1.npz（6574 帧 50fps，LAFAN1 dance1 subject1，经 scripts/convert_demo_motion.py 转为 v1.6.0 格式）。
- 命令：
  `MUJOCO_GL=egl environment/.venv/bin/python scripts/replay_policy.py --checkpoint-file scratch/model_49999.pt --motion-file data/processed/lafan_dance1_subject1.npz --video-length 300 --num-envs 4`
- 结果：EXIT=0，301 步；视频 outputs/videos/model_49999/rl-video-step-0.mp4（320x240，50fps，300 帧，566KB）；GIF 预览同目录 replay.gif；抽帧 frames/t*.png。
- 观察：4 个并行环境，画面中 G1 在棋盘地面上完成舞蹈动作（t=2.0s 双臂展开、t=4.0s 收臂），说明 checkpoint 加载成功、动作命令驱动闭环控制、EGL 离屏渲染可用。
- 未验证/限制：
  - 视频只录了单个环境视角（VideoRecorder 默认），不能据此判断"动作完成率"。
  - play=True 下无扰动、关闭 RSI；训练态 config 与评估指标未跑。
  - 视频分辨率 320x240 是默认值，可用 --video-width/height 提高。
- 结论：M1 的"官方预训练策略可回放"达成；下一步是把回放变成可量化评估（跟踪误差、脚滑、摔倒判据）。
- 日志：environment/logs/replay_demo.log

## 追加：960x540 版本（2026-11-12）

- 命令加 --width 960 --height 540，输出到 outputs/videos/model_49999/960x540/。
- 产物：rl-video-step-0.mp4（960x540、600 步 = 12.02 s、1,584,008 B）、preview.gif、frames/t1.0/t4.0/t7.0/t10.0.png。
- 日志：environment/logs/replay_960x540.log。

