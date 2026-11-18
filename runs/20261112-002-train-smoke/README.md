# run 20261112-002-train-smoke — 训练模式最小冒烟

- 目的：证明"训练能启动、参数确实更新、检查点能保存"，并记录迭代耗时。
- 位置：本地 WSL2 / RTX 3060 6GB。
- 设置：Mjlab-Tracking-Flat-Unitree-G1，play=False，num_envs=512，seed=0，10 次迭代，tensorboard logger。
- 命令：
  MUJOCO_GL=egl environment/.venv/bin/train Mjlab-Tracking-Flat-Unitree-G1 --env.commands.motion.motion-file data/processed/lafan_dance1_subject1.npz --env.scene.num-envs 512 --env.seed 0 --agent.seed 0 --agent.max-iterations 10 --agent.logger tensorboard --log-root runs/20261112-002-train-smoke
- 结果：10 次迭代全部完成；第 0 次 31.48 s（kernel 编译/预热），第 1-9 次稳定在 0.81-0.86 s/次；总墙钟约 53 s。
- 检查点：runs/20261112-002-train-smoke/g1_tracking/2026-11-12_15-25-13/model_0.pt、model_9.pt（20MB）。
- 内存：训练进程运行时系统 used 约 2.1 GB / 7.8 GB。
- 现象：初始策略是随机的，平均 episode 长度仅 12-13 步（很快因 ee_body_pos 终止），符合预期；这不是"训练失败"。
- 未测：显存峰值（该次未采样，见 003-train-scale）、学习曲线是否上升（10 次迭代远不够）。
