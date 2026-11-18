# 20261113-004 eval-s3-v2 (fight1_subject3, terminations ON)

目的：用修复后的评估脚本（逐帧记录 done / reset / motion frame）重测 LAB02 用的 fight1_subject3 前 600 帧片段。

命令（在项目根目录）：

    MUJOCO_GL=egl environment/.venv/bin/python scripts/eval_tracking.py \
      --checkpoint-file scratch/model_49999.pt \
      --motion-file data/processed/fight1_s3_000s_20s.npz \
      --out runs/20261113-004-eval-s3-v2 --seed 0

结果（seed 0，官方预训练 checkpoint，1 env）：

- reset 3 次，步 528 / 583 / 966，触发项全部是 ee_body_pos（z 方向 0.25 m 阈值）。
- motion_frame_furthest 529 → track_completion 0.5295；最长连续段 0–528（10.58 s），段内 root 漂移速率 14.31 cm/s。
- MPKPE 95.27 cm、R-MPKPE 9.74 cm、root_err 均值 94.42 cm、关节 RMSE 0.236 rad、pelvis_z_min 0.598 m、tilt 最大 57.57 度（参考最大 20.49 度）。
- 结论：这份片段在预训练策略上无法连续跟完 20 s；前 10.58 s 里姿态还行（R-MPKPE < 15 cm），位置持续漂移，之后被 ee_body_pos 打断。

限制：单 env、单 seed；未做多 seed 统计。