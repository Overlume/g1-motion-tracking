# 20261113-009 eval-s2-cont (fight1_subject2, terminations OFF)

目的：关掉 termination 后看 S2 片段的真实行为，回答"到底能跟多久"。

结果（seed 0）：

- 0 次 reset；第 213 步（4.28 s）摔倒（pelvis_z 0.111、tilt 166.2 度），completion 0.2152。
- root_err 均值 132.16 cm、末值 289.72 cm；漂移速率 51.57 cm/s；yaw 误差 |mean| 82.2 度。
- R-MPKPE 27.26 cm、关节 RMSE 0.397 rad —— 姿态也崩了，不只是位置。
- 参考动作本身：路径 34.42 m / 20 s，平均 1.72 m/s，峰值 5.79 m/s（由 npz 的 root_pos_w 计算）。
- 结论：这个片段（以及重定向后的高动态格斗动作）超出预训练策略能力，需要 M3 微调。