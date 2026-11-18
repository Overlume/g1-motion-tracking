# 20261113-005 eval-s3-cont (fight1_subject3, terminations OFF)

目的：关掉 termination，看策略在 S3 片段上不被打断时的真实行为。

命令：与 004 相同，加 --ignore-terminations，--out runs/20261113-005-eval-s3-cont。

结果（seed 0）：

- 0 次 reset；第 695 步（13.92 s）摔倒（pelvis_z 0.0917，tilt 104.2 度），motion_frame_furthest 697 → completion 0.6977。
- MPKPE 111.67 cm、R-MPKPE 12.73 cm、root_err 均值 109.15 cm（末值 174.52）、yaw 误差 |mean| 53.11 度、漂移速率 12.21 cm/s。
- 对照 004：开 termination 时看到的 3 次 reset 只是把 13.92 s 的摔倒提前切断，并没有掩盖别的故障。

限制：没有接触力传感器，摔倒判据是 pelvis_z < 0.15 m 连续 5 步。