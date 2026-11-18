# 20261113-007 eval-dance1-v2 (lafan_dance1_subject1, terminations ON)

目的：用 v2 协议复测 M1 的 dance1 基线，验证新脚本没有把原来合格的结果误判。

结果（seed 0）：

- reset 1 次，第 6572 步，触发项 motion_end（动作正常结束）。
- motion_frame_furthest 6573 → completion 0.9998；全程 131.48 s。
- MPKPE 10.74 cm（p95 17.37）、R-MPKPE 3.42 cm、root_err 9.78 cm、关节 RMSE 0.108 rad、pelvis_z_min 0.38 m、脚滑 450.9 / 546.0 cm。
- 段内漂移速率 0.01 cm/s（末值 0.03 cm），与 S3 的 14.31 cm/s 形成对照。
- 结论：v2 协议下 dance1 的结论与 M1 一致；同时说明"漂移"不是脚本的锅，而是动作速度决定的。