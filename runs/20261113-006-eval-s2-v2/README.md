# 20261113-006 eval-s2-v2 (fight1_subject2, terminations ON)

目的：用 v2 脚本复查最早那个 fight 片段（134.6-154.6 s 最激烈窗口），确认录像里动作反复重播的原因。

结果（seed 0）：

- reset 79 次，步 11, 24, 37, ... 间隔约 12.5 步（0.25 s），触发项全部 ee_body_pos。
- motion_frame_furthest 只有 13 → track_completion 0.013；每段都只有 0.24 s 左右。
- MPKPE 11.73 cm / R-MPKPE 8.18 cm 看起来很好，但这是 0.24 s 循环里的统计，不能代表 20 s 跟踪。
- 结论：runs/20261113-002 的 11.7 cm 与 outputs/videos/model_49999/960x540/fight1_134s 的视频都作废（视频是 0.24 s 循环反复播放，看起来像静止画面）。

限制：仅说明 termination 会频繁打断，真实行为见 009。