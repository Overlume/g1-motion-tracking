# LAB22 附件：策略 vs 参考的骨盆高度偏移

口径：offset = pelvis_z − ref_pelvis_z（逐帧，正值 = 策略骨盆比参考高，即「悬空」）。

| metrics | 帧数 | offset 中位 | p5 | p95 | \|offset\|>2cm 占比 | root_err 中位 |
| --- | --- | --- | --- | --- | --- | --- |
| runs/20261117-002-m4-eval/final/seed101/metrics.csv | 999 | -1.73 cm | -4.04 | +1.53 | 47% | 3.63 cm |
| runs/20261117-002-m4-eval/final/seed102/metrics.csv | 999 | -1.76 cm | -4.06 | +1.49 | 48% | 3.36 cm |
| runs/20261117-002-m4-eval/final/seed103/metrics.csv | 999 | -1.70 cm | -4.05 | +1.67 | 47% | 3.43 cm |
| runs/20261117-002-m4-eval/final/seed104/metrics.csv | 999 | -1.72 cm | -4.04 | +1.60 | 48% | 3.57 cm |
| runs/20261117-002-m4-eval/final/seed105/metrics.csv | 999 | -1.71 cm | -4.01 | +1.52 | 48% | 3.09 cm |

读法：

- 中位 offset 绝对值 < 2 cm → 策略跟着参考的骨盆高度走：参考穿地不会变成「策略悬空」，而是变成腿/脚的去向偏差（joint_rmse、脚滑）。
- 中位 offset 明显为正（> 2 cm）→ 策略整体抬高，说明它在回避地面穿模；此时完成度与 root 误差要扣除这一层。
- 与 foot_audit.md 的最大穿地帧对照，可验证「策略在穿地帧是否抬高」。
