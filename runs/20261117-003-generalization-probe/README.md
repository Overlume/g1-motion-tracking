# run 20261117-003-generalization-probe — 泛化探针（探索性，n=1）

- 目的：验证 `runs/20261117-001-m4-w15-finetune/model_52998.pt` 的 3.90 cm 是“学会了跟踪”还是“记住了这个片段”。
- 做法：同一 ckpt 直接换评估片段，不重训、不改配置；种子 101；`scripts/eval_tracking.py`，终止检测开、启动 DR 关、999 步。
- 三条臂：
  1. `heldout_s3`：同片段未见窗口，`data/processed/fight1_s3_120s_20s.npz`（新建，注意：这是**没参与训练**的 120–140 s 窗口）。
  2. `unseen_s2`：未见片段，`data/processed/fight1_s2_134s_20s.npz`（训练时从未用过）。
  3. `official_heldout_s3`：官方 `scratch/model_49999.pt` 在 `fight1_s3_120s_20s.npz` 上的对照。

## 新数据（本次生成，LAFAN1 许可，不公开分发）

- 源：`data/raw/lafan1/bvh/fight1_subject3.bvh`（7347 帧 @30 fps），窗口 `--start-frame 3600 --max-frames 600`（120–140 s）。
- 中间产物：`data/processed/gmr/fight1_s3_120s_20s.pkl` / `.csv`（sha256 48e9f1c6…）。
- 训练格式：`data/processed/fight1_s3_120s_20s.npz`（999 帧 @50 fps，sha256 4aac9984…）。
- 生成命令（与 LAB02 同一冻结流水线）：

```bash
environment/gmr-venv/bin/python scripts/gmr_bvh_to_csv.py \
  --bvh data/raw/lafan1/bvh/fight1_subject3.bvh \
  --start-frame 3600 --max-frames 600 \
  --out-pkl data/processed/gmr/fight1_s3_120s_20s.pkl \
  --out-csv data/processed/gmr/fight1_s3_120s_20s.csv
environment/.venv/bin/python scripts/csv_to_npz_local.py \
  --input-file data/processed/gmr/fight1_s3_120s_20s.csv \
  --output-file data/processed/fight1_s3_120s_20s.npz --input-fps 30 --output-fps 50
```

## 结果

| 臂 | 完成度 | reset | 终止原因 | root 误差 cm | R-MPKPE cm | 段数 / 最长段 |
| --- | --- | --- | --- | --- | --- | --- |
| heldout_s3（3000 iter 策略，未见窗口） | 0.101 | 24 | 全 ee_body_pos | 12.70 | 10.44 | 25 / 2.02 s |
| unseen_s2（3000 iter 策略，未见片段） | 0.024 | 50 | 全 ee_body_pos | 14.66 | 9.52 | 50 / 0.48 s |
| official_heldout_s3（官方，未见窗口） | 0.016 | 70 | 全 ee_body_pos | 3.06† | 7.16 | 70 / 0.32 s |

† reset 太频繁，几乎没走出片头，root 误差不可比；此类 rollout 只看完成度与存活段。

对照训练片段（`runs/20261117-002-m4-eval/final/seed101`）：完成度 0.999、1 次 motion_end reset、root 3.98 cm、单段 19.96 s。

## 判读

1. **特化，不是泛化**：同片段换窗口就从 19.96 s 掉到 2.02 s，说明它依赖的是这个 20 s 窗口的具体时序，而非“跟踪参考运动”的能力。
2. **官方策略也挂，但更快**：官方在未见窗口上 0.32 s 就终止（70 次 reset），3000 iter 特化策略 2.02 s；两者都不是“会泛化”，只是后者对同片段的邻近窗口略有残余泛化。
3. **不影响 M4 的结论，但改变表述**：M4 应写成“在该片段上，预算轴把误差压到 3.90 cm 且 0 摔”，不能写成“策略学会了 S3 动作”。
4. **对 M5 的含义**：扰动实验必须同时标注“被扰的是一个特化策略”。若要谈“抗扰能力”，先要有泛化能力，否则扰动只是把回放打散。

## 局限

- n=1 种子、1 个未见窗口、1 个未见片段。
- 未见窗口从 120 s 起（片段中段，起始姿态与训练窗口不同）；评估器按参考帧把机器人摆到位，故起始误差≈0，误差在 0.3–2 s 内爆到终止阈值，指向“运动节律不匹配”。
- `ee_body_pos` 终止阈值偏紧，是跨片段失败的最先触发项；未做阈值敏感性分析。

## 下一步（建议，见 docs/PLAN.md）

- LAB09：把本探针正式化为 5 种子 × 2 片段 × 2 策略（官方 / 3000 iter），进对比表。
- 若目标是“会跟踪”，需多片段训练（数据侧），单片段微调无法买到泛化。

