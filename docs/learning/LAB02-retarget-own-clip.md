# LAB02 — 自己把一个 LAFAN1 片段变成机器人动作

目标：你亲手跑完"人的 BVH 动作 → 机器人关节轨迹 → 策略回放评估"这条链路，并判断它稳不稳。
预计 20–30 分钟，其中命令 2 分钟、看结果 5 分钟、写预测 5 分钟。

## 0. 先写预测（不看结果，先写下来）

对手上这条路（我 2026-11-13 用的是 fight1_subject2 的 134.6–154.6 s，R-MPKPE 8.18 cm）：

1. 换成**同一个人体动作的另一个演员**（fight1_subject3）后，R-MPKPE 会更高、更低，还是差不多？为什么？
2. 会不会摔倒？
3. 机器人的 pelvis 平均高度会比 0.655 m 更高还是更低？

把三句预测先发给我。

## 1. 进环境

```bash
wsl.exe -d Ubuntu-22.04
cd <workspace>
```

## 2. 重定向（GMR，约 10 秒）

```bash
environment/gmr-venv/bin/python scripts/gmr_bvh_to_csv.py \
  --bvh data/raw/lafan1/bvh/fight1_subject3.bvh \
  --start-frame 0 --max-frames 600 \
  --out-pkl data/processed/gmr/fight1_s3_000s_20s.pkl \
  --out-csv data/processed/gmr/fight1_s3_000s_20s.csv
```

看输出里的这行：`retargeting source frames [0, 600) = 20.00 s at 30 fps`，最后 `qpos shape (600, 36)`。

## 3. 转成 mjlab 格式（约 20 秒）

```bash
environment/.venv/bin/python scripts/csv_to_npz_local.py \
  --input-file data/processed/gmr/fight1_s3_000s_20s.csv \
  --output-file data/processed/fight1_s3_000s_20s.npz \
  --input-fps 30 --output-fps 50
```

看输出：`output frames: 999`，以及 `joint_names: shape=(29,)`。

## 4. 量化评估（约 1–2 分钟，会占满 GPU 几秒到几十秒）

```bash
MUJOCO_GL=egl environment/.venv/bin/python scripts/eval_tracking.py \
  --checkpoint-file scratch/model_49999.pt \
  --motion-file data/processed/fight1_s3_000s_20s.npz \
  --seed 0 --out runs/20261113-003-eval-fight-s3
```

打开 `runs/20261113-003-eval-fight-s3/summary.json`，抄回这几个数：
`track_completion`、`fall_detected`、`mpkpe_mean_cm`、`r_mpkpe_mean_cm`、`joint_rmse_mean_rad`、`pelvis_z_mean`。

## 5. 回放看画面（可选，约 1 分钟）

```bash
MUJOCO_GL=egl environment/.venv/bin/python scripts/replay_policy.py \
  --checkpoint-file scratch/model_49999.pt \
  --motion-file data/processed/fight1_s3_000s_20s.npz \
  --num-envs 4 --video-length 600 --width 960 --height 540 --tag fight1_s3
```

视频在 `outputs/videos/model_49999/960x540/fight1_s3/rl-video-step-0.mp4`。

## 6. 要和我的数字比的三件事

| 我测的（subject2，134.6 s 起） | 你测的（subject3，0 s 起） |
|---|---|
| R-MPKPE 8.18 cm | ? |
| 关节 RMSE 0.301 rad | ? |
| pelvis 高度 0.655 m | ? |
| 无跌倒 | ? |

注意：这两段不完全是同一个时间窗口，所以差异里既有"换演员"也有"换片段"的成分——这正是要在报告里写清楚的混杂因素。

## 卡住时

- `ModuleNotFoundError: mink`：用错了 python，重定向必须用 `environment/gmr-venv/bin/python`，转换/评估用 `environment/.venv/bin/python`。
- 评估报 `motion joints ... are not the robot's ...`：npz 不是由 `csv_to_npz_local.py` 生成的，检查是不是写错文件。
- 显存不足（CUDA out of memory）：play/eval 只用 1 个 env，正常不会；若出现贴完整报错给我。
