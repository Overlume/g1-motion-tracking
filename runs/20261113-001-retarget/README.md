# 20261113-001 — LAFAN1 fight1 → GMR 重定向 → mjlab npz

## 目的与假设

把"别人的动作"换成"自己选的动作"：验证 LAFAN1 BVH 经 GMR 重定向到 Unitree G1 后，能直接进入 mjlab 跟踪任务。

假设：
1. GMR 的 G1 模型关节顺序与 mjlab `G1_XML` 一致，无需重排；
2. 重定向结果不违反 G1 关节限位；
3. 官方预训练策略（在 BONES-SEED 上训练、只看过 dance 类）对 LAFAN1 fight 片段的泛化会明显下降但不应立刻摔倒。

## 运行位置与环境

- 本机 WSL2 Ubuntu 22.04，RTX 3060 Laptop 6GB。
- GMR venv：`environment/gmr-venv`（torch 2.15.0+cpu、mujoco 4.0.1、MIT 许可，`repos/GMR` v0.2.0）。
- mjlab venv：`environment/.venv`（mjlab v1.6.0，torch 2.15.0+cu132）。

## 数据

- 源：`data/raw/lafan1/bvh/fight1_subject2.bvh`（LAFAN1，CC BY-NC-ND 4.0；7347 帧 = 244.9 s @30 fps）。
- 取窗口：`--start-frame 4039 --max-frames 600`（134.6–154.6 s，20.0 s）。
- 选窗依据（`scratch/pick_fight_window.py`）：以左右手骨`LeftHand/RightHand`世界速度的 20 s 滑窗均值排序，4039 起始窗均值 5.44 m/s 为全片最高（全片中位数 3.40 m/s），即打斗最激烈的一段。

## 命令

```bash
# 1) BVH -> pkl/csv（GMR venv）
environment/gmr-venv/bin/python scripts/gmr_bvh_to_csv.py \
  --bvh data/raw/lafan1/bvh/fight1_subject2.bvh \
  --start-frame 4039 --max-frames 600 \
  --out-pkl data/processed/gmr/fight1_s2_134s_20s.pkl \
  --out-csv data/processed/gmr/fight1_s2_134s_20s.csv

# 2) CSV -> mjlab npz（mjlab venv，不经过 W&B）
environment/.venv/bin/python scripts/csv_to_npz_local.py \
  --input-file data/processed/gmr/fight1_s2_134s_20s.csv \
  --output-file data/processed/fight1_s2_134s_20s.npz \
  --input-fps 30 --output-fps 50

# 3) 官方预训练策略回放
MUJOCO_GL=egl environment/.venv/bin/python scripts/replay_policy.py \
  --checkpoint-file scratch/model_49999.pt \
  --motion-file data/processed/fight1_s2_134s_20s.npz \
  --num-envs 4 --video-length 999 --width 960 --height 540 --tag fight1_134s
```

## 结果

- 重定向：600 帧耗时 6.1 s（≈20 ms/帧）；输出 qpos (600, 36) = 3 位移 + 4 四元数(wxyz) + 29 关节。
- 转换：input 600 帧 @30 fps → output 999 帧 @50 fps（19.98 s）。
- 校验（`scratch/check_limits.py`）：
  - `order match: True`（npz joint_names 与 mjlab G1 模型关节顺序完全一致）；
  - 关节限位越界样本 0 个；
  - root z 均值 0.769 m，范围 0.621–1.163 m（t≈14 s 有 47 帧 >0.9 m，对应原片跳跃/扑击）；
  - 关节速度最大 42.5 rad/s（t≈10 s 的快速挥拳段，属真实动作而非 NaN/抖动）。
- 回放：1000 步（20.0 s）完成，视频 `outputs/videos/model_49999/960x540/fight1_134s/rl-video-step-0.mp4`（6,388,070 B）。

## 结论

1. 流水线打通：LAFAN1 原始 BVH → G1 关节轨迹 → mjlab npz，全程本地、无 W&B 账号。
2. 关节顺序与限位都自动满足，不需要手工映射表。
3. 视觉上机器人能跟上打斗的站架与挥拳，但绿色参考虚影（ghost）与白色实体在 8–14 s 明显错开 → 需要量化（见 runs/20261113-002-eval-fight）。

## 未验证 / 风险

- GMR 重定向本身有误差（人体 1.75 m → G1 约 1.3 m 缩比、脚部接触假设），跟踪误差中有一部分来自"参考本身是否物理可行"。
- 只测了 fight1_subject2 一个片段，不能推广到全部 LAFAN1。
- 许可：LAFAN1 为 CC BY-NC-ND 4.0，重定向产物不公开再分发。
