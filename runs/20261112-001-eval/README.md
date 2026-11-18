# run 20261112-001-eval — 官方预训练策略的量化跟踪评估（seed=0，可复现）

- 目的：把 M1 的"看着像"变成可复现的数字；先定义指标再测。
- 运行位置：本地 WSL2 / RTX 3060 6GB，MuJoCo Warp，play=True，单环境，seed=0。
- 检查点：**官方预训练** scratch/model_49999.pt（sha256 f1bc404f...19）。
- 动作：data/processed/lafan_dance1_subject1.npz（6574 帧 @50Hz = 131.48 s）。
- 命令：MUJOCO_GL=egl environment/.venv/bin/python scripts/eval_tracking.py --motion-file data/processed/lafan_dance1_subject1.npz --seed 0 --out runs/20261112-001-eval
- 产物：metrics.csv（6574 行逐步数据）、summary.json、repro/summary.json（同种子第二次运行）、日志 environment/logs/eval_tracking.log 与 eval_tracking_repro.log。

## 指标定义

- MPKPE (cm)：14 个关键 body 世界坐标平均位置误差，含全局漂移。
- R-MPKPE (cm)：参考动作重新锚定到机器人当前根位置/朝向后计算，去掉漂移，只看姿态。
- joint RMSE (rad)：29 关节位置误差均方根。
- joint vel RMSE (rad/s)：29 关节速度误差均方根。
- 跌倒：pelvis 高度 < 0.15 m 连续 5 步。
- 脚滑 (cm)：参考脚处于支撑相（参考脚高 < 8 cm 且参考脚水平速度 < 1 cm/步），
  且机器人该脚接近水平（foot z 轴与世界 z 夹角 < 15 度）时，累加机器人踝 link 的水平位移。
  加入"脚水平"条件是为了排除原地转身时踝原点绕垂直轴画圈造成的假滑移；仍是近似。
- pelvis 倾角：pelvis 自身 z 轴与世界 z 夹角，并同时记录参考动作倾角作对照。

## 结果（seed=0，全部 6574 步）

| 指标 | 值 |
|---|---|
| 动作完成率 | 100%（6574/6574 步，131.48 s） |
| 跌倒 | 无；pelvis 高度均值 0.743 m，最低 0.382 m |
| MPKPE | 均值 10.74 cm，p95 17.43 cm，最大 23.88 cm |
| R-MPKPE | 均值 3.42 cm，p95 5.58 cm |
| 关节位置 RMSE | 均值 0.108 rad（≈6.2°），p95 0.161 rad |
| 关节速度 RMSE | 均值 0.950 rad/s |
| pelvis 倾角误差 | 均值 2.56°（参考动作最大倾角 130.5°，机器人与之同步） |
| 脚滑 | 左 448.4 cm、右 546.6 cm（131 s 累计，粗测） |

## 可复现性（同种子跑两次）

| 指标 | run1 | run2 | 差 |
|---|---|---|---|
| MPKPE 均值 (cm) | 10.738 | 10.753 | 0.015 |
| R-MPKPE 均值 (cm) | 3.420 | 3.422 | 0.002 |
| 关节 RMSE (rad) | 0.1084 | 0.1084 | 0.000 |
| 左脚滑 (cm) | 448.35 | 450.62 | 2.27 |

结论：姿态类指标可复现（差异 < 0.02 cm）；脚滑对轨迹微小差异敏感（差约 0.5%），比较脚滑时应同时报多次运行的均值。

## 必须记住的限制

- 早期两次未固定种子的运行结果（MPKPE 7.36 / 10.2 cm）不可用于比较，已废弃；本文件数字才有效。
- 脚滑仍是近似：用踝 link 位移 + 水平判定，未用真实接触点力；转身/侧移时可能高估。
- 只测了一个动作、一个官方检查点，不外推。
