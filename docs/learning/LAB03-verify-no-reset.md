# LAB03：亲手验证“录像不再重复”，并自己读出漂移

> 目标：不是再跑一遍命令，而是让你能自己回答三个问题：
> 1）为什么旧录像里动作会反复从头开始？
> 2）修复后的录像和评估，凭什么可信？
> 3）fight 片段到底漂在哪、漂多快？

预计 30–40 分钟。全部在 WSL 里做，工作目录固定为：

    cd <workspace>

为什么所有命令都以 `environment/.venv/bin/python` 开头：项目依赖（torch 2.15 + cu132、mujoco-warp、mjlab）装在这个项目内虚拟环境里，没有污染系统 Python，也不用 conda。检查点 `scratch/model_49999.pt` 是 mjlab 官方预训练权重（6.7 MB，sha256 f1bc404f...22319，来源 storage.googleapis.com/mjlab_beta/model_49999.pt）；它只包含 actor/critic 参数，不含配置，配置来自 `repos/mjlab_research` 里的 G1 tracking 任务。

---

## 第 1 步：读一份已经生成的 reset 记录（3 分钟）

先看修复后的连续录像自带的时间线文件：

    environment/.venv/bin/python -m json.tool outputs/videos/model_49999/960x540/fight_s3_cont/episode_log.json

请回答（写在本文件末尾的“我的记录”里）：
- `reset_count` 是多少？`ignore_terminations` 是什么值？
- 这两个值组合起来，说明这段录像里的机器人是“同一段连续仿真”，还是“被打断后重开”？

旧版脚本没有写 episode_log，所以只能对比“旧脚本当时的评估摘要”与“修复后的摘要”：

    environment/.venv/bin/python -m json.tool runs/20261113-002-eval-fight/summary.json | head -20
    environment/.venv/bin/python -m json.tool runs/20261113-006-eval-s2-v2/summary.json | head -30

同一个片段（fight1_subject2）两次评估：旧摘要说完成率 1.0，新摘要说 reset 79 次、completion 0.013。请用一句话解释这个差异从哪来。

---

## 第 2 步：自己跑 200 步，亲眼看到 RESET 行（10 分钟）

**先预测**：S3 片段（fight1_subject3 前 20 s，重定向自 LAFAN1）在预训练策略上，第一次触发 termination 大概在第几秒？把你的预测数字写下来。

然后运行（注意：这次 **不加** `--ignore-terminations`，让 termination 正常工作）：

    MUJOCO_GL=egl environment/.venv/bin/python scripts/replay_policy.py \
      --checkpoint-file scratch/model_49999.pt \
      --motion-file data/processed/fight1_s3_000s_20s.npz \
      --num-envs 1 --video-length 200 --width 640 --height 360 \
      --tag lab3_check

参数为什么这样给：`--num-envs 1` 保证只有一个机器人、一条时间线；`--video-length 200` 是 200 步 = 4 s（50 Hz 控制），够看到第一次打断；`--width/--height` 小一点是为了渲染快；`--tag` 决定输出到 `outputs/videos/model_49999/640x360/lab3_check/`。

终端里会出现形如

    RESET env=0 step=... t=...s terms=ee_body_pos -> motion frame 0

的行。回答：
- 第一次 RESET 在第几步、第几秒？和你的预测差多少？
- `terms=ee_body_pos` 是什么意思？去 `repos/mjlab_research/src/mjlab/tasks/tracking/tracking_env_cfg.py:257-282` 找它的定义（阈值 0.25、body 名），再对照 `repos/mjlab_research/src/mjlab/tasks/tracking/config/g1/env_cfgs.py:64-69` 看是哪些刚体。
- 为什么 RESET 之后动作会“从头开始”？（提示：play 配置里 `sampling_mode` 是 `start`，复位会把 motion 相位清零）

---

## 第 3 步：把 termination 关掉，量出“到底漂多快”（15 分钟）

再跑一遍同样的片段，但这次关掉 termination：

    MUJOCO_GL=egl environment/.venv/bin/python scripts/replay_policy.py \
      --checkpoint-file scratch/model_49999.pt \
      --motion-file data/processed/fight1_s3_000s_20s.npz \
      --num-envs 1 --video-length 400 --width 640 --height 360 \
      --ignore-terminations --tag lab3_cont

看两件事：
- 玩家机器人（白色）和参考 ghost（绿色半透明）什么时候开始明显错位？大概第几秒开始“对不上同一个位置”？
- 到 400 步（8 s）时，两者相距大约几个身位？

**先预测**：把 termination 关掉后，最终是“漂到很远但一直站着”，还是“中途摔倒”？写下你的选择，再用下一步的评估结果验证。

跑评估（不打视频，只存逐帧指标，约 1–2 分钟）：

    MUJOCO_GL=egl environment/.venv/bin/python scripts/eval_tracking.py \
      --checkpoint-file scratch/model_49999.pt \
      --motion-file data/processed/fight1_s3_000s_20s.npz \
      --out runs/lab3-eval-s3 --seed 0 --ignore-terminations

然后自己算漂移速率（不要照抄我给的数）：

    environment/.venv/bin/python - <<'PY'
import csv
rows = list(csv.DictReader(open("runs/lab3-eval-s3/metrics.csv")))
t = [float(r["time_s"]) for r in rows]
e = [float(r["root_err_cm"]) for r in rows]
n = len(t)
mt = sum(t)/n; me = sum(e)/n
slope = sum((a-mt)*(b-me) for a, b in zip(t, e)) / sum((a-mt)**2 for a in t)
print("rows:", len(rows))
print("drift_rate_cm_s = %.2f" % slope)
print("root_err first/last = %.1f / %.1f cm" % (e[0], e[-1]))
PY

问题：
- 你算出的 drift_rate 与 `runs/lab3-eval-s3/summary.json` 里的 `segments[0].drift_rate_cm_s` 是否一致？
- 摔倒发生在第几步？（看 summary.json 的 `fall_step` / `seconds_run`）
- 参考动作本身在这 20 s 里走了多远？用 npz 的 `root_pos_w` 自己算路径长度与平均速度，再和 dance1 对比（dance1 平均 0.60 m/s）。

---

## 第 4 步（可选，5 分钟）：4 个环境为什么会“各演各的”

    MUJOCO_GL=egl environment/.venv/bin/python scripts/replay_policy.py \
      --checkpoint-file scratch/model_49999.pt \
      --motion-file data/processed/fight1_s3_000s_20s.npz \
      --num-envs 4 --video-length 200 --width 640 --height 360 \
      --tag lab3_fourenv

打开 `outputs/videos/model_49999/640x360/lab3_fourenv/episode_log.json`，看 4 个 env 的 reset 步是否不同。这解释了你之前看到的“画面里总有机器人保持张臂站立”——那不是录制坏了，而是某个 env 刚被 reset 回第 0 帧。

---

## 我的记录（2026-11-14 用户作答 + 批改）

> 你的原始答案保存在 <用户桌面>/answer.txt；下面按 LAB03 的 6 个问题整理，并标注对错与依据。

### 1. 旧录像与连续录像的差别

**你的回答**：reset_count 是 0；ignore_terminations 是让 ManagerBasedRlEnv 检测到终止条件时不重置环境，而是等待录像完成；两个值组合说明是同一段连续仿真。

**批改：结论对，机制要修正。** 我们并没有让环境“检测到终止后不 reset”，而是把 termination_manager.compute 整个替换成恒 False —— 环境根本不认为 episode 结束过，所以既没有 reset、也不会“等待”，它会一路跑到 --video-length 帧。效果上等价于你说的连续仿真，但代码上要精确。

**你的第二问**：差异是旧策略在检测到机器人即将摔倒时 reset、让机器人回初始位置，所以没真正着地；新策略不重置，所以 completion 0.013。

**批改：只对一半。** 两次评估其实都开着 termination、都 reset 了（runs/20261113-006 记录 79 次 reset）。真正的差别是：旧脚本忽略 dones，reset 之后仍在数步数，于是 completion 记成 1.0；新脚本逐帧记录，completion 只按 motion 相位最远到达的帧算 → 0.013。触发 reset 的也不是“即将摔倒”，而是四个末端（左右踝、左右腕）的 z 与参考差超过 0.25 m。

### 2. 第一次 RESET

**你的回答**：4 s 内机器人没摔；实际跑 500 步时，大约 6 s 摔倒并紧接着重置。

**批改：你的观察正确，而且比我给的数字更接近你的这次运行。** 你自己的 episode_log（outputs/videos/model_49999/640x360/lab3_check/episode_log.json）显示 reset 2 次：step 334（6.68 s）和 step 395（7.90 s），触发项都是 ee_body_pos，reset 后 motion frame 回到 1。

我原先记的“10.56 s（step 528）”是另一次运行的数值 —— 这段动作的评估不可复现（见下面“新发现”）。以后引用 fight 片段时只能写“约 7–15 s 之间失稳”，不能写死某个步数。

**ee_body_pos 判据（你的答案正确，补充一点）**：涉及 left_ankle_roll_link、right_ankle_roll_link、left_wrist_yaw_link、right_wrist_yaw_link（repos/mjlab_research/src/mjlab/tasks/tracking/config/g1/env_cfgs.py:64-69）；判据是机器人相对参考在 z 方向的误差超过 0.25 m（bad_motion_body_pos_z_only），任意一个超阈值就终止 —— 不是四个都超。

**sampling_mode = start**：正确。复位会把 motion 相位清零，所以动作从第 0 帧重来。

### 3. 关掉 termination 之后

- **错位时间**：你说 2 s 开始明显错位、7 s 末直接摔倒。这与 400 步视频看到的时序一致；但 eval 报的 fall_step 652（13.04 s）来自另一次运行，两者不是同一条轨迹（见新发现）。
- **400 步时相距几个身位**：你说“ghost 已经离开视线，无法判断”。这是对的描述 —— root_err 在 8 s 前后已经在 60–120 cm 量级（随段波动，例如 step 200 是 119 cm、step 400 是 60 cm），相机取景固定，ghost 会先走出画面。判断距离应该用 metrics.csv 的 root_err_cm，不要靠肉眼比对画面。
- **预测对错**：你预测会摔倒，结果确实摔了（那一次 fall_step 652）。预测正确。
- **drift_rate 与 summary 一致**：说明你已经会用 metrics.csv 自己拟合斜率了，这一步通过。
- **参考动作走了多远（path_length: 18.01 m
net_displacement: 0.48 m
mean_speed: 0.90 m/s | max_speed: 2.56 m/s
x_range: 2.92 m | y_range: 4.37 m
root_z: min 0.417 max 0.829）**：你不会算，我补了工具 scripts/motion_stats.py。它会直接给出路径长度、净位移、平均/最大水平速度、x/y 活动范围。用法：

      environment/.venv/bin/python scripts/motion_stats.py data/processed/fight1_s3_000s_20s.npz
      environment/.venv/bin/python scripts/motion_stats.py data/processed/lafan_dance1_subject1.npz

  请自己跑一遍，把 S3 的数字填进来（我这边算的是：路径 18.01 m、净位移 0.48 m、平均 0.90 m/s、峰值 2.56 m/s、x 范围 2.92 m、y 范围 4.37 m；dance1 是 79.33 m、0.60 m/s、2.15 m/s）。关键理解：路径长度把“来回走”都累加，净位移只看首末位置 —— fight 片段就是走得多、回原位，所以路径长、净位移小。

### 4. 4 个环境各演各的，以及画面里只有 3 个机器人

**你的猜测**：是不是用了四个不同的随机种子？

**批改：方向接近，机制不同。** 我们只设了一个 seed。每个 env 用自己的随机流采样启动时的域随机化：质心偏移（x ±2.5 cm、y/z ±5 cm）、编码器偏置（±0.01 rad）、脚底摩擦（0.3–1.2），再加上这段动作本身是混沌的，所以各 env 的终止步各不相同。你自己的 lab3_fourenv 日志就是证据：env1 在 step 99、189，env0 在 395，env3 在 395（同时触发 anchor_pos 与 ee_body_pos），env2 在 448。

**画面只有 3 个机器人：不是录制丢帧。** 你的 lab3_fourenv episode_log 里 env 0/1/2/3 都有记录，说明 4 个 env 都在跑；是相机取景固定、4 个出生点沿一条线排开，640x360 时有一台在画面外。我抽了第一帧存成 outputs/videos/model_49999/640x360/lab3_fourenv/frame0.png，开头就只有 3 台入镜，之后漂移还会让更多机器人出画。

### 新发现（因你的提问而查到，已修）

1. **同一 seed 也不可复现**：dance1 跑 1500 步两次，root_err 7.730 vs 7.693 cm、脚滑 69.57 vs 70.11 cm（差 0.5%）；S3 这种混沌片段会把这点差异放大 —— 摔倒步出现 380 / 424 / 652 / 695 / 781，还有一次完全不摔（runs/20261113-010）。
2. **已改脚本**：scripts/replay_policy.py 与 scripts/eval_tracking.py 新增 --seed（默认 0）与 --keep-dr（默认关闭启动域随机化），episode_log 记录 seed 与 keep_startup_dr。复核手段：同一命令跑两次、比对 summary.json。
3. **评估规则补充（D11）**：fight 类片段只作定性结论（“何时失去平衡”的区间），定量比较必须 N≥5 个种子并报中位数与范围；dance1 这类平稳动作可作精确对照。

### 还没做的

- 用 scripts/motion_stats.py 补齐第 3 步第 5 问（S3 与 dance1 的速度对照）。
- 想验证“相机取景”结论，可以跑 --num-envs 4 --width 960 --height 540 看是否 4 台都入镜。
