# LAB04 — 亲手起一次训练（512 env × 100 iter）

> 这一课的目标**不是**"让机器人学会动作"（100 次迭代远不够），而是让你能独立回答四件事：
> 1）一次迭代里到底发生了什么、时间花在哪；
> 2）显存和内存的瓶颈在哪；
> 3）参数有没有真的更新、检查点能不能读回；
> 4）一次训练要写进 `runs/` 的哪些东西。
>
> 预计 40–60 分钟：命令 2–4 分钟（GPU），读日志 + 写结论 30 分钟。

工作目录固定：

    cd <workspace>

---

## 0. 先写预测（**跑之前**，写在本文件末尾或直接发我）

| 项目 | 我的预测 | 实测 | 偏差 / 解释 |
| --- | --- | --- | --- |
| 稳态迭代耗时（s，**不含**第 0 次） | 1.2s |  |  |
| 第 0 次迭代耗时（s，含 kernel 编译） | 107.6s |  |  |
| 显存峰值（MiB） | 300mib |  |  |
| 100 次迭代后 mean reward 比第 1 次：更高  |  |  |  |
| `model_0.pt` 与 `model_99.pt` 的 actor 参数是否不同 | 不同 |  |  |

用来校准量级的**已知实测**（别照抄，用它判断你的数字合理不合理）：

| 配置 | 迭代耗时 | 显存峰值 | 证据 |
| --- | --- | --- | --- |
| 512 env × 10 iter | 首次 31.48 s，随后 0.81–0.86 s | 当时没采样 | `runs/20261112-002-train-smoke` |
| 1024 env | 1.03 s | 653 MiB | `runs/20261112-003-train-scale` |
| 4096 env | 1.76–1.90 s | 1871 MiB | 同上 |

提示：显存随 env 数增长很慢（512→4096 只从约 0.5 GB 到 1.9 GB），说明 6 GB 不是瓶颈；真正吃时间的是每步固定开销 + PPO 更新。

---

## 1. 一次"迭代"里发生了什么（读 3 分钟）

拿默认配置算一遍，你就明白 100 次为什么不可能学会：

- `num_steps_per_env = 24`：每次迭代，每个 env 先连跑 24 个控制步采数据；
- 512 env × 24 步 = **12288 条转移** / 迭代；
- 每个控制步 `dt = 0.02 s` → 每次迭代只推进 **0.48 s** 仿真时间；
- 然后 PPO 用这 12288 条样本更新一次网络（actor + critic），每 50 次迭代存一次盘；
- 100 次 = 1.23 M env-steps ≈ **48 s 仿真时间**。

对照官方权重（文件名 `model_49999.pt`，mjlab 默认 4096 envs）：约 49999 × 4096 × 24 ≈ **4.9×10⁹** env-steps，是这次冒烟量级的上千倍。
所以这一课验证的是**机器在转、数字可信**，不是训练效果。学没学会是 M4 的事。

---

## 2. 跑（1 条命令，约 2–4 分钟）

```bash
bash scripts/lab04_smoke_train.sh 512 100
```

脚本会：自动建 `runs/YYYYMMDD-NNN-lab04-train-smoke/` → 写 `README.md`（命令、seed、环境版本、git 状态）→ 后台每 0.5 s 采一次显存 → 跑训练并把 stdout/stderr 全写进 `train.log` → 结束后算摘要写 `summary.json`。它**会把自己实际执行的完整命令打印出来**，不藏东西。

等价的裸命令（理解每个参数后再看脚本）：

```bash
MUJOCO_GL=egl environment/.venv/bin/train Mjlab-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file data/processed/lafan_dance1_subject1.npz \
  --env.scene.num-envs 512 \
  --agent.seed 0 \
  --agent.max-iterations 100 \
  --agent.num-steps-per-env 24 \
  --agent.save-interval 50 \
  --agent.logger tensorboard \
  --agent.run-name lab04 \
  --log-root runs/<你的 run 目录>/logs
```

| 参数 | 含义 | 这里给的值 / 为什么 |
| --- | --- | --- |
| `Mjlab-Tracking-Flat-Unitree-G1` | 任务 id：G1 跟踪、平地、训练模式（play=False） | 唯一在用的任务 |
| `--env.commands.motion.motion-file` | 参考动作 npz | dance1：6574 帧 / 131.48 s / 50 fps，已证明平稳，适合冒烟 |
| `--env.scene.num-envs` | 并行环境数，每个 env 是一份独立物理世界 | 512：6 GB 内的稳妥值，迭代约 0.85 s |
| `--agent.seed` | **唯一有效的种子** | 0，和之前所有基线一致 |
| `--agent.max-iterations` | 训练迭代数 | 100：够看到稳定耗时 + 存盘，又不浪费时间 |
| `--agent.num-steps-per-env` | 每次迭代每 env 采样多少控制步 | 24（框架默认，与官方一致，别动） |
| `--agent.save-interval` | 每多少次迭代存一次 ckpt | 50 → 会得到 `model_0 / model_50 / model_99` |
| `--agent.logger` | 日志后端 | `tensorboard`：默认是 `wandb`，要联网+登录，本机不用 |
| `--agent.run-name` | 日志目录后缀 | `lab04`，便于在日志目录里认出来 |
| `--log-root` | 日志根目录 | 指向本次 run 目录下的 `logs/` |
| `MUJOCO_GL=egl` | 离屏 OpenGL 后端 | 不录像也要设，避免 GLFW 打不开 display |

三个容易踩的点：

1. **`--env.seed` 没用**。`train.py` 里 `cfg.env.seed = seed` 会被 `--agent.seed` 覆盖（旧命令里两个都传过，是历史遗留）。
2. 默认 logger 是 wandb：不加 `--agent.logger tensorboard` 会卡在登录/联网。
3. `--log-root` 不是最终目录：mjlab 会在里面再套 `g1_tracking/<时间戳>_lab04/`，ckpt、`params/*.yaml`、tensorboard event 都在那一层。

---

## 3. 跑完看什么（10 分钟）

### 3.1 目录

```bash
ls -R runs/<你的 run 目录> | head -30
```

应有：`README.md`、`env_versions.txt`、`train.log`、`vram_mib.txt`、`iter_times.csv`、`summary.json`，以及 `logs/g1_tracking/<时间戳>_lab04/` 下的 `model_0.pt`、`model_50.pt`、`model_99.pt`（各约 20 MB，含 actor+critic+optimizer）和 `params/env.yaml`、`params/agent.yaml`。

### 3.2 日志里认几行

```bash
grep -E "Iteration time|Mean reward|Mean episode length|Mean action std|value loss|surrogate loss" runs/<你的 run 目录>/train.log | head -20
grep -E "Iteration time" runs/<你的 run 目录>/train.log | tail -5
```

- 第 0 次通常 20–35 s：kernel 编译 + 预热，**不要**拿它算平均；
- 稳态应在 0.8–0.9 s；偶发一次十几秒是 WSL 调度/时钟跳变（2026-11-12 那次出现过 15.13 s），记录但别当稳态；
- `Mean action std` 从 1.00 开始：这是探索噪声，训练久了会变，100 次内变化很小是正常的。

### 3.3 摘要

```bash
environment/.venv/bin/python -m json.tool runs/<你的 run 目录>/summary.json
```

看 `iter_time_s.median`、`vram_mib.peak`、`checkpoints`、`actor_delta.mean_abs_delta`、`mean_reward.first/last`。

### 3.4 自己证明"参数真的更新了"

不要信"训练跑完了"这句话，自己比权重：

```bash
environment/.venv/bin/python - <<'PY'
import glob, torch
fs = sorted(glob.glob("runs/<你的 run 目录>/logs/*/*/model_*.pt"))
print("checkpoints:", fs)
a = torch.load(fs[0], map_location="cpu", weights_only=False)
b = torch.load(fs[-1], map_location="cpu", weights_only=False)
print("iter:", a["iter"], "->", b["iter"], "| actor tensors:", len(a["actor_state_dict"]))
tot = n = 0
for k, va in a["actor_state_dict"].items():
    if k.endswith("_mean") or k.endswith("_var"):
        continue  # 跳过观测归一化的统计量，只看网络权重
    d = (b["actor_state_dict"][k] - va).abs()
    tot += float(d.sum()); n += va.numel()
print("mean |dw| =", tot / max(n, 1))
PY
```

注意：`model_0.pt` 是**第 0 次迭代之后**的权重（已经做过一次 PPO 更新），不是随机初始化。想拿纯随机初始化做对照，可跑 `--agent.max-iterations 0`（不推荐专门跑一次，理解这点即可）。

### 3.5 证明"检查点能读回"（脚本末尾会打印这条命令）

```bash
MUJOCO_GL=egl environment/.venv/bin/python scripts/replay_policy.py \
  --checkpoint-file runs/<你的 run 目录>/logs/g1_tracking/<时间戳>_lab04/model_99.pt \
  --motion-file data/processed/lafan_dance1_subject1.npz \
  --num-envs 1 --video-length 200 --width 640 --height 360 --ignore-terminations --tag lab04_ckpt
```

预期：能加载、能出视频；画面大概率很难看（随机初始化的网络只训了 100 次迭代）。这不是失败，正好说明"能读回"和"训得好"是两件事。视频落在 outputs/videos/model_99/640x360/lab04_ckpt/（目录名取检查点文件名，所以不是 model_49999）。不加 --ignore-terminations 的话，这个只训了 100 次的策略会每隔几步就触发 ee_body_pos 终止并 RESET（终端连续打印 RESET 行、episode_log.json 里 reset_count 几十次）——那同样证明"策略在闭环里跑起来了"，只是还站不住。

---

## 4. 写结论（10 分钟）

把下表填进 `runs/<你的 run 目录>/README.md` 的"我的记录"：

1. 512 env 的迭代耗时与 1024 / 4096 env 对比（1.03 s / 1.8 s），说明瓶颈在哪？为什么 env 数 ×8 而时间只 ×2？
2. 显存峰值离 6 GB 还有多少余量？训练时另开一个终端跑 `free -h`，观察 WSL 内存（总 7.6 GB）还剩多少。
3. 100 次迭代后 mean reward 的变化，能不能说明"学会了"？为什么不能？
4. `model_0.pt` 与 `model_99.pt` 的 `mean |dw|` 是多少？为 0 意味着什么？

---

## 5. 卡住怎么办

| 现象 | 处理 |
| --- | --- |
| `CUDA out of memory` | 降 `--env.scene.num-envs 256` 重跑；把完整报错贴给我 |
| 卡在 `wandb: (1) Create a W&B account` | 漏了 `--agent.logger tensorboard` |
| `train: command not found` | 不在项目根目录，或环境没装；`ls environment/.venv/bin/train` 确认 |
| `FileNotFoundError: ... motion-file` | 动作 npz 路径错；`ls data/processed/` 确认 |
| 结束后还有 nvidia-smi 进程 | `pkill -f "nvidia-smi --query-gpu"` |
| 日志出现负数迭代时间 | WSL 时钟跳变（已知问题），记录，用中位数 |
| 跑到一半想中断 | `Ctrl-C` 一次即可；已存的 ckpt 仍可用，但这次 run 不能当"完成" |

---

## 6. 通过标准

- `runs/<你的 run 目录>/` 里 6 个文件齐全，`summary.json` 能解析；
- `iter_time_s.median` 落在 0.7–1.2 s（512 env 的正常量级）；
- 三个 ckpt 存在，`actor_delta.mean_abs_delta > 0`；
- 你能用一句话说清"一次迭代里发生了什么"，并解释为什么 100 次迭代不足以学会动作。

做到以上 = M3-a 冒烟训练由你亲手完成，我们才进入 M3-b 单参数对照（改 `motion_global_root_pos` 权重）。

---

## 我的记录（用户填写）

### 预测（跑之前填）

（见第 0 节表格）

### 实测与解释

（跑完填）

### 我卡住/没想通的地方

（照实写，不丢人，这些是下一轮要讲的地方）
