# LAB 01 — 你自己从终端跑通一次策略回放

预计 30–40 分钟。目标不是"看结果"，而是让你的手在键盘上完成：进入环境 → 跑通 → 采样显存 → 改参数 → 读代码 → 读出指标。

规则：遇到报错**不要自己乱改**，把完整报错贴给我；每一步的"你应该看到"就是验收标准，对不上就停下来。

---

## 第 0 步：进门（约 3 分钟）

1. 打开 Windows Terminal（或 PowerShell），输入 `wsl` 回车 → 提示符变成 `<user>@...:~$`。
2. 依次输入并回车：

```
cd <workspace>
pwd
ls
```

**你应该看到**：`pwd` 输出 `<workspace>`；`ls` 列出 README.md、AGENTS.md、docs、environment、repos、scripts、data、runs、outputs、scratch 等。

意义：整个项目只有一个根目录，后面所有命令都在这里执行。Windows 资源管理器里那个 wsl.localhost 路径是同一个地方的另一种写法。

---

## 第 1 步：确认"虚拟环境"是什么（约 5 分钟）

我们**不激活**虚拟环境，而是直接写路径调用里面的 python：

```
environment/.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**你应该看到**：`2.15.0+cu132 True`（约 5–10 秒，因为要加载 CUDA）。

理解要点：
- `environment/.venv/` 是本项目隔离的 Python 安装，和系统 Python 3.10 完全分开；删掉它不影响系统，但本项目就跑不了。
- `torch.cuda.is_available()` 为 True 说明 WSL 的 GPU 直通正常——这是整条链路的前提。
- 为什么不用 `conda activate`？因为写绝对路径的命令可复制、可写进脚本、不依赖当前 shell 状态。习惯它。

**小验证**：`which python3` 指向 `/usr/bin/python3`（系统 Python），和上面那个不是同一个。

---

## 第 2 步：第一次自己跑回放（约 10 分钟）

在项目根目录执行（行尾的反斜杠是"命令没写完，下一行继续"，照抄即可（Windows Terminal 里直接多行粘贴即可））：

```
MUJOCO_GL=egl environment/.venv/bin/python scripts/replay_policy.py \
  --checkpoint-file scratch/model_49999.pt \
  --motion-file data/processed/lafan_dance1_subject1.npz \
  --num-envs 2 --video-length 200 --width 320 --height 240 \
  --out-dir outputs/videos/lab1
```

**你应该看到**：先是 `Warp 1.19.0 initialized: ...` 和一堆 `mujoco_warp` kernel 日志，然后 40–90 秒（首次更慢），最后打印视频路径。用 `ls outputs/videos/lab1/model_49999/` 能看到 `rl-video-step-0.mp4`。

**这条命令在做什么**（逐段读）：
- `MUJOCO_GL=egl`：告诉 MuJoCo"没有显示器，用 EGL 离屏渲染"。忘了写会直接报错，这是最常见的坑。
- `--checkpoint-file`：策略网络权重（约 7 MB 的 .pt），不是仿真模型。
- `--motion-file`：参考动作（6574 帧关节角 + 根轨迹的 npz）。
- `--num-envs 2`：两个并行仿真环境，同时跑同一个动作。
- `--video-length 200`：录 200 步（50 Hz → 4 秒）。
- `--width/--height/--out-dir`：分辨率与输出目录。

为什么 200 步、320x240？因为**先要快**。跑通链路比画质重要，画质只在对外展示时才有意义。

---

## 第 3 步：亲眼看显存（约 5 分钟）

开**第二个**终端窗口（Terminal 里 Ctrl+Shift+T），输入：

```
nvidia-smi --query-gpu=memory.used --format=csv -l 1
```

它每秒打印一次显存占用。回到第一个终端重复第 2 步的命令，观察第二个窗口的数字变化（约 200 MiB 波动，跑完回落）。看完按 Ctrl+C。

**为什么必须这样测**：`torch.cuda.max_memory_allocated()` 只统计 PyTorch 自己分配的内存，而 MuJoCo Warp 用自己的显存池——之前正是这个坑让"显存只用了 2 MiB"这种假数字出现过。**不同工具管各自的账本。**

---

## 第 4 步：改一个参数，先预测再验证（约 10 分钟）

把 `--num-envs` 依次改成 `1` 和 `4`，各跑一次（其余不变）。每次记录墙钟耗时（命令前加 `time ` 即可）。

**先写下你的预测**：环境数 1 → 2 → 4，耗时会不会成比例（1 倍 → 2 倍 → 4 倍）？

（答案：不会。耗时主要是固定开销——渲染、策略前向、Python 循环——多个环境在 GPU 上被一起算。这也解释了训练时 512 → 4096 环境只慢 2.2 倍。）

把三次数字记下来，这是你亲手采的第一组实验数据。

---

## 第 5 步：读三处代码（约 10 分钟）

在项目根目录执行 `code .`（用 VS Code Remote-WSL 打开项目）。

1. `scripts/replay_policy.py` 第 1–60 行：`load_env_cfg` → 改 `num_envs`/`motion_file` → 建 `ManagerBasedRlEnv` → `runner.load` → 循环 `env.step`。"加载预训练策略并跑闭环"的全部就在这里。
2. `repos/mjlab_research/src/mjlab/tasks/tracking/mdp/commands.py` 第 585–615 行：找 `ghost` 和 `ghost_color`。这就是视频里那个绿色虚影——参考姿态，不是机器人。
3. `repos/mjlab_research/src/mjlab/tasks/tracking/config/g1/env_cfgs.py`：搜 `anchor` 和 `terminations`，这里定义"跟踪哪些身体、什么时候算失败"。

读完回答一个问题：**想让画面里多一个机器人，是哪个变量？**（你已经改过它了）

---

## 第 6 步：用 Python 读指标（约 5 分钟）

```
environment/.venv/bin/python -c "import json; d=json.load(open('runs/20261112-001-eval/summary.json')); print(d['mpkpe_mean_cm'], d['r_mpkpe_mean_cm'])"
```

**你应该看到**：`10.738 3.42`。summary.json 就是键值对，`d['键名']` 取值，不需要我来转述。

---

## 交付给我的三件事

1. 第 4 步三次耗时（1 / 2 / 4 环境），以及你的预测对不对。
2. 第 3 步看到的显存范围。
3. 任何一步卡住时的**完整报错**（别删，别只贴最后一行）。

## 常见卡住

- 报 GLFW 或 `cannot open display` → 忘了 `MUJOCO_GL=egl`。
- 报 `FileNotFoundError` → 检查是否在项目根目录（`pwd`）。
- 卡在 kernel 日志很久 → 首次编译，正常，等它。
- 视频目录为空 → 看命令最后是否打印输出路径，再 `ls outputs/videos/lab1`。
