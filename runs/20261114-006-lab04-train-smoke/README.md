# run 20261114-006-lab04-train-smoke — LAB04 冒烟训练（用户亲手执行）

- 目的：验证 (a) 训练迭代能推进 (b) 参数确实更新 (c) 检查点能读回；为 M3-b 建立耗时/显存基线。**不是**学习效果实验。
- 执行：2026-11-14 16:23:08 +0800；GPU：NVIDIA GeForce RTX 3060 Laptop GPU（driver 616.64）
- 规模：num_envs=512，iterations=100，seed=0，num_steps_per_env=24 → 12288 条样本/迭代
- 动作：data/processed/lafan_dance1_subject1.npz（LAFAN1 dance1，6574 帧 / 131.48 s / 50 fps）
- git：no-commit，未提交改动 7 个文件

## 命令

```bash
MUJOCO_GL=egl environment/.venv/bin/train Mjlab-Tracking-Flat-Unitree-G1 --env.commands.motion.motion-file data/processed/lafan_dance1_subject1.npz --env.scene.num-envs 512 --agent.seed 0 --agent.max-iterations 100 --agent.num-steps-per-env 24 --agent.save-interval 50 --agent.logger tensorboard --agent.run-name lab04 --log-root runs/20261114-006-lab04-train-smoke/logs
```

## 环境

```
[env] 2026-11-14 16:23:08 +0800
python 3.13.18
torch 2.15.0+cu132 cuda_available True
gpu NVIDIA GeForce RTX 3060 Laptop GPU
mujoco 3.16.0
warp 1.19.0
git no-commit dirty_files=7
uname Linux 6.18.33.2-microsoft-standard-WSL2 x86_64
```

## 结果（脚本自动填）

见 summary.json、train.log、vram_mib.txt、iter_times.csv。

## 我的记录（用户手填）

### 预测 vs 实测

| 项目 | 我的预测 | 实测 | 偏差/解释 |
| --- | --- | --- | --- |
| 稳态迭代耗时 s（不含第 0 次） | 1.2s | 0.89s | 0.31s |
| 第 0 次迭代耗时 s | 107.6s | 11.84s | 95.76s |
| 显存峰值 MiB | 300mib | 505mib | 205mib |
| 100 次后 mean reward 走势 | 更高 | 更高 | 一致 |

### 四个问题

1. 512 env 的迭代耗时与 1024 / 4096 env 对比，说明瓶颈在哪？
2. 显存峰值离 6 GB 余量多少？训练时 free -h 的可用内存是多少？
3. 100 次迭代后 mean reward 变化能不能说明学会了？为什么？
4. model_0.pt 与 model_99.pt 的 actor 参数平均变化量是多少？为 0 意味着什么？

### 结论（一句话）
1.答：迭代耗时根本没有按照环境数量的成倍增加而成倍变长，这说明瓶颈在我的环境控制步。
2.答：还有5.5GB的显存余量；但是可用内存我忘记看了
3.答：虽然mean reward 变大了，但是不能说明学会了，它整体还是个负数，说明还是没有取得较好的成果。
4.答：mean |dw| = 4.862017477958183；为0以为这模型权重没有变化，那要么是权重传不回来，要么是已经训练到极致了。

---

## 复盘（智能体补充，2026-11-14 17:20；原文保留不改）

### 1. 第 4 题的 mean |dw| = 4.862 是被污染的

`obs_normalizer.count` 是一个"已见样本数"计数器，从 0 涨到 1,216,506。把它和权重张量一起求平均，平均值就被这一个标量拉到了 4.86。它不代表权重变化。

分项（`model_0.pt` → `model_99.pt`，actor_state_dict，跳过 `*_mean/*_var`）：

| 张量 | shape | mean abs delta | max abs delta |
| --- | --- | --- | --- |
| obs_normalizer._std | (1,160) | 1.18e-01 | 1.07e+00 |
| obs_normalizer.count | () | **1.22e+06** | 1.22e+06 |
| distribution.std_param | (29,) | 1.09e-01 | 1.50e-01 |
| mlp.0.weight | (512,160) | 7.27e-03 | 5.07e-02 |
| mlp.0.bias | (512,) | 1.79e-02 | 4.45e-02 |
| mlp.2.weight | (256,512) | 7.10e-03 | 4.38e-02 |
| mlp.2.bias | (256,) | 1.46e-02 | 3.78e-02 |
| mlp.4.weight | (128,256) | 7.64e-03 | 3.99e-02 |
| mlp.4.bias | (128,) | 1.18e-02 | 3.20e-02 |
| mlp.6.weight | (29,128) | 8.73e-03 | 4.20e-02 |
| mlp.6.bias | (29,) | 1.03e-02 | 3.07e-02 |

**只看策略网络（4 个 weight + 4 个 bias，250,397 个参数）**：mean |dw| ≈ **7.3e-3**，相对量级（mean|dw| / mean|w|）约 15–35%。这已经足以说明"参数确实在更新"，而且 std_param 从 1.00 掉到约 0.89，与日志里 `Mean action std: 1.00 → 0.89` 对得上。

`summary.json` 里的 `0.007368` 是脚本当时的近似口径（没算进 `_std`/`std_param`），也不完整；准确说法是上面的 7.3e-3 + 逐张量表。

**教训**：算"平均变化量"之前，先看每个张量的量级；只要混进一个 counter/step 之类的标量，均值就没有意义。

### 2. iter 19 / 53 / 88 的负耗时不是训练问题

`Iteration time: -1.65s`（以及 `Time elapsed` 倒退）来自 WSL2 墙上时钟偶发回跳，rsl_rl 用 `time.time()` 差分算迭代耗时，于是出现负值。物理步进用的是仿真时间，不受影响。已修 `scripts/lab04_summarize.py`：非正样本不进稳态统计，但会记 `nonpositive_samples` 计数。

### 3. 第 1 题的结论方向对，但理由要再精确一点

实测：512 env 0.89 s、1024 env 1.03 s、4096 env 1.76–1.90 s。env 数 ×8，时间只 ×2 → 线性拟合约 **0.76 s 固定开销 + 0.26 ms/(env·iter)**。也就是 512 env 时约 85% 的时间花在"跟 env 数无关"的部分（PPO 更新、CPU↔GPU 同步、日志、复位），所以"加 env 几乎免费"，瓶颈不在物理仿真步。

### 4. 第 3 题：reward 上升是"活得更短"的假象

Mean reward -2.26 → -0.33，但 Mean episode length 14.25 → 6.98，`Episode_Termination/ee_body_pos` 从 25.1 涨到 44.2（每迭代计数）。回合越短，累积的负奖励（action_rate 等）越少，回报自然变高——这不是学好了，而是更早被终止。所以第 3 题答案对，但证据要加这一条。

### 5. 没记录的两项

`free -h` 可用内存没记（下次训练时补）；显存峰值 505 MiB，6 GB 余量约 5.5 GB。
