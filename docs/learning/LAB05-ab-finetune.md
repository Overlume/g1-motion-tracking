# LAB05 — 亲手做一个"只改一处"的训练对照（M3-b）

> 前置：LAB04 已完成（你已经跑过一次 512 env × 100 iter 的冒烟训练，见过 Iteration time / Mean reward / summary.json）。
> 这一节对应 PLAN 里 M3 的第三条验收："用户亲手改一个有明确意义的参数，先预测后对比"。

## 1. 要回答的问题

官方预训练策略在 **fight1_s3** 片段上 5 个种子全摔（中位 11.66 s）。我们想知道：

> 把"根相对世界的水平位置"这一项奖励调强、调尖之后，从官方权重续训 300 次迭代，
> S3 上的摔倒率 / 根漂移会不会变好？

**只改一个奖励项**，其他全部不动——这是对照实验的全部意义。

## 2. 改的是哪一项，数学上意味着什么

文件：`repos/mjlab_research/src/mjlab/tasks/tracking/tracking_env_cfg.py` 第 210–214 行

```python
"motion_global_root_pos": RewardTermCfg(
  func=mdp.motion_global_anchor_position_error_exp,
  weight=0.5,                                    # -> 1.0
  params={"command_name": "motion", "std": 0.3}, # -> 0.15
),
```

奖励函数的实现（`tasks/tracking/mdp/rewards.py:26-33`）：

```python
error = sum((anchor_pos_w - robot_anchor_pos_w) ** 2)   # 根（anchor）世界位置误差的平方和
return exp(-error / std ** 2)
```

- **weight**：这一项在总奖励里的权重（每步再乘 dt=0.02）。0.5 → 1.0 就是这份"位置奖励"的最大贡献翻倍。
- **std**：高斯核的宽度。std 0.3 → 0.15，等价于把"可容忍的位置误差"**减半**：误差 10 cm 时奖励从 exp(-0.111)=0.89 掉到 exp(-0.444)=0.64；误差 20 cm 时从 0.64 掉到 0.17。
- 注意：`std` 变小不只是"更严格"，也让**误差大时奖励趋近 0**——误差 30 cm 时几乎拿不到任何梯度信号。这就是为什么这件事值得做实验，而不是拍脑袋"调强一定更好"。

## 3. 先写预测（必须，在跑之前）

在 `docs/learning/LAB05-prediction.md` 里写下你的预测与理由（三句以内即可）：

1. 相对 A 组，B 组在 S3 上的**摔倒次数**会变多、变少还是不变？为什么？
2. B 组的**根漂移速率**（drift_rate_cm_s）会变好还是变差？
3. 训练日志里 `Metrics/motion/error_anchor_pos` 这一项，B 组应该比 A 组更大还是更小？为什么？

第 3 题是个陷阱题：奖励变尖以后，**奖励值**当然会变小，但**物理误差**未必。把这两个分清楚，是这一节最重要的收获。

## 4. 动手（WSL 里，项目根目录）

一次总耗时约 30 分钟（2 × 9-10 min 训练 + 10 次 × 约 1 min 评估）。

### 4.1 跑 A 组（原始奖励，对照组）

```bash
cd <workspace>
bash scripts/lab05_ab_finetune.sh train A
```

- 默认 4096 env × 300 iter，seed 0，动作 `data/processed/fight1_s3_000s_20s.npz`，从 `scratch/model_49999.pt` 续训 300 次（49999 → 50298）。
- 脚本会先校验 config 的 sha256 必须是原始值（`f756b4d0…b9a82`），否则直接拒绝——防止你拿着改过的配置跑"对照组"。
- 产物：`runs/YYYYMMDD-NNN-m3b-a-finetune/`（command.txt、params.txt、cfg_snapshot.py、train.log、vram_mib.txt、logs/…/model_50000/50100/…/50298.pt）。
- 训练时另开一个终端看：`nvidia-smi -l 1`（看显存和 GPU 利用率），以及 `free -h`（LAB04 你忘了看的那个）。

### 4.2 亲手改两行

把第 212 行 `weight=0.5,` 改成 `weight=1.0,`；把第 213 行里的 `"std": 0.3` 改成 `"std": 0.15`。

改完自己检查一遍（应该只差这两行）：

```bash
diff -u runs/*-m3b-a-finetune/cfg_snapshot.py repos/mjlab_research/src/mjlab/tasks/tracking/tracking_env_cfg.py
```

### 4.3 跑 B 组（修改后奖励，实验组）

```bash
bash scripts/lab05_ab_finetune.sh train B
```

- 脚本会做三道校验：与 A 组快照 diff **只能有 2 行增减**；这 2 行必须恰好是 weight 与 std；否则拒绝开跑。这是"一次只改一个因素"的机器化执行。
- 产物：`runs/YYYYMMDD-NNN-m3b-b-finetune/`，另含 `diff_A.txt`。

### 4.4 评估 + 汇总

```bash
bash scripts/lab05_ab_finetune.sh eval     # A/B 各 5 个种子（101-105），termination 保持开启
bash scripts/lab05_ab_finetune.sh report
```

评估走已有的 v2 协议（`scripts/eval_tracking.py`）：1 env、逐帧记 done、分段、摔倒按 pelvis 高度判定。按 D11，**只看分布**：摔/不摔次数、摔倒时间中位与范围、完成度、root_err、drift_rate 的中位与范围。

`report` 会在 `runs/…-m3b-eval/` 下生成 `comparison.md` 与 `comparison.json`，训练侧的 error_anchor_pos / 终止次数也一并列出。

## 5. 跑完看什么

1. **A 组能不能续训**：日志里应有 `Loading model checkpoint from: …/pretrained/model_49999.pt` 和 `Learning iteration 49999/50299`。第一次迭代慢（编译 + 载入），稳态约 1.8–1.9 s。
2. **B 组 diff_A.txt**：应该只有两行 `-` / `+`。
3. **比较表**：重点不是"谁的中位数大"，而是**两组分布是否重叠**。5 个种子的 min–max 若大面积重叠，正确结论是"300 iter 尺度上不可分辨"。
4. **训练侧物理量**：`error_anchor_pos` 与 `Episode_Termination/anchor_pos` 是奖励函数之外的物理量，可以横比。
5. **显存**：4096 env 预期约 1.9 GB（LAB04 的 512 env 是 505 MiB），远低于 6 GB。

## 6. 收尾：把配置还原（**别忘了**）

对照实验做完，config 必须回到原始状态，否则后续所有训练都偷偷带着"1.0 / 0.15"：

```bash
cp runs/*-m3b-a-finetune/cfg_snapshot.py repos/mjlab_research/src/mjlab/tasks/tracking/tracking_env_cfg.py
sha256sum repos/mjlab_research/src/mjlab/tasks/tracking/tracking_env_cfg.py   # 应等于 f756b4d0…b9a82
```

`scripts/lab05_ab_finetune.sh` 的 `train A` 也会替你守住这条线（sha 不对就拒绝），但收尾还原仍然要你亲手做一次。

## 7. 常见坑

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `train A` 报 config sha256 不是 A 态 | 你上一轮改过没还原 | 按第 6 节还原，再跑 |
| `train B` 报 `期望只改 2 行` | 顺手改了别的（比如 std 别的项） | 看 /tmp/lab05_diff.txt，只保留这两行 |
| `train B` 报 `config 与 A 组完全相同` | 改了没保存 / 改错文件 | 检查 `sed -n '210,215p'` 那个文件 |
| 日志里 `Iteration time: -1.65s` | WSL2 墙上时钟跳变，不是训练出错 | 忽略负值样本；report 已统计其个数 |
| 评估很慢 | 每次评估都要重新编译一次内核/图 | 正常，单次约 40–60 s |
| `eval` 找不到 `model_50298.pt` | 训练没跑完 | 检查 train.log 末尾与退出码 |

## 8. 通过标准

- [ ] A、B 两个 run 目录齐全，`params.txt` 里除 cfg_sha256 外全部相同
- [ ] `diff_A.txt` 只有 2 行改动
- [ ] 两组的 `model_50298.pt` 都能被 `scripts/replay_policy.py` 读回（可选，一条命令即可）
- [ ] `comparison.md` 有 5 个种子的分布，且你的结论没有拿单次 fall_step 说事
- [ ] config 已还原，sha256 与原始值一致


## 9. 结果（2026-11-15 实测，2026-11-16 补写）

完整判读在 `runs/20261115-002-m3b-eval/README.md`，这里只放要点。

| 指标（5 种子 101-105） | A（原奖励） | B（weight 1.0 / std 0.15） |
| --- | --- | --- |
| 摔倒 | 0/5 | 0/5 |
| 完成度 | 0.999 | 0.999 |
| root xy 误差 中位 cm | 43.34 | 52.72 |
| 漂移速率 中位 cm/s | 0.06 | 0.70 |
| R-MPKPE 中位 cm | 5.707 | 6.016 |
| 训练 error_anchor_pos（末 30 iter） | 0.386 m | 0.508 m |

两件事值得记住：

1. **300 iter 就不摔了，但不等于学会。** 官方预训练策略在 S3 上 5/5 摔倒，续训 300 iter（官方预算的 0.6%）后两组都跑满 999 步；可 root 误差 43/53 cm，只有根相对姿态 R-MPKPE 5.7/6.0 cm 还不错。看结果必须同时报“摔不摔”和“跟得准不准”。
2. **把 std 从 0.3 收到 0.15 反而更差。** 奖励核 `exp(-d^2/std^2)` 在 d≈0.45 m 处的梯度幅值从 ≈1.05 掉到 ≈4.9e-3，权重 x2 补不回来——核宽决定“信号有没有”，权重只决定“信号多强”。以后调这个奖励，先保 std 与期望误差同量级，只动 weight。

三道预测的判分：Q1 结论对（0/5 vs 0/5，但理由不完整——真正该预测的是“两组都不摔了”）；Q2 错（0.06 -> 0.70 cm/s）；Q3 错（0.386 -> 0.508 m）。Q2 与 Q3 其实在量同一件事，同方向错说明是一个判断错误，不是两处笔误。

下一步见 `docs/PLAN.md` 的“下一步（二选一）”。
