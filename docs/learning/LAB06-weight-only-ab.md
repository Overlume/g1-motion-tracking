# LAB06 —— 只改一个权重：`motion_global_root_pos` 0.5 -> 1.5（std 保持 0.3）

预计 35 分钟（训练 2 x 300 iter ≈ 21 min，评估 10 次 ≈ 10 min，其余是你的思考时间）。

## 1. 为什么还要做一次 A/B

LAB05 一次改了两个参数（weight 0.5 -> 1.0 且 std 0.3 -> 0.15），结果 B 组更差（root 误差 43.34 -> 52.72 cm）。
我们在 `docs/DECISIONS.md` 的 D13 里给了一个机制解释：**std 收窄把奖励核的“作用域”掐死了**，权重加倍补不回来。但那只是解释，不是实验证据 —— 严格说，LAB05 分不开“weight 变大”和“std 变小”各自的贡献。

这个 LAB 就是那个缺失的对照：**只动 weight，std 一动不动**。

| 项 | LAB05 | LAB06（本次） |
| --- | --- | --- |
| weight | 0.5 -> 1.0 | 0.5 -> **1.5** |
| std | 0.3 -> 0.15 | **0.3 不变** |
| 改动行数 | 2 行 | **1 行** |
| 规模 | 4096 env x 300 iter | 同 |
| 动作 / 种子 / 评估 | S3 / seed 0 / 101-105 | 同 |

## 2. 要改的那一行

文件：`repos/mjlab_research/src/mjlab/tasks/tracking/tracking_env_cfg.py`，第 210-214 行。现在是：

```python
    "motion_global_root_pos": RewardTermCfg(
      func=mdp.motion_global_anchor_position_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.3},
    ),
```

**只把 `weight=0.5,` 改成 `weight=1.5,`。** `params` 那一行的 `std: 0.3` 一个字符都别动。

为什么用 1.5 而不是 1.0：1.0 在 LAB05 里已经和“std 减半”绑在一起，没法复用；1.5 是 3 倍，比 LAB05 的 2 倍更强，如果“权重有正作用”它应该更明显。改得够大而方向错了，也比改得小到测不出来强。

## 3. 先想清楚：三种可能的结局各说明什么

| 假设 | 内容 | 对 B 的预测（root_err 中位） | 若成立，下一步该做什么 |
| --- | --- | --- | --- |
| H1 核宽塌缩 | LAB05 变差全怪 std，权重本身是有效的 | B 明显好于 A（5-15% 以上），R-MPKPE 不退化 | 回到 weight 调参路线：0.5 -> 1.5 -> 2.5，std 固定在 0.3 |
| H2 观测缺失 | policy 根本看不见自己的世界位置，奖励再大也无从优化 | B 与 A 差别在噪声内（<3%），A/B 分布重叠 | 别调奖励了，去补观测（全球系根位置/根速度进 policy obs） |
| H3 权重过载 | 权重太大会让策略牺牲姿态去追位置 | B 的 root_err 略好但 R-MPKPE / joint_rmse 变差，甚至开始摔 | 记下“收益边际 + 代价”，weight 取 1.0 左右折中 |

这三档是可区分的——前提是 A 组这次也重跑一遍，用来量“同一个配置两次跑差多少”。**A 组重跑本身就是本实验的对照组**，别跳过它直接看 B。

## 4. 你的预测（先写，再跑）

新建 `docs/learning/LAB06-prediction.md`，回答这 4 题，写完再去跑命令：

1. B 组 root xy 误差中位数相对 A 会**变好 / 不变 / 变差**？幅度落在哪个区间：<3%、5-15%、>20%？
2. 如果 weight 1.5 有代价，你认为**哪个指标会先暴露**（R-MPKPE / joint_rmse / 摔倒次数 / 训练 `error_anchor_pos`）？为什么？
3. A 组这次重跑（同 config、同 seed、同 300 iter），root xy 误差中位数与 LAB05 的 **43.34 cm** 会差多少？你判断这次“重置实验”能复现到什么程度？
4. 训练侧 `Metrics/motion/error_anchor_pos`（物理量，不是奖励）B 相对 A 会**变小 / 不变 / 变大**？

第 3 题最容易被忽略：如果 A 重跑得到 47 cm 而不是 43 cm，那么后面“B 比 A 好 8%”这种结论就站不住了。

## 5. 四条命令（项目根目录，WSL 里逐条执行）

```bash
bash scripts/lab06_weight_only.sh train A
# 亲手改 weight=0.5 -> 1.5（只这一行），然后：
bash scripts/lab06_weight_only.sh train B
bash scripts/lab06_weight_only.sh eval
bash scripts/lab06_weight_only.sh report
```

参数含义与 LAB05 完全一样（讲义 `docs/learning/LAB05-ab-finetune.md` 第 4 节）：`train A|B` 后可以跟 `num_envs iters seed`，默认 4096 300 0。

脚本会替你守住三条线：

- A 组要求 config 是原始态（sha256 `f756b4d0…b9a82`）；
- B 组要求与 A 组快照相比**恰好 +1/-1 行**，且 210-214 行里是 `weight=1.5,` 和 `"std": 0.3`；
- 每组跑完核对最后迭代号 = 50298，并记录墙钟、退出码。

新增记录（比 LAB05 多）：`gpu_samples.csv`（每 2 s 一次显存/利用率/温度/功耗）、`gpu_before.txt`、`gpu_after.txt`、`mem_before.txt`、`mem_after.txt`。

## 6. 机器状态：内存比显存更紧张

你昨天补的 `free -h` 快照（训练 A/B 期间）：**总 7.6 GiB、已用 5.5 GiB、可用 1.8 GiB、swap 2.0 GiB 已用 1.5 GiB**；`nvidia-smi`：显存 1865/6144 MiB（够用）、利用率 72%、温度 **86 °C**。

两件事值得记住：

- 显存完全不是瓶颈（只用 30%）；**瓶颈在 CPU/内存侧**：只剩 1.8 GiB 可用，swap 已经用了 1.5 GiB。跑 4096 env 训练时如果开始换页，迭代耗时会抖。
- 86 °C 接近 RTX 3060 Laptop 的降频线（通常 87 °C 左右），功耗 82/93 W、利用率只有 72%，说明机器不是“算力跑满”，而是被 CPU 侧或热限制拖住。**所以跨天比耗时没有意义**，必须像现在这样把温度/内存一起记下来。

开跑前建议：关掉浏览器和其他占内存的程序，`free -h` 确认 available > 3 GiB；不要让笔记本垫在被子/床上。

## 7. 跑完看什么

1. `runs/<日期>-NNN-lab06-eval/comparison.md`：A、B 各 5 个种子的中位与范围。
2. **先比 A 与 LAB05-A**（同配置）：差多少？这是本实验的噪声底。
3. 再比 B 与 A：差幅是否超过噪声底？落在第 3 节哪一档？
4. 看 `R-MPKPE` 有没有退化（H3 的代价信号）。
5. 两个组的训练侧 `error_anchor_pos`（物理量，可跨组比）方向是否与评估一致。

结论写法（三选一，别含糊）：

- “B 比 A 好 X%，超过噪声底，H1 得到支持”；
- “B 与 A 的分布重叠，在 300 iter 尺度上不可分辨，倾向 H2”；
- “B 的 root 误差改善但 R-MPKPE 退化 Y%，是 H3 的折中”。

## 8. 常见坑

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `train A` 报 config sha256 不是 A 态 | 上一轮改过没还原 | 按第 10 节还原 |
| `train B` 报“期望只改 1 行，实际 +2/-2” | 顺手把 std 也改了 | 本实验**不动 std**，改回去 |
| `train B` 报“std 必须还是 0.3” | 想复制 LAB05 的改法 | 见上，LAB06 只动 weight |
| `Iteration time: -1.65s` | WSL2 墙上时钟跳变 | 忽略负值，report 已统计个数 |
| 迭代耗时比 11-12 的 1.76-1.90 s 慢 | 内存吃紧 / GPU 高温 | 看 `gpu_samples.csv` 的温度与显存，别直接怪训练脚本 |
| `eval` 报找不到 `model_50298.pt` | 训练没跑完 | 看 train.log 末尾与退出码 |

## 9. 通过标准

- [ ] `docs/learning/LAB06-prediction.md` 在跑之前就写好了（4 题都答）
- [ ] A、B 两个 run 目录齐全，`params.txt` 除 `cfg_sha256` 外完全一致
- [ ] `diff_A.txt` 只有 1 行改动（weight）
- [ ] `comparison.md` 有 5 个种子；结论里同时出现“A vs LAB05-A 的差”和“B vs A 的差”
- [ ] config 已还原，sha256 = `f756b4d0…b9a82`

## 10. 收尾还原

```bash
cp runs/*-lab06-a-finetune/cfg_snapshot.py repos/mjlab_research/src/mjlab/tasks/tracking/tracking_env_cfg.py
sha256sum repos/mjlab_research/src/mjlab/tasks/tracking/tracking_env_cfg.py   # 应等于 f756b4d0…b9a82
```

## 11. 自测记录（智能体）

2026-11-16 用 64 env x 6 iter 全流程自测（`scratch/lab06-selftest`，gitignored）：A 组 77 s、B 组 51 s（脚本正确识别“只改了 weight”并拒绝其他改动）、eval 2 种子、report 正常出表（标题与口径已参数化）；随后 config 已还原并核对 sha256。

## 12. 结果（2026-11-16 实测，用户跑完）

- 评估（5 种子 101–105，S3，终止检测开）：A root 误差 41.52 (39.82–42.59) cm → B 25.72 (24.63–26.47) cm，**-38.1%**，两组分布不重叠；MPKPE 42.65 → 26.86；R-MPKPE 6.250 → 6.103（-2.4%）；joint_rmse 0.1668 → 0.1677（+0.5%）；tilt 误差 +14.2%、左脚滑移 +24.8%；两组均 0/5 摔倒、完成度 0.999。
- 与 LAB05 合看（root 中位 cm）：LAB05-A 43.34 / LAB06-A 41.52（同配置重跑，**噪声底 ≈4%**）/ LAB05-B 52.72（w1.0 + std0.15）/ LAB06-B 25.72。
- 事前假设：**H1 成立**（收益远超 5–15% 的预期上限），**H2、H3 不成立**（joint_rmse 与 R-MPKPE 都没退化，代价反而是 tilt +14% 与滑移 +25%，量级很小）。
- 预测判分：Q1 方向对、幅度低估 2.5–7 倍；Q2 错（joint_rmse 无变化）；Q3 “可复现”对、30 cm 的数字错（实际 41.52，分布级可复现、位级不可复现）；Q4 对（0.392 → 0.2084 m）。
- 完整判读、机制与 6 条局限：`runs/20261116-003-lab06-eval/README.md`；结论记为 `D15`。
- 由此定下的规则：**std 固定 0.3，要增强根位置跟踪只加 weight**（D13 由猜想升为结论）。
