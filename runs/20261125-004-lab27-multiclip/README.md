# LAB27 多片段联合训练（S3 0 s + dance1 0 s）

## 1. 目的与假设

- 目的：单片段训练是不是 held-out 只有 0.100 的成因？给策略第二段动作后，held-out / unseen 会不会动。
- 假设：held-out 涨到 ≥0.15（Δ ≥ +0.05）且 ≥7/10 种子同向 = 数据是瓶颈之一。
- 设计：训练 = S3 0–20 s（原样，999 帧）+ dance1 0–20 s（地面校正后，1000 帧）；其余全同 M4（热启动 scratch/model_49999.pt、3000 iter、4096 envs、seed 0、目标 iter 52998、DR 默认、160 维观测）。任何差异只能来自数据。

## 2. 协议核对

| 项 | 值 | 核对方式 |
|---|---|---|
| 热启动 | scratch/model_49999.pt sha f1bc404f…f22319 | sha256sum |
| 片段 1 | data/processed/fight1_s3_000s_20s.npz sha 965f2c4b…b66c66（999 帧） | sha256sum |
| 片段 2 | data/processed/dance1_000s_20s_gfix.npz sha a5961085…2bdc（1000 帧） | sha256sum |
| 训练动作 | packed 目录 motion_packed：motion_ends = [999 1999]、joint_names 29 个、joint_pos (1999, 29) | np.load |
| 训练配置 | 4096 envs、3000 iter、agent.seed 0、save_interval 500 | logs/…/params/env.yaml、agent.yaml |
| ckpt | model_52998.pt sha **8745f96ee2136e9187b8c2441812d72d95de125f846b147af14c76d680125dac** | sha256sum（已排除 pretrained/ 副本） |
| 评估 | scripts/eval_tracking.py，终止检测开、启动 DR 关、999 步；种子 101–110 | eval/<policy>/<window>/seed*/summary.json |

地面校正（Part A，D41 闸门 max ≤20 mm 且 p99 ≤20 mm）：

| 窗口 | 穿地 max 前 → 后 | p99 前 → 后 | 结论 |
|---|---|---|---|
| dance1 0–20 s | 32.9 → **3.5 mm** | 14.8 → 0.8 mm | PASS |
| dance1 60–80 s | 9.9 → **5.9 mm** | 0.5 → 2.5 mm | PASS |

（60–80 s 的片段 sha 5dbd4a04…bf57。）

## 3. 时间与成本

- 训练：2026-11-25 18:15:01 → 19:58:48 = **1 h 43 min 47 s**（单段跑完，无续训）；每 500 iter 一个 ckpt，从 model_50000 到 model_52998。
- 评估：80 次（2 策略 × 4 窗口 × 10 种子）≈50 min（按 LAB21 实测 37.6 s/次估算）。
- 合计 ≈2 h 34；run 目录 75 MB（logs 63 MB 是 ckpt，不入库）。

## 4. 结果

| 窗口 | M4（单片段） | LAB27（双片段） | 配对 Δ | 存活帧 M4 → LAB27 |
|---|---|---|---|---|
| 训练片段 S3 0 s | 0.999 ± 0.000 | 0.999 ± 0.000 | ±0.000（10/10 相同） | 998 → 998 |
| held-out S3 120 s | 0.099 ± 0.002 | 0.097 ± 0.001 | −0.003（0 好 / 0 差 / 10 平） | 99 → 96 |
| unseen S2 134 s | 0.025 ± 0.001 | 0.026 ± 0.001 | +0.001（10/10 平） | 25 → 26 |
| dance1 60–80 s | 0.061 ± 0.009 | **0.108 ± 0.029** | **+0.046**（9 好 / 0 差 / 1 平） | 61 → 108 |

dance 窗口逐种子（M4 → LAB27）：s101 0.063→0.072、s102 0.069→0.096、s103 0.065→0.104、s104 0.065→0.089、s105 0.065→**0.175**、s106 0.065→0.087、s107 0.044→0.105、s108 0.067→0.147、s109 0.067→0.103、s110 0.041→0.098。

误差两个口径（同一批 summary.json）：

| 窗口 | MPKPE M4 → LAB27 | root 误差 M4 → LAB27 | reset 次数 M4 → LAB27 |
|---|---|---|---|
| train | 5.65 → **7.51 cm** | 3.86 → **6.04 cm** | 1.0 → 1.0 |
| held-out | 18.71 → 10.16 cm | 12.89 → 4.66 cm | 23.5 → **40.3** |
| unseen | 17.73 → 21.69 cm | 14.85 → 19.90 cm | 48.6 → 40.5 |
| dance | 17.12 → 19.91 cm | 14.63 → 16.96 cm | 24.4 → 12.8 |

## 5. 机制

1. **dance 家族的提升是真的，但只限这一族**：dance1 60–80 s 在训练里从未出现，却从 0.061 翻到 0.108（9/10 种子更好），存活 61 → 108 帧、reset 24.4 → 12.8 —— 说明策略真的从 dance1 0–20 s 学到了可复用的东西，只是这份「学到」没能迁移到 fight1 S3 120 s 或 S2。
2. **train 的精度代价明显**：完成度 0.999 不变，但 MPKPE 5.65 → 7.51（+33%）、root 3.86 → 6.04（+56%）→ 两段动作共享同一策略容量，第二段分走了表达能力。
3. **held-out 的误差改善是假象**：MPKPE 18.71 → 10.16 只因存活更短（reset 23.5 → 40.3 次、存活 99 → 96 帧），跨策略比误差必须先对齐存活（D42 决定 2）。
4. **unseen 更糙**：MPKPE +3.96 cm、root +5.05 cm，完成度 +0.001 是「多撑半帧」级别的噪声。
5. **距离效应只是提示**：dance 60–80 s 距训练窗口 40–60 s（0.108），held-out S3 120 s 距训练窗口 120 s（0.097），但两者动作不同，不能直接当距离证据 → 需要同动作的窗口序列（LAB28）。

## 6. 判读

1. 属 **(B) 只涨训练动作家族**：dance1 的另一窗口涨 +0.046，held-out / unseen 持平。
2. 「多片段训练提升泛化」在本轮**不成立**；引用时必须带「只提升训练动作家族的另一窗口」这个限定。
3. 完成度会低估代价：train 完成度不变而 MPKPE +33%、root +56%（与 D41 同一教训）。
4. 数据侧扩容的下一步不是加更多段，而是先量「同一段动作上距训练窗口多远开始失效」（LAB28）。
5. 本 ckpt 不替代 M4 作主力（train 精度明显更差）；继续扩数据要同时报 train 的 MPKPE / root。

## 7. 局限

- 只有 1 个训练 seed（LAB25 已证明换 seed 能把 65% 变 23%）。
- 第二段只有 20 s，且仍是同一批重定向数据（多样性没变，只是加了一段）。
- 50/50 的配比没做消融；两段等长只是选择。
- held-out 与 dance 两格动作不同，不能把 0.097 vs 0.108 当距离效应的证据。
- 没叠 LAB12 抗推协议，不知道多片段后鲁棒性是否变化。

## 8. 文件清单

- README.md（本文件）、eval_summary.md / eval_summary.json（runs/<run>/* 才入库的那两份）
- eval/（80 个 seed 目录 + lab27_eval.md|json|png，不入库）
- motion_packed/（root_pos_w.npy 等 6 个字段 + motion_ends.npy + joint_names.json + motion_names.json）
- logs/g1_tracking/2026-11-25_18-15-01_lab27-multiclip/（params/、model_50000…52998.pt、tfevents）
- 片段（不入库，.gitignore 忽略 data/processed）：dance1_000s_20s_gfix.npz sha a5961085…2bdc、dance1_060s_20s_gfix.npz sha 5dbd4a04…bf57

## 9. 复现

```bash
# Part A：地面校正 + 闸门
environment/.venv/bin/python scripts/lab27_ground_fix.py \
  --src data/processed/lafan_dance1_subject1.npz --start-s 0 --dur-s 20 \
  --out data/processed/dance1_000s_20s_gfix.npz
environment/.venv/bin/python scripts/lab27_ground_fix.py \
  --src data/processed/lafan_dance1_subject1.npz --start-s 60 --dur-s 20 \
  --out data/processed/dance1_060s_20s_gfix.npz

# Part B：多片段训练（可中断，重跑同一条即续训）
environment/.venv/bin/python -u scripts/lab27_multiclip_train.py \
  --run-root runs/20261125-004-lab27-multiclip --mode full 2>&1 | tee runs/20261125-004-lab27-multiclip/train.log

# Part C：评估 + 出表
LAB27_RUN=runs/20261125-004-lab27-multiclip bash scripts/lab27_eval.sh run
```

核对点：ckpt sha 应为 8745f96e…125dac；motion_ends 应为 [999 1999]；eval 下应有 80 个 summary.json。
