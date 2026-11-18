# LAB11：终止到底是不是倒地——零 GPU 姿态审计

> 承接：LAB10（M5 扰动）与 `runs/20261118-002-m5-video-check`。
> 目标：把"4 个视频里看到的"扩展成"80 次 rollout 的统计"，并看清一件事——**M5 里被叫做"终止"的东西，其实不只一种**。
> 成本：0 GPU，纯读日志，2 分钟；不需要改动任何配置。

## 1 为什么值得做

M5 报告里，12 格粗扫 + 5 档细化一共 80 次评估，写的是"终止/失稳"。但：

- `fall_detected`（骨盆 < 0.15 m 连续 5 步）对 80 次全是 False —— 它只认"贴地平躺"；
- 视频核对的 4 例显示：3 个终止格在终止前 0.04–0.09 s 已经跪坐/手撑地，1 个未终止格蹲到 0.506 m 又站回来。

于是问题变成：**41 次提前终止里，到底多少次是真的姿态崩了？** 如果大部分是"还站着但误差超阈值"，那报告里"高推力 → 全部失稳"的说法就站不住。

## 2 判据（脚本已经写好，你只读不改）

`scripts/m5_posture_audit.py` 对每个 rollout 只取**第一次 reset 之前**的帧，算两个量：

| 量 | 定义 | 含义 |
| --- | --- | --- |
| `end_drop` | 复位前 0.4 s 内 `ref_pelvis_z - pelvis_z` 的均值 | 骨盆比参考低多少（米）|
| `end_tiltex` | 同一窗口内 `tilt_deg - ref_tilt_deg` 的均值 | 躯干比参考歪多少（度）|

分类：

| 类别 | 条件 | 含义 |
| --- | --- | --- |
| `survived` | 第一次 reset 只有 `motion_end` | 跑完 20 s 动作 |
| `collapse` | 提前 reset 且 `end_drop >= 0.08 m` 或 `end_tiltex >= 13 deg` | 复位瞬间姿态已崩 |
| `upright_trip` | 提前 reset，但两项都没到 | 复位瞬间还站得住，属误差判罚；**之后会不会倒未知** |

阈值 8 cm / 13° 是从 4 例录像的分界里取的中间值：真倒的末段是"低 10–12 cm 或歪 15–24°"，撑住的是"低 0–0.3 cm、歪 3.4–3.8°"。

脚本每次运行都会拿这 4 例自检（`py_j30` s101、`py_j60` s101、`py_j27` s103 应为 collapse，`py_j27` s101 应为 survived），不一致就返回非零——这是防止"改了判据却没发现"的护栏。

## 3 先写预测（别偷看 canonical 结果）

写到 `docs/learning/LAB11-prediction.md`，再跑命令：

**Q1** 80 次 rollout 里，`survived` / `collapse` / `upright_trip` 各占多少？（给个量级，比如 40/20/20）

**Q2** 41 次提前终止里，哪一类方向最容易变成 `collapse`？为什么？

**Q3** 120 N·s（3.60 m/s）这种最强推力，复位那一刻机器人更可能"已经倒了"还是"还站着"？为什么？

**Q4** 如果 `upright_trip` 占多数，M5 报告里"高推力下全部终止 = 全部失稳"这句话要怎么改？

## 4 跑

**A. 复现 canonical（应该和 README 完全一致）**

~~~bash
cd /home/overlume/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
environment/.venv/bin/python scripts/m5_posture_audit.py \
  --runs-dir runs \
  --patterns m5-perturb m5-threshold-refine \
  --out scratch/lab11-repro
diff <(cut -d, -f1-4,6-9 runs/20261118-003-posture-audit/posture_audit.csv) \
     <(cut -d, -f1-4,6-9 scratch/lab11-repro/posture_audit.csv) && echo "REPRODUCED"
~~~

**B. 读结果**

~~~bash
sed -n '1,50p' runs/20261118-003-posture-audit/posture_audit.md   # 总计 + 视频校验 + 分格
sed -n '50,140p' runs/20261118-003-posture-audit/posture_audit.md  # 逐次明细
~~~

## 5 怎么读（顺序不要乱）

1. 先看**总计**：三类各多少。这一行就决定了后面所有话该怎么说。
2. 再看**视频校验表**：4 例必须全"一致"，否则脚本判据不可信。
3. 再看**分格表**的 collapse 列：哪些格出现了真倒？只有 +y 吗？
4. 最后看**逐次明细**里的 `dt_after_push`（推后多久复位）：如果某格是 0.14–0.30 s，那它几乎不可能已经倒地——人形从站立到倒地需要约 0.4–0.8 s。

## 6 常见坑

| 坑 | 说明 |
| --- | --- |
| 用了复位之后的帧 | reset 之后是新回合（站姿），会把"倒地"平均掉，必须截断在第一次 reset |
| 把 `upright_trip` 当"安全" | 它只是"复位那一刻还站着"；回合在那里结束，3 s 后倒没倒不知道 |
| 把 `collapse` 当"摔成平躺" | 本项目里是跪坐/坐倒/手撑地，骨盆 0.42–0.55 m，不是平躺 |
| 只信阈值 | 8 cm / 13° 是操作阈值；+y 60 的 14.7° 与 `upright_trip` 的 11.4° 只差 3°，边界格要复数种子看 |
| 以为这步能回答"倒没倒" | 只能回答"复位那一刻姿态崩没崩"；"之后会不会倒"需要 LAB12 的 `--ignore-terminations` 恢复实验 |

## 7 通过标准

- [ ] 自己那份 `scratch/lab11-repro/posture_audit.csv` 与 canonical 逐行一致（`REPRODUCED`）
- [ ] 能说出三类各多少次，并指出 collapse 只出现在哪些格
- [ ] 四道预测题都答了，且能说出 Q3 的物理理由（倒地需要时间；0.14–0.30 s 不足以倒下）
- [ ] 能用一句话说清"终止 ≠ 倒地"，并给出改进方向（放宽终止 / 跑恢复段）

## 8 结果记录

跑完后把你的判分写进 `docs/learning/LAB11-prediction.md` 末尾（我判），并把 canonical 的路径记到 `docs/PLAN.md` 的 M5 行。canonical 结果已经在 [runs/20261118-003-posture-audit/README.md](runs/20261118-003-posture-audit/README.md) 里，含完整分格表与局限。
