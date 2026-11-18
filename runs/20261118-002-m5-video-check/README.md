# 20261118-002 M5 终止格视频核对（4 例）

## 目的

M5 报告里写“终止/失稳”，但 `fall_detected` 全为 False，两者矛盾。本 run 用录像直接看：终止格在终止瞬间到底是“站着但误差大”，还是“已经倒了”。

抽 3 个终止格 + 1 个未终止对照：

| 案例 | 方向/冲量 | eval 终止项与时刻 | 备注 |
| --- | --- | --- | --- |
| `py_j30_s101` | +y 150 N（30 N·s） | `anchor_pos` @8.62 s | M5 粗扫里 +y 的第一档终止 |
| `py_j60_s101` | +y 300 N（60 N·s） | `ee_body_pos` @8.48 s | 另一种终止项 |
| `py_j27_s103` | +y 135 N（27 N·s） | `ee_body_pos` @10.46 s | 边界格：延迟 2.3 s 才终止 |
| `py_j27_s101` | +y 135 N（27 N·s） | 无（存活） | 对照：未终止是什么样 |

## 方法

- 现有 `scripts/replay_policy.py` 原来不支持外力，本 run 给它加了 `--perturb-body/--perturb-force/--perturb-start-s/--perturb-duration-s`（与 `eval_tracking.py` 同一套 `write_external_wrench_to_sim` 口径）。
- 其余条件与 eval 完全一致：同一 ckpt（`model_52998`）、同一 motion、同一 seed、DR 关、终止检测开、8.0–8.2 s 施加同样力。
- 录像：`--video-length 600`（12.02 s）、1280×720、`--camera follow --camera-margin 1.05 --camera-min-distance 2.6`。**只为本 run 看清姿态临时收紧取景，不改脚本默认值**（默认仍是 margin 1.3 / min-distance 5.2）。

## 复现一致性（先确认视频看的就是 eval 的那一局）

| 案例 | eval `first_reset_after_s` | 录像里的 reset | 差 |
| --- | --- | --- | --- |
| `py_j30_s101` | 8.62 s | 8.64 s（step 432，`anchor_pos`） | 0.02 s = 1 帧 |
| `py_j60_s101` | 8.48 s | 8.48 s（step 424，`ee_body_pos`） | 0 |
| `py_j27_s103` | 10.46 s | 10.48 s（step 524，`ee_body_pos`） | 0.02 s = 1 帧 |

终止项、终止时刻都对得上（1 帧差来自评估脚本 `(step+1)/50` 的取整），可以认为录像与评估是同一局。

## 观察

![j30 终止格](outputs/figures/evidence/m5_fall_check_py_j30_s101.jpg)

`py_j30_s101`：8.55 s（终止前 0.09 s）已经跪坐在地、双手撑地；8.62 s 是 reset 后的新回合（站姿、双臂外展）。

![j60 终止格](outputs/figures/evidence/m5_fall_check_py_j60_s101.jpg)

`py_j60_s101`：8.44 s（终止前 0.04 s）坐倒，双腿前伸、上身后仰；8.60 s 起是新回合。

![j27 终止格](outputs/figures/evidence/m5_fall_check_py_j27_s103.jpg)

`py_j27_s103`：9.6–10.0 s 还是深蹲前倾（未倒），10.30 s 手撑地，10.42 s 四肢着地，10.55 s 起是新回合——这条与“延迟 2.3 s 才终止”吻合：它先撑了一会儿才被 `ee_body_pos` 判定失败。

![j27 存活对照](outputs/figures/evidence/m5_fall_check_py_j27_s101.jpg)

`py_j27_s101`（未终止）：8.60 s 蹲到极低、手接近地面，9.2–10.4 s 一直低位踉跄，但没有倒下，最后站回来（相机在 11 s 后跟不上，机器人出画）。

## 数值对照（同一批 rollout 的 `metrics.csv`）

| 案例 | 窗口内 `pelvis_z` 最低 | 参考动作同一时刻 | 最大倾角 | `fall_detected` |
| --- | --- | --- | --- | --- |
| `py_j30_s101` | 0.425 m | 0.636 m | 27.5° | False |
| `py_j60_s101` | 0.546 m | 0.636 m | 39.8° | False |
| `py_j27_s103` | 0.524 m | 0.581 m | 53.0° | False |
| `py_j27_s101` | 0.506 m | 0.581 m | 31.1° | False |

倒地判据是 `pelvis_z < 0.15 m` 连续 5 步（`scripts/eval_tracking.py:41-42`）。跪坐/手撑地时骨盆还有 0.42–0.55 m，**所以永远触发不了**。

## 结论

1. **3 个终止格在终止前 0.04–0.09 s（j27 是 0.06 s）已经跪倒/坐倒/手撑地**——终止不是“站着但误差大”，它就是失败姿势本身。
2. **`fall_detected` 不可靠**：它只认“贴地平躺（骨盆 <0.15 m）”，对“跪倒/坐倒/手撑地”全部漏报。之前的 D20 说“终止≠倒地”是错的，应改成“**终止≈倒地（跪/坐），但 `fall_detected` 只覆盖平躺**”。
3. **未终止格确实没倒**：`py_j27_s101` 蹲到 0.506 m、倾角 31°、踉跄 2 s 后站回，与“未终止但偏离 90 cm”的描述一致——它是“差点倒但撑住了”，不是“没被打到”。
4. 因此报告用词：终止格可以写“失稳倒地（跪/坐姿，非平躺）”并给视频证据；不能写“摔倒”同时引用 `fall_detected`，也不能写“只是跟踪终止、机器人还站着”。

## 局限

1. 只核对了 4 例（3 终止 + 1 存活），不足以给出“终止格 100% 倒地”的统计断言。
2. `camera follow` 在 11 s 后滞后，`py_j27_s101` 最后 1 s 机器人出画。
3. 没有做全身接触点/足底接触分析，姿态判断依赖目视。
4. 视频 12 s 截断，没有看到 j27_s103 的“复位后”行为。

## 复现

```bash
CK=runs/20261117-001-m4-w15-finetune/logs/g1_tracking/2026-11-17_09-25-17_m4-w15/model_52998.pt
MUJOCO_GL=egl environment/.venv/bin/python scripts/replay_policy.py \
  --checkpoint-file "$CK" --motion-file data/processed/fight1_s3_000s_20s.npz \
  --seed 101 --num-envs 1 --video-length 600 --width 1280 --height 720 \
  --camera follow --camera-margin 1.05 --camera-min-distance 2.6 \
  --perturb-body torso_link --perturb-force 0,150,0 \
  --perturb-start-s 8.0 --perturb-duration-s 0.2 \
  --out-dir scratch/m5video-close --tag py_j30_s101

# 抽帧 + 拼图
ffmpeg -y -ss 8.55 -i scratch/m5video-close/model_52998/1280x720/py_j30_s101/rl-video-step-0.mp4 -frames:v 1 frame.png
```

## 脚本改动

`scripts/replay_policy.py`：新增 4 个 `--perturb-*` 参数，逐步在 `env.step` 前调用 `robot.write_external_wrench_to_sim`；`episode_log.json` 里记录 `perturb` 字段。默认不施加外力，原有回放行为不变。
