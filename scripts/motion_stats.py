"""统计一个 mjlab 动作 npz 的运动学规模（只依赖 numpy）。

用法:
    environment/.venv/bin/python scripts/motion_stats.py data/processed/fight1_s3_000s_20s.npz

输出: 帧数、时长、路径长度、净位移、平均/最大水平速度、x/y 活动范围。
路径长度 = 相邻帧根位置水平位移之和（走来走去会累加）；
净位移 = 首末帧根位置的水平直线距离（回到原点则为 0）。
平均速度 = 路径长度 / 时长，用来和策略能跟上的速度比较（dance1 约 0.60 m/s）。
"""

import sys

import numpy as np


def main() -> None:
  if len(sys.argv) < 2:
    print(__doc__)
    return
  path = sys.argv[1]
  d = np.load(path)
  p = d["root_pos_w"].astype(np.float64)
  fps = 50.0
  if "fps" in d.files:
    fps = float(np.asarray(d["fps"]).reshape(-1)[0])
  dt = 1.0 / fps
  n = p.shape[0]
  seg = np.diff(p[:, :2], axis=0)
  step = np.linalg.norm(seg, axis=1)
  path_len = float(step.sum())
  net = float(np.linalg.norm(p[-1, :2] - p[0, :2]))
  speed = step / dt
  print("file:", path)
  print("frames:", n, "| fps:", fps, "| duration: %.2f s" % (n * dt))
  print("path_length: %.2f m" % path_len)
  print("net_displacement: %.2f m" % net)
  print("mean_speed: %.2f m/s | max_speed: %.2f m/s" % (float(speed.mean()), float(speed.max())))
  print("x_range: %.2f m | y_range: %.2f m" % (float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1]))))
  print("root_z: min %.3f max %.3f" % (float(p[:, 2].min()), float(p[:, 2].max())))
  per_s = int(round(fps))
  print("per-second path length / mean speed:")
  for i in range(0, n - per_s, per_s):
    blk = step[i:i + per_s]
    print("  t=%3d-%3d s  path %6.2f m  mean %.2f m/s" % (i // per_s, (i + per_s) // per_s, float(blk.sum()), float(blk.mean() / dt)))


if __name__ == "__main__":
  main()
  