"""查一个 mjlab 动作 npz 在指定时间窗内的运动方向与速度（只依赖 numpy，只读）。

用法:
    environment/.venv/bin/python scripts/motion_probe.py <motion.npz> --t0 8.0 --t1 8.2
    environment/.venv/bin/python scripts/motion_probe.py <motion.npz> --scan 0.2

输出（世界系，右手系，z 向上；航向角 0=+x，90=+y，逆时针）:
    - 根（骨盆）平均线速度 (vx, vy, vz) 与水平速率
    - 该窗口的位移向量与位移方向
    - 骨盆 yaw 变化（朝向变化）
    - waist_yaw/roll/pitch 三个腰关节的变化（躯干相对骨盆的姿态）
不加 --t0/--t1 时按 --scan 秒逐窗扫描，用来看整段动作的"往哪走、多快"。
"""

import argparse

import numpy as np


def yaw_deg(q) -> float:
  x, y, z, w = q
  return float(np.degrees(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))))


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("motion")
  ap.add_argument("--t0", type=float, default=None)
  ap.add_argument("--t1", type=float, default=None)
  ap.add_argument("--scan", type=float, default=0.2, help="扫描窗宽（秒）")
  ap.add_argument("--fps", type=float, default=None,
                  help="覆盖帧率；npz 无 fps 字段时（如 dance1）务必显式给出，否则按 50 处理")
  args = ap.parse_args()

  d = np.load(args.motion)
  if args.fps is not None:
    fps = float(args.fps)
  elif "fps" in d.files:
    fps = float(np.asarray(d["fps"]).reshape(-1)[0])
  else:
    fps = 50.0
    print("[warn] npz 无 fps 字段，按 50 fps 处理；若不对请用 --fps 覆盖")
  rp = d["root_pos_w"].astype(np.float64)
  rq = d["root_quat_w"].astype(np.float64)
  rv = d["root_lin_vel_w"].astype(np.float64)
  jp = d["joint_pos"].astype(np.float64)
  names = [str(n) for n in d["joint_names"]]
  idx = {n: i for i, n in enumerate(names)}
  waist = [n for n in names if "waist" in n]

  print("file: %s | fps: %.4f | frames: %d | duration: %.2f s"
        % (args.motion, fps, len(rp), len(rp) / fps))

  def window(t0: float, t1: float) -> None:
    i0 = int(round(t0 * fps))
    i1 = int(round(t1 * fps))
    i0 = max(0, min(i0, len(rp) - 2))
    i1 = max(i0 + 1, min(i1, len(rp) - 1))
    v = rv[i0:i1 + 1]
    vmean = v.mean(axis=0)
    sp = np.linalg.norm(v, axis=1)
    dp = rp[i1] - rp[i0]
    hdg_v = np.degrees(np.arctan2(vmean[1], vmean[0]))
    hdg_d = np.degrees(np.arctan2(dp[1], dp[0]))
    print("\n== t=%.2f-%.2f s  frames %d-%d (%d frames)" % (t0, t1, i0, i1, i1 - i0 + 1))
    print("  v_mean      (vx,vy,vz) = (%+.3f, %+.3f, %+.3f) m/s  |  |v_xy| = %.3f m/s"
          % (vmean[0], vmean[1], vmean[2], np.linalg.norm(vmean[:2])))
    print("  speed|v|    mean %.3f  min %.3f  max %.3f m/s" % (sp.mean(), sp.min(), sp.max()))
    print("  displacement dx=%+.3f dy=%+.3f dz=%+.3f m  ->  |d_xy|=%.3f m, dir %+.1f deg"
          % (dp[0], dp[1], dp[2], np.linalg.norm(dp[:2]), hdg_d))
    print("  heading     velocity %+.1f deg | displacement %+.1f deg" % (hdg_v, hdg_d))
    print("  pelvis yaw  %+.1f -> %+.1f deg (delta %+.1f)" % (yaw_deg(rq[i0]), yaw_deg(rq[i1]), yaw_deg(rq[i1]) - yaw_deg(rq[i0])))
    for n in waist:
      a = np.degrees(jp[i0, idx[n]])
      b = np.degrees(jp[i1, idx[n]])
      print("  %-18s %+7.1f -> %+7.1f deg (delta %+5.1f)" % (n, a, b, b - a))

  if args.t0 is not None and args.t1 is not None:
    window(args.t0, args.t1)
  else:
    step = args.scan
    t = 0.0
    dur = len(rp) / fps
    while t + step <= dur + 1e-9:
      window(t, t + step)
      t += step


if __name__ == "__main__":
  main()
