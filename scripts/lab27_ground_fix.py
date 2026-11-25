#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LAB27 Part A：参考动作的「地面高度校正」+ 切片（CPU）。

为什么需要：LAB22 审计显示重定向产物脚底中位悬空 0.8–2.8 cm、最深穿地 5.3–6.0 cm，
即「同一段动作里有的帧脚在空中、有的帧穿到地下」，是重定向高度标定的问题（D41 判读 2）。
多片段训练需要新片段过 D41 的闸（max 穿地 ≤20 mm 且 p99 ≤20 mm），所以先把新片段做高度校正：

  1) 逐帧 FK 求两脚底板最低点 zmin；
  2) 接触帧（zmin < 5 cm）把整体竖直偏移设成 offset = -zmin（脚刚好落地）；
  3) 腾空帧的 offset 用相邻接触帧线性插值（保持腾空弧线，不硬压）；
  4) offset 加到 root_pos_w 的 z 上，root_lin_vel_w 的 z 分量加 d(offset)/dt；
  5) 可选按 --start-s / --dur-s 切片，输出 <out> 的 npz。

注意：本脚本**不改** S3 0s（训练基线片段）——改了它，既有 M4 数字就不可比。只对新增片段用。

用法：
  environment/.venv/bin/python scripts/lab27_ground_fix.py \
      --src data/processed/lafan_dance1_subject1.npz --start-s 0 --dur-s 20 \
      --out data/processed/dance1_000s_20s_gfix.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "repos/mjlab_research/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
CONTACT_TOL = 0.05     # 5 cm：判「该帧有脚在地面附近」
SMOOTH = 3             # offset 的滑动平均窗口（帧），压掉单帧毛刺


def build_model():
  import mujoco

  m = mujoco.MjModel.from_xml_path(str(XML))
  d = mujoco.MjData(m)
  joint = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): i for i in range(m.njnt)}
  qadr = {k: m.jnt_qposadr[v] for k, v in joint.items()}
  feet = {}
  for side, body in (("L", "left_ankle_roll_link"), ("R", "right_ankle_roll_link")):
    bid = m.body(body).id
    feet[side] = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_contype[g] != 0]
  return m, d, qadr, feet


def sole_z(m, d, geoms) -> float:
  lo = np.inf
  for g in geoms:
    R = d.geom_xmat[g].reshape(3, 3)
    p = d.geom_xpos[g]
    s = m.geom_size[g]
    for sx in (-s[0], s[0]):
      for sy in (-s[1], s[1]):
        for sz in (-s[2], s[2]):
          lo = min(lo, float((p + R @ np.array([sx, sy, sz]))[2]))
  return lo


def zmin_series(arr, joint_names, m, d, qadr, feet):
  import mujoco

  n = arr["joint_pos"].shape[0]
  out = np.empty(n)
  for t in range(n):
    d.qpos[:] = 0.0
    d.qpos[0:3] = arr["root_pos_w"][t]
    d.qpos[3:7] = arr["root_quat_w"][t]
    for i, jn in enumerate(joint_names):
      d.qpos[qadr[jn]] = arr["joint_pos"][t, i]
    mujoco.mj_kinematics(m, d)
    out[t] = min(sole_z(m, d, feet["L"]), sole_z(m, d, feet["R"]))
  return out


def smooth(a: np.ndarray, w: int) -> np.ndarray:
  if w <= 1:
    return a
  k = np.ones(w) / w
  pad = w // 2
  ext = np.pad(a, pad, mode="edge")
  return np.convolve(ext, k, mode="valid")[: a.size]


def stats(zmin: np.ndarray) -> dict:
  pen = np.maximum(0.0, -zmin)
  return {
    "frames": int(zmin.size),
    "sole_p50_cm": round(float(np.percentile(zmin, 50)) * 100, 2),
    "sole_p1_cm": round(float(np.percentile(zmin, 1)) * 100, 2),
    "pen_max_mm": round(float(pen.max()) * 1000, 1),
    "pen_p99_mm": round(float(np.percentile(pen, 99)) * 1000, 1),
    "contact_ratio": round(float((np.minimum(zmin, CONTACT_TOL) <= CONTACT_TOL).mean()), 3),
  }


def main() -> int:
  p = argparse.ArgumentParser(description="LAB27: 参考动作地面高度校正 + 切片")
  p.add_argument("--src", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--start-s", type=float, default=0.0)
  p.add_argument("--dur-s", type=float, default=None, help="默认整段")
  p.add_argument("--smooth", type=int, default=SMOOTH)
  args = p.parse_args()

  a = dict(np.load(ROOT / args.src)) if not Path(args.src).is_absolute() else dict(np.load(args.src))
  names = [str(x) for x in a["joint_names"]]
  fps = float(a["fps"][0]) if "fps" in a else 50.0
  n_all = a["joint_pos"].shape[0]

  s0 = int(round(args.start_s * fps))
  n = n_all - s0 if args.dur_s is None else int(round(args.dur_s * fps))
  sl = slice(s0, s0 + n)
  for k in ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel"):
    a[k] = a[k][sl]

  m, d, qadr, feet = build_model()
  z_before = zmin_series(a, names, m, d, qadr, feet)

  contact = z_before < CONTACT_TOL
  off = np.full(n, np.nan)
  off[contact] = -z_before[contact]
  idx = np.arange(n)
  valid = ~np.isnan(off)
  if valid.sum() < 2:
    raise SystemExit("[ERR] 接触帧不足（%d），无法插值" % valid.sum())
  off = np.interp(idx, idx[valid], off[valid])
  off = smooth(off, args.smooth)

  a["root_pos_w"] = a["root_pos_w"].copy()
  a["root_pos_w"][:, 2] += off
  dt = 1.0 / fps
  d_off = np.gradient(off, dt)
  a["root_lin_vel_w"] = a["root_lin_vel_w"].copy()
  a["root_lin_vel_w"][:, 2] += d_off
  a["gfix_offset"] = off.astype(np.float32)
  a["gfix_note"] = np.array(["lab27 ground fix: contact clamp + linear interp, smooth=%d" % args.smooth])

  z_after = zmin_series(a, names, m, d, qadr, feet)
  before, after = stats(z_before), stats(z_after)
  info = {"src": args.src, "out": args.out, "start_s": args.start_s, "dur_s": round(n / fps, 2),
          "before": before, "after": after, "smooth": args.smooth,
          "offset_p50_cm": round(float(np.median(off)) * 100, 2),
          "offset_p1_cm": round(float(np.percentile(off, 1)) * 100, 2),
          "offset_p99_cm": round(float(np.percentile(off, 99)) * 100, 2)}
  out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(out_path, **a)
  print(json.dumps(info, ensure_ascii=False, indent=2))
  ok = after["pen_max_mm"] <= 20.0 and after["pen_p99_mm"] <= 20.0
  print("[gate] %s（max %.1f mm / p99 %.1f mm，阈值 20/20）"
        % ("PASS" if ok else "FAIL", after["pen_max_mm"], after["pen_p99_mm"]))
  return 0 if ok else 2


if __name__ == "__main__":
  raise SystemExit(main())
