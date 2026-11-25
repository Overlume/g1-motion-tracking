#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LAB22：参考动作（重定向产物）的脚底接触 / 穿地审计（CPU，无需 GPU）。

对每个 npz 运动片段逐帧做 FK：
  1) 把 root_pos_w / root_quat_w（wxyz）+ 29 个 joint_pos 写进 G1 的 qpos；
  2) mj_kinematics 后取两脚踝 roll 连杆下 7 块碰撞 box（脚底板）的最低角点 z；
  3) 统计最低脚底高度、穿地深度、接触比例、双足支撑 / 腾空比例、骨盆高度。

输出：<out>/foot_audit.md|json（+ 可选图）。
用法：
  environment/.venv/bin/python scripts/lab22_foot_audit.py \
      --out-dir runs/20261125-002-lab22-foot-contact 2>&1 | tee runs/20261125-002-lab22-foot-contact/audit.log
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "repos/mjlab_research/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
DEFAULT_CLIPS = [
    "data/processed/fight1_s3_000s_20s.npz",
    "data/processed/fight1_s3_120s_20s.npz",
    "data/processed/fight1_s2_134s_20s.npz",
    "data/processed/lafan_dance1_subject1.npz",
]
CONTACT_TOL = 0.02   # 2 cm：一足判「接触」
FLIGHT_TOL = 0.05    # 5 cm：两足都高于此值判「腾空」
PEN_WARN = 0.020     # 20 mm：穿地警告线
PEN_FAIL = 0.050     # 50 mm：穿地不通过线


def build_model():
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    joint = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): i for i in range(m.njnt)}
    qadr = {k: m.jnt_qposadr[v] for k, v in joint.items()}
    feet = {}
    for side, body in (("L", "left_ankle_roll_link"), ("R", "right_ankle_roll_link")):
        bid = m.body(body).id
        feet[side] = [g for g in range(m.ngeom)
                      if m.geom_bodyid[g] == bid and m.geom_contype[g] != 0]
    return m, d, qadr, feet


def sole_z(m, d, geoms) -> float:
    """脚底板最低角点 z（碰撞 box：size z 常为 0，取 8 个角点即够）。"""
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


def clip_heights(m, d, qadr, feet, npz_path: Path):
    import mujoco

    a = np.load(npz_path)
    names = [str(x) for x in a["joint_names"]] if "joint_names" in a.files else []
    if not names:
        raise SystemExit("%s 缺 joint_names" % npz_path)
    missing = [n for n in names if n not in qadr]
    if missing:
        raise SystemExit("%s 的关节名不在模型里：%s" % (npz_path, missing[:3]))
    n = a["joint_pos"].shape[0]
    fps = float(a["fps"][0]) if "fps" in a.files else 50.0
    zl, zr = np.empty(n), np.empty(n)
    for t in range(n):
        d.qpos[:] = 0.0
        d.qpos[0:3] = a["root_pos_w"][t]
        d.qpos[3:7] = a["root_quat_w"][t]
        for i, jn in enumerate(names):
            d.qpos[qadr[jn]] = a["joint_pos"][t, i]
        mujoco.mj_kinematics(m, d)
        zl[t] = sole_z(m, d, feet["L"])
        zr[t] = sole_z(m, d, feet["R"])
    return zl, zr, fps, n


def summarize(path: Path, zl, zr, fps, n) -> dict:
    zmin = np.minimum(zl, zr)
    pen = np.maximum(0.0, -zmin)
    both = (zl < CONTACT_TOL) & (zr < CONTACT_TOL)
    either = (zl < CONTACT_TOL) | (zr < CONTACT_TOL)
    flight = (zl > FLIGHT_TOL) & (zr > FLIGHT_TOL)
    worst = int(np.argmax(pen))
    above = np.where(zmin > CONTACT_TOL)[0]
    longest_hover = 0
    if above.size:
        runs, cur = [], 1
        for k in range(1, above.size):
            if above[k] == above[k - 1] + 1:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)
        longest_hover = max(runs)
    return {
        "clip": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "frames": int(n),
        "fps": fps,
        "duration_s": round(n / fps, 2),
        "zL_p50": round(float(np.percentile(zl, 50)), 4),
        "zL_min": round(float(zl.min()), 4),
        "zR_p50": round(float(np.percentile(zr, 50)), 4),
        "zR_min": round(float(zr.min()), 4),
        "sole_p50": round(float(np.percentile(zmin, 50)), 4),
        "sole_p1": round(float(np.percentile(zmin, 1)), 4),
        "sole_min": round(float(zmin.min()), 4),
        "pen_max_mm": round(float(pen.max()) * 1000, 1),
        "pen_p99_mm": round(float(np.percentile(pen, 99)) * 1000, 1),
        "pen_frames_gt5mm": int((pen > 0.005).sum()),
        "pen_frames_gt20mm": int((pen > 0.020).sum()),
        "pen_frames_gt50mm": int((pen > 0.050).sum()),
        "worst_frame": int(worst),
        "worst_time_s": round(float(worst) / fps, 2),
        "contact_L": round(float((zl < CONTACT_TOL).mean()), 3),
        "contact_R": round(float((zr < CONTACT_TOL).mean()), 3),
        "contact_either": round(float(either.mean()), 3),
        "contact_both": round(float(both.mean()), 3),
        "flight_ratio": round(float(flight.mean()), 3),
        "longest_hover_frames": int(longest_hover),
        "longest_hover_s": round(longest_hover / fps, 2),
        "pelvis_z_min": round(float(np.load(path)["root_pos_w"][:, 2].min()), 3),
        "pelvis_z_med": round(float(np.median(np.load(path)["root_pos_w"][:, 2])), 3),
    }


def verdict(r: dict) -> str:
    if r["pen_max_mm"] > PEN_FAIL * 1000:
        return "FAIL"
    if r["pen_max_mm"] > PEN_WARN * 1000 or r["pen_p99_mm"] > PEN_WARN * 1000:
        return "WARN"
    return "OK"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--clips", nargs="*", default=DEFAULT_CLIPS)
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--title", default="LAB22 参考动作脚底接触 / 穿地审计")
    args = ap.parse_args()

    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    m, d, qadr, feet = build_model()
    rows, curves = [], {}
    for c in args.clips:
        p = Path(c)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise SystemExit("缺片段：%s" % p)
        zl, zr, fps, n = clip_heights(m, d, qadr, feet, p)
        r = summarize(p, zl, zr, fps, n)
        r["verdict"] = verdict(r)
        rows.append(r)
        curves[r["clip"]] = {"t": (np.arange(n) / fps).round(3).tolist(),
                             "zmin": np.minimum(zl, zr).round(4).tolist()}
        print("[audit] %-46s pen_max %6.1f mm | sole p50 %5.3f | contact either %.2f | %s"
              % (r["clip"].split("/")[-1], r["pen_max_mm"], r["sole_p50"], r["contact_either"], r["verdict"]))

    md = ["# %s" % args.title, "",
          "口径：逐帧 FK 到 G1 的脚底板碰撞 box（每足 7 块，底 z = -0.025 m 相对 ankle_roll）；",
          "接触 = 最低脚底 z < %.0f mm；腾空 = 两足 > %.0f mm；穿地 = 最低脚底 z 为负的深度；" % (CONTACT_TOL * 1000, FLIGHT_TOL * 1000),
          "判定：max 穿地 ≤20 mm 且 p99 ≤20 mm = OK；≤50 mm = WARN；>50 mm = FAIL。", "",
          "| 片段 | 时长 s | 脚底 p50 | 脚底 p1 | 最低 | 穿地 p99 | 穿地 max | >5mm 帧 | >20mm 帧 | 接触(任一足) | 双足 | 腾空 | 最长悬空 s | 判定 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        md.append("| %s | %.1f | %.3f | %.3f | %.3f | %.1f mm | %.1f mm | %d | %d | %.0f%% | %.0f%% | %.0f%% | %.2f | **%s** |"
                  % (r["clip"], r["duration_s"], r["sole_p50"], r["sole_p1"], r["sole_min"],
                     r["pen_p99_mm"], r["pen_max_mm"], r["pen_frames_gt5mm"], r["pen_frames_gt20mm"],
                     100 * r["contact_either"], 100 * r["contact_both"], 100 * r["flight_ratio"],
                     r["longest_hover_s"], r["verdict"]))
    md += ["", "## 逐片段细节", ""]
    for r in rows:
        md.append("- **%s**：%d 帧 / %.1f s @ %.0f fps；最深穿地 %.1f mm 出现在第 %d 帧（%.2f s）；骨盆最低 %.3f m、中位 %.3f m；"
                  "左/右足接触 %.0f%% / %.0f%%。"
                  % (r["clip"].split("/")[-1], r["frames"], r["duration_s"], r["fps"], r["pen_max_mm"],
                     r["worst_frame"], r["worst_time_s"], r["pelvis_z_min"], r["pelvis_z_med"],
                     100 * r["contact_L"], 100 * r["contact_R"]))
    md += ["", "## 怎么读", "",
           "1. **穿地 max** 是硬指标：正数是脚踩进了地面（参考动作物理不可行），< 20 mm 可当数值噪声。",
           "2. **接触(任一足)** 是软指标：跳舞/格斗类参考本来就有跳跃与小碎步；但要能说清「多少时间有支撑」。",
           "3. **脚底 p50** 若比 0 高 2–4 cm，说明这段参考整体悬空（重定向高度没对齐），策略只能靠屈膝/踮脚去凑。",
           "4. 判定 FAIL 不等于「数据作废」，而是「这段参考在真实地面上不可能逐帧复现」，要在报告里限定结论。", ""]
    (out / "foot_audit.md").write_text("\n".join(md), encoding="utf-8")
    (out / "foot_audit.json").write_text(json.dumps({"rows": rows, "curves": curves}, ensure_ascii=False), encoding="utf-8")
    print("[audit] 写盘：%s / foot_audit.{md,json}" % out)

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
        ax = axes[0]
        for name, c in curves.items():
            ax.plot(c["t"], c["zmin"], lw=1.0, label=name.split("/")[-1].replace(".npz", ""))
        ax.axhline(0, color="#444", lw=1, ls="--")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("min foot sole z (m)")
        ax.set_title("lowest foot sole over time", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

        ax = axes[1]
        labels = [r["clip"].split("/")[-1].replace(".npz", "") for r in rows]
        vals = [r["pen_max_mm"] for r in rows]
        colors = ["#2ca02c" if r["verdict"] == "OK" else "#ff7f0e" if r["verdict"] == "WARN" else "#d62728" for r in rows]
        ax.bar(range(len(rows)), vals, color=colors)
        ax.axhline(20, ls="--", lw=1, color="#ff7f0e")
        ax.axhline(50, ls="--", lw=1, color="#d62728")
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("max ground penetration (mm)")
        ax.set_title("penetration by clip", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        fig.suptitle("LAB22: reference motion foot contact audit", fontsize=11)
        fig.tight_layout()
        figpath = ROOT / "outputs/figures/lab22_foot_audit.png"
        fig.savefig(figpath, dpi=130)
        print("[audit] 图：%s" % figpath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
