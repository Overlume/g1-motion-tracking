#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LAB22 附件：策略 vs 参考的骨盆高度偏移（零 GPU，读已有 metrics.csv）。

回答一个问题：参考动作穿地时，策略是「抬着骨盆悬空」还是「跟着参考高度、用腿去凑」？
  offset = pelvis_z - ref_pelvis_z（同一帧）
  |offset| 很小（< 2 cm）→ 策略跟着参考高度走 → 参考穿地只能被关节/接触吸收（表现为关节误差、脚滑）
  offset 明显为正 → 策略整体抬高（悬空）

用法：
  environment/.venv/bin/python scripts/lab22_policy_offset.py \
      --out-dir runs/20261125-002-lab22-foot-contact \
      --metrics runs/20261117-002-m4-eval/final/seed103/metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_metrics(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                rows.append((float(r["time_s"]), float(r["pelvis_z"]), float(r["ref_pelvis_z"]),
                             float(r.get("root_err_cm") or "nan")))
            except (KeyError, ValueError):
                continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--metrics", nargs="*", default=[])
    ap.add_argument("--glob", default="runs/20261117-002-m4-eval/final/*/metrics.csv")
    ap.add_argument("--max-time", type=float, default=0.0, help="只统计 time_s 小于该值的帧（0 = 全部）")
    args = ap.parse_args()

    paths = [Path(p) for p in args.metrics]
    if not paths:
        paths = sorted(ROOT.glob(args.glob))
    if not paths:
        raise SystemExit("没有找到 metrics.csv：检查 --glob")

    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    lines = ["# LAB22 附件：策略 vs 参考的骨盆高度偏移", "",
             "口径：offset = pelvis_z − ref_pelvis_z（逐帧，正值 = 策略骨盆比参考高，即「悬空」）。",
             "",
             "| metrics | 帧数 | offset 中位 | p5 | p95 | \\|offset\\|>2cm 占比 | root_err 中位 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    data = []
    for p in paths:
        rows = read_metrics(p)
        if args.max_time > 0:
            rows = [r for r in rows if r[0] <= args.max_time]
        if not rows:
            print("[offset] 跳过（无有效行）：%s" % p)
            continue
        off = [r[1] - r[2] for r in rows]
        err = [r[3] for r in rows if r[3] == r[3]]
        rec = {
            "metrics": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
            "frames": len(off),
            "offset_med_cm": round(100 * st.median(off), 2),
            "offset_p5_cm": round(100 * sorted(off)[int(0.05 * (len(off) - 1))], 2),
            "offset_p95_cm": round(100 * sorted(off)[int(0.95 * (len(off) - 1))], 2),
            "share_gt2cm": round(sum(1 for x in off if abs(x) > 0.02) / len(off), 3),
            "root_err_med_cm": round(st.median(err), 2) if err else None,
        }
        data.append(rec)
        lines.append("| %s | %d | %+.2f cm | %+.2f | %+.2f | %.0f%% | %s |"
                     % (rec["metrics"], rec["frames"], rec["offset_med_cm"], rec["offset_p5_cm"],
                        rec["offset_p95_cm"], 100 * rec["share_gt2cm"],
                        "%.2f cm" % rec["root_err_med_cm"] if rec["root_err_med_cm"] is not None else "—"))
        print("[offset] %s：中位 %+.2f cm，|Δ|>2cm 占 %.0f%%" % (rec["metrics"], rec["offset_med_cm"], 100 * rec["share_gt2cm"]))

    lines += ["", "读法：", "",
              "- 中位 offset 绝对值 < 2 cm → 策略跟着参考的骨盆高度走：参考穿地不会变成「策略悬空」，而是变成腿/脚的去向偏差（joint_rmse、脚滑）。",
              "- 中位 offset 明显为正（> 2 cm）→ 策略整体抬高，说明它在回避地面穿模；此时完成度与 root 误差要扣除这一层。",
              "- 与 foot_audit.md 的最大穿地帧对照，可验证「策略在穿地帧是否抬高」。", ""]
    (out / "policy_offset.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "policy_offset.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[offset] 写盘：%s" % (out / "policy_offset.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
