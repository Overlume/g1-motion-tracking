#!/usr/bin/env python
"""LAB20 报告：DR 加强策略 vs M4，3 个窗口 × N 个种子的完成度对比。

输入目录结构（由 scripts/lab20_eval.sh 生成）：
  <run>/eval/<policy>/<window>/seed<k>/summary.json

输出：
  <run>/eval/lab20_eval.md   人读表格（含配对差、判定提示）
  <run>/eval/lab20_eval.json 机器读
  <run>/eval/lab20_eval.png  分组柱状图（--figure）
"""

import argparse
import json
import math
from pathlib import Path

WINDOWS = ["train", "heldout", "unseen"]
WINDOW_CN = {"train": "训练片段（S3 0 s）", "heldout": "未见窗口（S3 120 s）", "unseen": "未见片段（S2 134 s）"}
POLICY_CN = {"m4": "M4（默认 DR）", "dr": "LAB20（DR 加强）"}
POLICY_EN = {"m4": "M4 (default DR)", "dr": "LAB20 (harder DR)"}


def load_cell(path: Path):
  f = path / "summary.json"
  if not f.exists():
    return None
  try:
    d = json.loads(f.read_text())
  except Exception:
    return None
  return {
    "completion": float(d.get("track_completion", 0.0)),
    "fall": bool(d.get("fall_detected", False)),
    "resets": int(d.get("reset_count", 0)),
  }


def stats(vals):
  n = len(vals)
  if n == 0:
    return None
  mean = sum(vals) / n
  sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
  return {"n": n, "mean": mean, "sd": sd, "min": min(vals), "max": max(vals)}


def collect(run: Path):
  data = {}
  for policy in ("m4", "dr"):
    for window in WINDOWS:
      d = run / "eval" / policy / window
      cells = {}
      if d.is_dir():
        for seed_dir in sorted(d.glob("seed*")):
          try:
            seed = int(seed_dir.name.replace("seed", ""))
          except ValueError:
            continue
          c = load_cell(seed_dir)
          if c is not None:
            cells[seed] = c
      data.setdefault(policy, {})[window] = cells
  return data


def fmt(x, nd=3):
  return "—" if x is None else ("%." + str(nd) + "f") % x


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", required=True)
  ap.add_argument("--title", default="LAB20 (DR-strengthened) vs M4")
  ap.add_argument("--figure", action="store_true")
  args = ap.parse_args()

  run = Path(args.run).resolve()
  if not run.is_dir():
    raise SystemExit("run 目录不存在：%s" % run)
  data = collect(run)

  out = {"run": str(run), "windows": WINDOWS, "policies": {}, "paired": {}}
  for policy in ("m4", "dr"):
    out["policies"][policy] = {}
    for w in WINDOWS:
      cells = data.get(policy, {}).get(w, {})
      out["policies"][policy][w] = {
        "seeds": {str(s): c for s, c in cells.items()},
        "stats": stats([c["completion"] for c in cells.values()]),
        "falls": sum(1 for c in cells.values() if c["fall"]),
      }

  lines = ["# " + args.title, "",
           "- run：%s" % run,
           "- 口径：scripts/eval_tracking.py，终止检测开、启动 DR 关、999 步；完成度 = track_completion（到过的最远帧 / 总帧）",
           ""]
  lines.append("## 1. 各格完成度")
  lines.append("")
  lines.append("| 窗口 | M4（默认 DR） | LAB20（DR 加强） | 差（DR − M4） |")
  lines.append("|---|---|---|---|")
  for w in WINDOWS:
    m4s = out["policies"]["m4"].get(w, {}).get("stats")
    drs = out["policies"]["dr"].get(w, {}).get("stats")
    if m4s and drs:
      diff = drs["mean"] - m4s["mean"]
      lines.append("| %s | %s ± %s（n=%d） | %s ± %s（n=%d） | %+.3f |" % (
        WINDOW_CN[w], fmt(m4s["mean"]), fmt(m4s["sd"]), m4s["n"],
        fmt(drs["mean"]), fmt(drs["sd"]), drs["n"], diff))
    elif m4s:
      lines.append("| %s | %s ± %s（n=%d） | — | — |" % (WINDOW_CN[w], fmt(m4s["mean"]), fmt(m4s["sd"]), m4s["n"]))
    elif drs:
      lines.append("| %s | — | %s ± %s（n=%d） | — |" % (WINDOW_CN[w], fmt(drs["mean"]), fmt(drs["sd"]), drs["n"]))
    else:
      lines.append("| %s | — | — | — |" % WINDOW_CN[w])

  lines += ["", "## 2. 逐种子配对差（同一 seed、同一窗口）", ""]
  for w in WINDOWS:
    m4c = data.get("m4", {}).get(w, {})
    drc = data.get("dr", {}).get(w, {})
    common = sorted(set(m4c) & set(drc))
    if not common:
      lines.append("- %s：没有可配对的种子" % WINDOW_CN[w])
      continue
    diffs = {s: drc[s]["completion"] - m4c[s]["completion"] for s in common}
    better = sum(1 for d in diffs.values() if d > 0.01)
    worse = sum(1 for d in diffs.values() if d < -0.01)
    same = len(diffs) - better - worse
    lines.append("- %s：n=%d，DR 更好 %d / 更差 %d / 差不多 %d；逐种子差 = %s" % (
      WINDOW_CN[w], len(common), better, worse, same,
      ", ".join("s%d %+0.3f" % (s, diffs[s]) for s in common)))
    out["paired"][w] = {str(s): diffs[s] for s in common}

  out_json = run / "eval" / "lab20_eval.json"
  out_json.parent.mkdir(parents=True, exist_ok=True)
  out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")

  md = run / "eval" / "lab20_eval.md"
  md.write_text("\n".join(lines) + "\n")

  if args.figure:
    try:
      import matplotlib
      matplotlib.use("Agg")
      import matplotlib.pyplot as plt
      import numpy as np

      fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
      x = np.arange(len(WINDOWS))
      width = 0.36
      for i, policy in enumerate(("m4", "dr")):
        means, errs = [], []
        for w in WINDOWS:
          st = out["policies"][policy].get(w, {}).get("stats")
          means.append(st["mean"] if st else 0.0)
          errs.append(st["sd"] if st else 0.0)
        ax.bar(x + (i - 0.5) * width, means, width, yerr=errs, capsize=3,
               label=POLICY_EN[policy], color=("#8c8c8c" if policy == "m4" else "#c0392b"))
        for xi, m in zip(x + (i - 0.5) * width, means):
          ax.text(xi, m + 0.02, "%.3f" % m, ha="center", fontsize=8)
      ax.set_xticks(x)
      ax.set_xticklabels(["train (S3 0s)", "heldout (S3 120s)", "unseen (S2 134s)"])
      ax.set_ylabel("track completion")
      ax.set_ylim(0, 1.05)
      ax.set_title(args.title)
      ax.legend(loc="upper right", fontsize=8)
      ax.grid(axis="y", alpha=0.3)
      fig.tight_layout()
      png = run / "eval" / "lab20_eval.png"
      fig.savefig(png)
      print("[lab20] 图：%s" % png)
    except Exception as e:  # 图失败不影响表
      print("[lab20] 画图失败：%r" % (e,))

  print("[lab20] 表：%s" % md)
  print("[lab20] json：%s" % out_json)
  print("")
  print("\n".join(lines))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
