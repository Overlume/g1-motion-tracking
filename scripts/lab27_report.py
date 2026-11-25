#!/usr/bin/env python
"""LAB27 报告：多片段联合训练策略 vs M4，4 个窗口 × N 个种子的完成度 / 存活帧数对比。

输入目录结构（由 scripts/lab27_eval.sh 生成）：
  <run>/eval/<policy>/<window>/seed<k>/summary.json

输出：
  <run>/eval/lab27_eval.md    人读表格（含配对差）
  <run>/eval/lab27_eval.json  机器读
  <run>/eval/lab27_eval.png   双面板图（--figure）
  <run>/eval_summary.md|json  副本（runs/<run>/*.md 才入库）
"""

import argparse
import json
import math
from pathlib import Path

WINDOWS = ["train", "heldout", "unseen", "dance"]
WINDOW_CN = {
    "train": "训练片段（S3 0 s）",
    "heldout": "未见窗口（S3 120 s）",
    "unseen": "未见片段（S2 134 s）",
    "dance": "新片段未见窗口（dance1 60 s）",
}
POLICY_CN = {"m4": "M4（单片段）", "lab27": "LAB27（双片段）"}
POLICY_EN = {"m4": "M4 (1 clip)", "lab27": "LAB27 (2 clips)"}
XLABEL = ["train S3 0s", "heldout S3 120s", "unseen S2 134s", "dance1 60s"]


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
    "surv_frames": int(d.get("motion_frame_furthest", 0) or 0),
    "fall": bool(d.get("fall_detected", False)),
    "resets": int(d.get("reset_count", 0) or 0),
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
  for policy in ("m4", "lab27"):
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
  ap.add_argument("--title", default="LAB27 multiclip vs M4")
  ap.add_argument("--figure", action="store_true")
  args = ap.parse_args()

  run = Path(args.run).resolve()
  if not run.is_dir():
    raise SystemExit("run 目录不存在：%s" % run)
  data = collect(run)

  out = {"run": str(run), "windows": WINDOWS, "policies": {}, "paired": {}}
  for policy in ("m4", "lab27"):
    out["policies"][policy] = {}
    for w in WINDOWS:
      cells = data.get(policy, {}).get(w, {})
      out["policies"][policy][w] = {
        "seeds": {str(s): c for s, c in cells.items()},
        "stats": stats([c["completion"] for c in cells.values()]),
        "surv": stats([c["surv_frames"] for c in cells.values()]),
        "falls": sum(1 for c in cells.values() if c["fall"]),
      }

  lines = ["# " + args.title, "",
           "- run：%s" % run,
           "- 口径：scripts/eval_tracking.py，终止检测开、启动 DR 关、999 步；完成度 = track_completion（到过的最远帧 / 总帧）",
           "- 存活帧 = summary.json 的 motion_frame_furthest（终止时走到第几帧，满 999 表示跑完）",
           "- 主指标：held-out（S3 120 s）完成度；M4 该窗口 = 0.100，unseen（S2 134 s）= 0.025",
           ""]
  lines.append("## 1. 各格完成度")
  lines.append("")
  lines.append("| 窗口 | M4（单片段） | LAB27（双片段） | 差（LAB27 − M4） |")
  lines.append("|---|---|---|---|")
  for w in WINDOWS:
    m4s = out["policies"]["m4"].get(w, {}).get("stats")
    l27s = out["policies"]["lab27"].get(w, {}).get("stats")
    if m4s and l27s:
      lines.append("| %s | %s ± %s（n=%d） | %s ± %s（n=%d） | %+.3f |" % (
        WINDOW_CN[w], fmt(m4s["mean"]), fmt(m4s["sd"]), m4s["n"],
        fmt(l27s["mean"]), fmt(l27s["sd"]), l27s["n"], l27s["mean"] - m4s["mean"]))
    elif m4s:
      lines.append("| %s | %s ± %s（n=%d） | — | — |" % (WINDOW_CN[w], fmt(m4s["mean"]), fmt(m4s["sd"]), m4s["n"]))
    elif l27s:
      lines.append("| %s | — | %s ± %s（n=%d） | — |" % (WINDOW_CN[w], fmt(l27s["mean"]), fmt(l27s["sd"]), l27s["n"]))
    else:
      lines.append("| %s | — | — | — |" % WINDOW_CN[w])

  lines += ["", "## 2. 各格存活帧数（motion_frame_furthest，满 999 = 跑完）", "",
            "| 窗口 | M4 | LAB27 |", "|---|---|---|"]
  for w in WINDOWS:
    m4s = out["policies"]["m4"].get(w, {}).get("surv")
    l27s = out["policies"]["lab27"].get(w, {}).get("surv")
    lines.append("| %s | %s | %s |" % (
      WINDOW_CN[w],
      "—" if not m4s else "%d ± %d" % (round(m4s["mean"]), round(m4s["sd"])),
      "—" if not l27s else "%d ± %d" % (round(l27s["mean"]), round(l27s["sd"]))))

  lines += ["", "## 3. 逐种子配对差（同一 seed、同一窗口）", ""]
  for w in WINDOWS:
    m4c = data.get("m4", {}).get(w, {})
    l27c = data.get("lab27", {}).get(w, {})
    common = sorted(set(m4c) & set(l27c))
    if not common:
      lines.append("- %s：没有可配对的种子" % WINDOW_CN[w])
      continue
    diffs = {s: l27c[s]["completion"] - m4c[s]["completion"] for s in common}
    better = sum(1 for d in diffs.values() if d > 0.01)
    worse = sum(1 for d in diffs.values() if d < -0.01)
    same = len(diffs) - better - worse
    lines.append("- %s：n=%d，LAB27 更好 %d / 更差 %d / 差不多 %d；逐种子差 = %s" % (
      WINDOW_CN[w], len(common), better, worse, same,
      ", ".join("s%d %+0.3f" % (s, diffs[s]) for s in common)))
    out["paired"][w] = {str(s): diffs[s] for s in common}

  out_json = run / "eval" / "lab27_eval.json"
  out_json.parent.mkdir(parents=True, exist_ok=True)
  out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
  md_text = "\n".join(lines) + "\n"
  (run / "eval" / "lab27_eval.md").write_text(md_text)
  (run / "eval_summary.md").write_text(md_text)
  (run / "eval_summary.json").write_text(out_json.read_text())

  if args.figure:
    try:
      import matplotlib
      matplotlib.use("Agg")
      import matplotlib.pyplot as plt
      import numpy as np

      fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=140)
      x = np.arange(len(WINDOWS))
      width = 0.36
      for i, policy in enumerate(("m4", "lab27")):
        means = [out["policies"][policy][w]["stats"]["mean"] if out["policies"][policy][w]["stats"] else 0.0 for w in WINDOWS]
        errs = [out["policies"][policy][w]["stats"]["sd"] if out["policies"][policy][w]["stats"] else 0.0 for w in WINDOWS]
        axes[0].bar(x + (i - 0.5) * width, means, width, yerr=errs, capsize=3,
                    label=POLICY_EN[policy], color=("#8c8c8c" if policy == "m4" else "#1f77b4"))
        for xi, m in zip(x + (i - 0.5) * width, means):
          axes[0].text(xi, m + 0.02, "%.3f" % m, ha="center", fontsize=8)
      axes[0].set_xticks(x)
      axes[0].set_xticklabels(XLABEL, fontsize=8)
      axes[0].set_ylabel("track completion")
      axes[0].set_ylim(0, 1.05)
      axes[0].set_title("completion per window")
      axes[0].legend(loc="upper right", fontsize=8)
      axes[0].grid(axis="y", alpha=0.3)

      for i, policy in enumerate(("m4", "lab27")):
        means = [out["policies"][policy][w]["surv"]["mean"] if out["policies"][policy][w]["surv"] else 0.0 for w in WINDOWS]
        axes[1].bar(x + (i - 0.5) * width, means, width,
                    label=POLICY_EN[policy], color=("#8c8c8c" if policy == "m4" else "#1f77b4"))
        for xi, m in zip(x + (i - 0.5) * width, means):
          axes[1].text(xi, m + 10, "%d" % round(m), ha="center", fontsize=8)
      axes[1].set_xticks(x)
      axes[1].set_xticklabels(XLABEL, fontsize=8)
      axes[1].set_ylabel("motion frame furthest")
      axes[1].set_ylim(0, 1050)
      axes[1].set_title("survival frames (999 = finished)")
      axes[1].legend(loc="upper right", fontsize=8)
      axes[1].grid(axis="y", alpha=0.3)

      fig.suptitle(args.title)
      fig.tight_layout()
      png = run / "eval" / "lab27_eval.png"
      fig.savefig(png)
      print("[lab27] 图：%s" % png)
    except Exception as e:
      print("[lab27] 画图失败：%r" % (e,))

  print("[lab27] 表：%s" % (run / "eval" / "lab27_eval.md"))
  print("[lab27] json：%s" % out_json)
  print("")
  print("\n".join(lines))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
