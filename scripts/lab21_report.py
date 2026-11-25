#!/usr/bin/env python
"""LAB21 报告：世界系根位置/速度观测策略 vs M4，3 窗口 × N 种子。

输入：
  <run>/eval/obs/<window>/seed<k>/summary.json        （scripts/lab21_eval.sh 生成）
  <baseline>/<window>/seed<k>/summary.json            （默认 LAB20 已跑好的 M4 基线）

输出：
  <run>/lab21_obs.md      人读表格（每窗口均值±sd、配对差、方向一致性、判定）
  <run>/lab21_obs.json    机器读
  outputs/figures/lab21_obs.png   左：三窗口完成度对比；右：逐种子配对差

判定线（讲义 §3 预先写定）：
  完成度变化 >= +0.05 → 明显改善；<= -0.05 → 明显变差；其间 → 无显著变化；
  另外要求同向种子 >= 7/10 才算「方向一致」，否则只写「分散」。
"""

import argparse
import json
import math
from pathlib import Path

WINDOWS = ["train", "heldout", "unseen"]
WINDOW_CN = {"train": "训练片段（S3 0 s）", "heldout": "未见窗口（S3 120 s）", "unseen": "未见片段（S2 134 s）"}
POLICY_CN = {"m4": "M4（基线）", "obs": "LAB21（+世界系根位姿）"}
POLICY_EN = {"m4": "M4 (baseline)", "obs": "LAB21 (+world root obs)"}
METRICS = [("completion", "track_completion", "完成度"), ("mpkpe", "mpkpe_mean_cm", "MPKPE cm"),
           ("r_mpkpe", "r_mpkpe_mean_cm", "R-MPKPE cm"), ("root_err", "root_err_mean_cm", "root err cm")]


def load_cell(path: Path):
  f = path / "summary.json"
  if not f.exists():
    return None
  try:
    d = json.loads(f.read_text())
  except Exception:
    return None
  out = {}
  for key, field, _cn in METRICS:
    out[key] = float(d.get(field, float("nan")))
  out["resets"] = int(d.get("reset_count", 0))
  out["ckpt"] = str(d.get("checkpoint", ""))
  return out


def collect(root: Path):
  data = {}
  for window in WINDOWS:
    cells = {}
    d = root / window
    if d.is_dir():
      for seed_dir in sorted(d.glob("seed*")):
        try:
          seed = int(seed_dir.name.replace("seed", ""))
        except ValueError:
          continue
        c = load_cell(seed_dir)
        if c is not None:
          cells[seed] = c
    data[window] = cells
  return data


def stats(vals):
  n = len(vals)
  if n == 0:
    return None
  mean = sum(vals) / n
  sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
  return {"n": n, "mean": mean, "sd": sd, "min": min(vals), "max": max(vals)}


def fmt(x, nd=3):
  return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else ("%." + str(nd) + "f") % x


def paired(base, new, key):
  seeds = sorted(set(base) & set(new))
  diffs = [(s, new[s][key] - base[s][key]) for s in seeds]
  diffs = [(s, d) for s, d in diffs if not math.isnan(d)]
  if not diffs:
    return None
  vals = [d for _s, d in diffs]
  better = sum(1 for _s, d in diffs if d > 0.005)
  worse = sum(1 for _s, d in diffs if d < -0.005)
  return {"n": len(diffs), "mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals),
          "better": better, "worse": worse, "per_seed": {str(s): d for s, d in diffs}}


def verdict(comp_delta, better, worse, n):
  if comp_delta is None:
    return "数据不足"
  if comp_delta >= 0.05 and better >= max(7, int(0.7 * n)):
    return "明显改善（≥ +0.05 且同向 ≥ 7/%d）" % n
  if comp_delta <= -0.05 and worse >= max(7, int(0.7 * n)):
    return "明显变差（≤ -0.05 且同向 ≥ 7/%d）" % n
  if abs(comp_delta) < 0.01:
    return "无显著变化（|Δ| < 0.01）"
  return "分散/弱效应（Δ = %+.3f，好 %d / 差 %d）" % (comp_delta, better, worse)


def main() -> int:
  ap = argparse.ArgumentParser(description="LAB21 report")
  ap.add_argument("--run", required=True)
  ap.add_argument("--baseline", default="runs/20261121-003-lab20-dr/eval/m4")
  ap.add_argument("--figure", action="store_true")
  ap.add_argument("--title", default="LAB21: world-frame root pos/vel observation vs M4")
  args = ap.parse_args()

  run = Path(args.run).resolve()
  base_root = Path(args.baseline).resolve()
  base = collect(base_root)
  obs = collect(run / "eval" / "obs")

  lines = ["# %s" % args.title, ""]
  lines.append("基线目录：%s（M4，3 窗口 × 10 种子）" % base_root)
  lines.append("")
  lines.append("口径：scripts/eval_tracking.py，终止检测开、startup DR 关、跑满动作帧数；种子 101–110。")
  lines.append("")
  lines.append("| 窗口 | 策略 | n | 完成度 均值±sd | MPKPE cm | R-MPKPE cm | root err cm |")
  lines.append("| --- | --- | --- | --- | --- | --- | --- |")
  table = {}
  for window in WINDOWS:
    for name, data in (("m4", base), ("obs", obs)):
      cells = data.get(window, {})
      if not cells:
        lines.append("| %s | %s | 0 | — | — | — | — |" % (WINDOW_CN[window], POLICY_CN[name]))
        continue
      st = {k: stats([c[k] for c in cells.values()]) for k, _f, _cn in METRICS}
      n = st["completion"]["n"]
      lines.append("| %s | %s | %d | %.3f ± %.3f | %.2f | %.2f | %.2f |" % (
        WINDOW_CN[window], POLICY_CN[name], n,
        st["completion"]["mean"], st["completion"]["sd"],
        st["mpkpe"]["mean"], st["r_mpkpe"]["mean"], st["root_err"]["mean"]))
      table.setdefault(window, {})[name] = {k: v for k, v in st.items()}

  lines.append("")
  lines.append("## 逐窗口配对差（LAB21 − M4，仅共同种子）")
  lines.append("")
  lines.append("| 窗口 | 完成度 Δ 均值 | 好/差 种子数 | R-MPKPE Δ | root err Δ | 判定 |")
  lines.append("| --- | --- | --- | --- | --- | --- |")
  summary = {}
  for window in WINDOWS:
    b, o = base.get(window, {}), obs.get(window, {})
    pc = paired(b, o, "completion")
    pr = paired(b, o, "r_mpkpe")
    pe = paired(b, o, "root_err")
    if pc is None:
      lines.append("| %s | — | — | — | — | 数据不足 |" % WINDOW_CN[window])
      summary[window] = None
      continue
    v = verdict(pc["mean"], pc["better"], pc["worse"], pc["n"])
    lines.append("| %s | %+.3f（%+.3f ~ %+.3f） | %d / %d | %s | %s | %s |" % (
      WINDOW_CN[window], pc["mean"], pc["min"], pc["max"], pc["better"], pc["worse"],
      ("%+.2f" % pr["mean"]) if pr else "—", ("%+.2f" % pe["mean"]) if pe else "—", v))
    summary[window] = {"completion": pc, "r_mpkpe": pr, "root_err": pe, "verdict": v}

  lines.append("")
  lines.append("## 怎么读")
  lines.append("")
  lines.append("1. **先看 held-out 的 Δ**：这是 P7 想解决的问题（窗口专家）。Δ ≥ +0.05 才算「有效」，")
  lines.append("   0.10 → 0.15 这种量级才值得写进结论；若只是 0.101 → 0.105，不要写成改善。")
  lines.append("2. **再看 train**：如果 train 掉了（例如 0.999 → 0.9），说明新输入干扰了专精，不是「更好的泛化」。")
  lines.append("3. **R-MPKPE 是去漂移的姿态误差**，比完成度更接近「动作像不像」；它若同时变好，改善才可信。")
  lines.append("4. **落回 D34 的框架**：held-out 完成度在 M4 上是「窗口记忆外溢」的产物；如果 LAB21 把它抬起来，")
  lines.append("   要再看 unseen 是否跟着动 —— 只有 unseen 也动，才是真泛化，否则只是换了记忆方式。")
  lines.append("")

  # JSON
  payload = {"run": str(run), "baseline": str(base_root), "table": table, "paired": summary}
  (run / "lab21_obs.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
  (run / "lab21_obs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
  print("\n".join(lines))

  if args.figure:
    plot(run, base, obs, args.title)
  return 0


def plot(run: Path, base, obs, title: str) -> None:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
  ax = axes[0]
  width = 0.35
  xs = list(range(len(WINDOWS)))
  for i, (name, data, color) in enumerate((("m4", base, "#8c8c8c"), ("obs", obs, "#c0392b"))):
    means = [stats([c["completion"] for c in data.get(w, {}).values()])["mean"] if data.get(w) else 0.0
             for w in WINDOWS]
    sds = [stats([c["completion"] for c in data.get(w, {}).values()])["sd"] if data.get(w) else 0.0
           for w in WINDOWS]
    ax.bar([x + (i - 0.5) * width for x in xs], means, width, yerr=sds, capsize=3,
           label=POLICY_EN[name], color=color)
    for x, m in zip(xs, means):
      ax.text(x + (i - 0.5) * width, m + 0.01, "%.3f" % m, ha="center", fontsize=9)
  ax.set_xticks(xs)
  ax.set_xticklabels(["train (S3 0s)", "heldout (S3 120s)", "unseen (S2 134s)"])
  ax.set_ylabel("track completion")
  ax.set_title("completion by window")
  ax.legend(fontsize=8)
  ax.grid(axis="y", alpha=0.3)

  ax = axes[1]
  for i, w in enumerate(WINDOWS):
    b, o = base.get(w, {}), obs.get(w, {})
    seeds = sorted(set(b) & set(o))
    ds = [o[s]["completion"] - b[s]["completion"] for s in seeds]
    ax.scatter([i] * len(ds), ds, s=22, alpha=0.8, color="#c0392b")
    if ds:
      ax.plot([i - 0.25, i + 0.25], [sum(ds) / len(ds)] * 2, color="k", lw=1.4)
  ax.axhline(0, color="k", lw=0.8)
  ax.axhline(0.05, color="#2e7d32", ls="--", lw=0.9)
  ax.axhline(-0.05, color="#b71c1c", ls="--", lw=0.9)
  ax.set_xticks(range(len(WINDOWS)))
  ax.set_xticklabels(["train", "heldout", "unseen"])
  ax.set_ylabel("completion delta (LAB21 - M4)")
  ax.set_title("per-seed paired delta (dashed: +-0.05)")
  ax.grid(axis="y", alpha=0.3)

  fig.suptitle(title, fontsize=12)
  fig.tight_layout()
  out = Path("outputs/figures/lab21_obs.png")
  out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out, dpi=140)
  print("[lab21] figure -> %s" % out)


if __name__ == "__main__":
  raise SystemExit(main())
