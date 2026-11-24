#!/usr/bin/env python
"""LAB25：摩擦上界收敛实验的汇总（跨训练 seed + f30 的 −y 扩测）。

读 <run>/arms/<arm>/perturb/rep*/recovery.json，与已有基线对比：
  * nx_j60：M4（LAB15/LAB16 seed0）、LAB24 的 f20/f30（seed0）、本次的 f20s1/f30s1（seed1）
  * ny_j60 / ny_j45：M4（LAB16 / LAB17）vs f30（0.2–3.0，LAB24 ckpt 扩测）
输出 <run>/lab25_converge.md|json 与 outputs/figures/lab25_converge.png。
"""

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
QB = chr(96)


def q(s):
  return s.replace("QB", QB)


ARMS = {"f30ny60": "ny_j60", "f30ny45": "ny_j45", "f20s1": "nx_j60", "f30s1": "nx_j60",
        "f20s2": "nx_j60", "f30s2": "nx_j60"}

BASELINE = {
  "M4_nx": [RUNS / "20261120-001-lab15-boundary20/recovery.json",
            RUNS / "20261120-002-lab16-fliprate/rep2/recovery.json",
            RUNS / "20261120-002-lab16-fliprate/rep3/recovery.json"],
  "f20s0_nx": [RUNS / ("20261123-001-lab24-fric-scan/arms/f20/perturb/rep%d/recovery.json" % i) for i in (1, 2, 3)],
  "f30s0_nx": [RUNS / ("20261123-001-lab24-fric-scan/arms/f30/perturb/rep%d/recovery.json" % i) for i in (1, 2, 3)],
  "M4_ny60": [RUNS / "20261120-001-lab15-boundary20/recovery.json",
              RUNS / "20261120-002-lab16-fliprate/rep2/recovery.json",
              RUNS / "20261120-002-lab16-fliprate/rep3/recovery.json"],
  "M4_ny45": [RUNS / ("20261120-003-lab17-y-transition/rep%d/recovery.json" % i) for i in (1, 2, 3)],
}


def load(path, cell):
  if not Path(path).exists():
    return []
  try:
    rows = json.loads(Path(path).read_text())
  except Exception:
    return []
  return [r for r in rows if r.get("cell") == cell]


def stats(rows):
  out = {"n": len(rows), "k": 0, "root": None, "drift": 0, "yield": 0, "edge": 0, "ks": []}
  if not rows:
    return out
  out["k"] = sum(1 for r in rows if r.get("cls") == "down")
  roots = [r["root_err_end_cm"] for r in rows if r.get("cls") != "down"]
  out["root"] = round(st.median(roots), 1) if roots else None
  for r in rows:
    if r.get("cls") == "down":
      continue
    dev = abs(r.get("longest_dev_s") or 0.0)
    root = r.get("root_err_end_cm") or 0.0
    if dev >= 0.5 and root >= 30:
      out["yield"] += 1
    elif dev < 0.5 and root >= 30:
      out["drift"] += 1
    elif dev <= 0.3:
      out["edge"] += 1
  return out


def per_rep(paths, cell):
  return [stats(load(p, cell)) for p in paths]


def fmt_ks(reps):
  if not reps or all(r["n"] == 0 for r in reps):
    return "—"
  return "/".join(str(r["k"]) if r["n"] else "—" for r in reps)


def fmt_pool(reps):
  ks = [r["k"] for r in reps]
  ns = [r["n"] for r in reps]
  if not ns or sum(ns) == 0:
    return "—"
  return "%d/%d（%d%%）" % (sum(ks), sum(ns), round(100.0 * sum(ks) / sum(ns)))


def fmt_root(reps):
  vals = [r["root"] for r in reps if r["root"] is not None]
  return "%.1f" % st.median(vals) if vals else "—"


def fisher_cell(a, b):
  out = []
  for ra, rb in zip(a, b):
    if not ra["n"] or not rb["n"]:
      out.append("—")
      continue
    _, p = fisher_exact([[ra["k"], ra["n"] - ra["k"]], [rb["k"], rb["n"] - rb["k"]]])
    out.append("%.3f" % p)
  return "/".join(out) if out else "—"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", required=True)
  ap.add_argument("--figure", action="store_true")
  ap.add_argument("--title", default="LAB25 摩擦上界收敛（跨训练 seed + f30 的 -y 扩测）")
  ap.add_argument("--figure-title", default="LAB25: friction upper bound - cross training seed and -y check")
  args = ap.parse_args()

  run = Path(args.run)
  if not run.is_absolute():
    run = ROOT / run
  md = [q("# %s") % args.title, "", "- run：%s" % run.relative_to(ROOT), ""]

  new = {}
  for arm, cell in ARMS.items():
    paths = [run / ("arms/%s/perturb/rep%d/recovery.json" % (arm, i)) for i in (1, 2, 3)]
    new[arm] = per_rep(paths, cell)

  base = {}
  for key, paths in BASELINE.items():
    cell = "ny_j60" if key == "M4_ny60" else ("ny_j45" if key == "M4_ny45" else "nx_j60")
    base[key] = per_rep(paths, cell)

  m4nx = base["M4_nx"]
  md += [q("## 表 1 nx_j60：同一摩擦上界的两个训练 seed"), "",
         "| 策略（摩擦上界） | 训练 seed | k（逐遍） | 池化 k/n | 非倒地 root 中位 (cm) | 漂移 ≥30 cm | vs M4 p（逐遍） |",
         "| --- | --- | --- | --- | --- | --- | --- |",
         "| M4（0.3–1.2 默认 DR，对照） | 0 | %s | %s | %s | — | — |" % (fmt_ks(m4nx), fmt_pool(m4nx), fmt_root(m4nx)),
         "| f20（0.2–2.0） | 0（LAB24） | %s | %s | %s | %d/%d | — |" % (
             fmt_ks(base["f20s0_nx"]), fmt_pool(base["f20s0_nx"]), fmt_root(base["f20s0_nx"]),
             sum(r["drift"] for r in base["f20s0_nx"]), sum(r["n"] - r["k"] for r in base["f20s0_nx"])),
         "| f20（0.2–2.0） | **1（本次）** | %s | %s | %s | %d/%d | %s |" % (
             fmt_ks(new["f20s1"]), fmt_pool(new["f20s1"]), fmt_root(new["f20s1"]),
             sum(r["drift"] for r in new["f20s1"]), sum(r["n"] - r["k"] for r in new["f20s1"]),
             fisher_cell(new["f20s1"], m4nx)),
         "| f30（0.2–3.0） | 0（LAB24） | %s | %s | %s | %d/%d | — |" % (
             fmt_ks(base["f30s0_nx"]), fmt_pool(base["f30s0_nx"]), fmt_root(base["f30s0_nx"]),
             sum(r["drift"] for r in base["f30s0_nx"]), sum(r["n"] - r["k"] for r in base["f30s0_nx"])),
         "| f30（0.2–3.0） | **1（本次）** | %s | %s | %s | %d/%d | %s |" % (
             fmt_ks(new["f30s1"]), fmt_pool(new["f30s1"]), fmt_root(new["f30s1"]),
             sum(r["drift"] for r in new["f30s1"]), sum(r["n"] - r["k"] for r in new["f30s1"]),
             fisher_cell(new["f30s1"], m4nx)),
         ]

  for arm, label in (("f20s2", "f20（0.2–2.0）"), ("f30s2", "f30（0.2–3.0）")):
    if any(r["n"] for r in new[arm]):
      md.append("| %s | **2（分支）** | %s | %s | %s | %d/%d | %s |" % (
          label, fmt_ks(new[arm]), fmt_pool(new[arm]), fmt_root(new[arm]),
          sum(r["drift"] for r in new[arm]), sum(r["n"] - r["k"] for r in new[arm]),
          fisher_cell(new[arm], m4nx)))
  md += [""]

  md += [q("## 表 2 f30（0.2–3.0）的 −y 扩测（零训练，复用 LAB24 ckpt）"), "",
         "| 格 | M4 k（逐遍） | f30 k（逐遍） | 池化 M4 / f30 | 非倒地 root 中位 M4 / f30 (cm) | Fisher p（逐遍） |",
         "| --- | --- | --- | --- | --- | --- |",
         "| ny_j60 | %s | %s | %s / %s | %s / %s | %s |" % (
             fmt_ks(base["M4_ny60"]), fmt_ks(new["f30ny60"]), fmt_pool(base["M4_ny60"]),
             fmt_pool(new["f30ny60"]), fmt_root(base["M4_ny60"]), fmt_root(new["f30ny60"]),
             fisher_cell(new["f30ny60"], base["M4_ny60"])),
         "| ny_j45 | %s | %s | %s / %s | %s / %s | %s |" % (
             fmt_ks(base["M4_ny45"]), fmt_ks(new["f30ny45"]), fmt_pool(base["M4_ny45"]),
             fmt_pool(new["f30ny45"]), fmt_root(base["M4_ny45"]), fmt_root(new["f30ny45"]),
             fisher_cell(new["f30ny45"], base["M4_ny45"])),
         ""]

  f20s1 = [r["k"] for r in new["f20s1"]]
  f30s1 = [r["k"] for r in new["f30s1"]]
  ny60 = [r["k"] for r in new["f30ny60"]]
  ny45 = [r["k"] for r in new["f30ny45"]]

  b1 = ("1. **f20 的 65%% 复现了吗**：seed1 逐遍 %s（池化 %s）。若每遍 ≥9/20 → 「非单调」有第二个训练 seed 支持；"
        "若 ≤4/20 → seed0（65%%）更像一次训练不稳，需要再训 seed2（讲义 §6 分支）。") % (
      "/".join(str(x) for x in f20s1) or "—", fmt_pool(new["f20s1"]))
  b2 = ("2. **f30 的 0/60 复现了吗**：seed1 逐遍 %s（池化 %s），非倒地 root 中位 %s cm。若 k ≤2/20 且 root 中位 ≤40 → "
        "3.0 档的收益不依赖单一训练 seed；若 k 明显回升（≥5/20）→ 3.0 档也只是「某一次训练的运气」。") % (
      "/".join(str(x) for x in f30s1) or "—", fmt_pool(new["f30s1"]), fmt_root(new["f30s1"]))
  b3 = ("3. **3.0 档改不改 −y**：ny_j60 逐遍 %s、ny_j45 逐遍 %s（M4 同格 %s / %s）。若 ny_j60 ≤10/20 → 抬摩擦上界也动了 −y，"
        "与 LAB23/LAB20b「−y 无收益」冲突，必须写成新发现并解释；若仍 ≥15/20 → 「摩擦档只对 −x 有效」加强。") % (
      "/".join(str(x) for x in ny60) or "—", "/".join(str(x) for x in ny45) or "—",
      fmt_ks(base["M4_ny60"]), fmt_ks(base["M4_ny45"]))

  md += [q("## 判读顺序（先看结论，再核数字）"), "", b1, b2, b3,
         "4. 任何「≤」判断都要带逐遍极差与种子集合；单遍数字不进结论（D28/D38）。",
         ""]

  data = {
    "run": str(run),
    "arms": {a: {"cell": ARMS[a], "reps": new[a]} for a in ARMS},
    "baseline": base,
  }
  out_md = run / "lab25_converge.md"
  out_md.write_text(q("\n".join(md)))
  (run / "lab25_converge.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))

  if args.figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    labels = ["M4\nseed0", "f20\nseed0", "f20\nseed1", "f30\nseed0", "f30\nseed1"]
    vals = [100.0 * r["k"] / r["n"] if r["n"] else 0 for r in
            [stats(sum([load(BASELINE["M4_nx"][0], "nx_j60")], [])),
             base["f20s0_nx"][0] if base["f20s0_nx"] else {"k": 0, "n": 0},
             new["f20s1"][0] if new["f20s1"] else {"k": 0, "n": 0},
             base["f30s0_nx"][0] if base["f30s0_nx"] else {"k": 0, "n": 0},
             new["f30s1"][0] if new["f30s1"] else {"k": 0, "n": 0}]]
    colors = ["#7f7f7f", "#1f77b4", "#1f77b4", "#d62728", "#d62728"]
    ax.bar(range(5), vals, color=colors)
    for i, v in enumerate(vals):
      ax.text(i, v + 1, "%.0f%%" % v, ha="center", fontsize=9)
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("nx_j60 down rate (%)")
    ax.set_title("friction upper bound: seed0 vs seed1", fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    groups = ["ny_j60", "ny_j45"]
    m4 = [100.0 * sum(r["k"] for r in base["M4_ny60"]) / max(1, sum(r["n"] for r in base["M4_ny60"])),
          100.0 * sum(r["k"] for r in base["M4_ny45"]) / max(1, sum(r["n"] for r in base["M4_ny45"]))]
    f3 = [100.0 * sum(r["k"] for r in new["f30ny60"]) / max(1, sum(r["n"] for r in new["f30ny60"])),
          100.0 * sum(r["k"] for r in new["f30ny45"]) / max(1, sum(r["n"] for r in new["f30ny45"]))]
    x = [0, 1]
    ax.bar([i - 0.18 for i in x], m4, width=0.36, color="#7f7f7f", label="M4 (default DR)")
    ax.bar([i + 0.18 for i in x], f3, width=0.36, color="#d62728", label="f30 (0.2-3.0, seed0 ckpt)")
    for i, v in enumerate(m4):
      ax.text(i - 0.18, v + 1, "%.0f%%" % v, ha="center", fontsize=9)
    for i, v in enumerate(f3):
      ax.text(i + 0.18, v + 1, "%.0f%%" % v, ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("down rate (%)")
    ax.set_title("-y check: does 3.0 change -y?", fontsize=10)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(args.figure_title, fontsize=11)
    fig.tight_layout()
    out = ROOT / "outputs/figures/lab25_converge.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print("[lab25] 图：%s" % out)

  print("[lab25] 表：%s" % out_md)


if __name__ == "__main__":
  main()
