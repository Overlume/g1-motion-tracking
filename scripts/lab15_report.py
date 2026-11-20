#!/usr/bin/env python
"""LAB15 统计：每格倒地率 + 95% 置信区间（Clopper–Pearson）+ 两向/两档 Fisher 精确检验 + 图。

输入：run 目录下的 recovery.json（scripts/lab12_recovery_report.py 的产物）
输出：<run>/boundary20.md、<run>/boundary20.json，以及可选的图。

用法：
    environment/.venv/bin/python scripts/lab15_report.py \
        --dir runs/202611xx-NNN-lab15-boundary20 \
        --figure outputs/figures/lab15_boundary20.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, fisher_exact

CELL_RE = re.compile(r"^(p|n)(x|y)_j(\d+)$")
DIR_LABEL = {("p", "x"): "+x", ("n", "x"): "-x", ("p", "y"): "+y", ("n", "y"): "-y"}
COLOR = {"+x": "#1f77b4", "-x": "#ff7f0e", "+y": "#2ca02c", "-y": "#d62728"}
MARK = {"+x": "s", "-x": "o", "+y": "^", "-y": "D"}
FORCE_N = lambda imp: imp / 0.2  # 冲量 N*s -> 0.2 s 方波下的力 N

# 5 种子基线（各自 run 目录见下），用于对照"加种子后是否移动"
BASELINE = {
    ("-x", 60): (1, 5, "runs/20261119-001-lab12-recovery"),
    ("-y", 60): (4, 5, "runs/20261119-001-lab12-recovery"),
    ("-x", 80): (5, 5, "runs/20261119-001-lab13b-true-mid"),
    ("-y", 80): (5, 5, "runs/20261119-001-lab13b-true-mid"),
}


def cp_interval(k: int, n: int, alpha: float = 0.05):
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def load(run: Path):
    f = run / "recovery.json"
    if not f.exists():
        raise SystemExit("找不到 %s（先跑评估：bash scripts/lab15_boundary20.sh run）" % f)
    rows = json.loads(f.read_text(encoding="utf-8"))
    cells = {}
    for r in rows:
        m = CELL_RE.match(r.get("cell", ""))
        if not m:
            continue
        d = DIR_LABEL[(m.group(1), m.group(2))]
        imp = int(m.group(3))
        c = cells.setdefault((d, imp), {"down": [], "transient": [], "stable": []})
        cls = r.get("cls")
        if cls in c:
            c[cls].append(int(r.get("seed", 0)))
    return cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--figure", default=None)
    args = ap.parse_args()

    run = Path(args.dir)
    cells = load(run)
    if not cells:
        raise SystemExit("recovery.json 里没有可解析的格子")

    keys = sorted(cells, key=lambda k: (k[0], k[1]))
    stats = {}
    lines = ["# LAB15 边界 20 种子统计", "",
             "- 输入：`%s`" % run,
             "- 判分：与 LAB12 相同（scripts/lab12_recovery_report.py；stable / transient / down）",
             "- 目的：把边界两格（−x/−y 的 60 与 80 N*s）从 5 种子加到 20 种子，倒地率分辨率 20% -> 5%",
             "",
             "## 每格倒地率",
             "",
             "| 格 | 冲量 | 力 | N | down | transient | stable | 倒地率 | 95% CI (Clopper–Pearson) | 5 种子基线 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]

    for key in keys:
        d, imp = key
        c = cells[key]
        n = len(c["down"]) + len(c["transient"]) + len(c["stable"])
        k = len(c["down"])
        lo, hi = cp_interval(k, n)
        base = BASELINE.get((d, imp))
        base_s = "—" if base is None else "%d/%d" % (base[0], base[1])
        stats[key] = {"dir": d, "impulse": imp, "force_n": int(FORCE_N(imp)), "n": n, "down": k,
                      "transient": len(c["transient"]), "stable": len(c["stable"]),
                      "rate": (k / n if n else 0.0), "ci": [lo, hi],
                      "down_seeds": sorted(c["down"]), "baseline": base_s}
        lines.append("| `%s_j%d` | %d N*s | %d N | %d | %d | %d | %d | %.0f%% | [%.0f%%, %.0f%%] | %s |"
                     % (("n" if d[0] == "-" else "p") + d[1], imp, imp, int(FORCE_N(imp)),
                        n, k, len(c["transient"]), len(c["stable"]), 100 * k / n,
                        100 * lo, 100 * hi, base_s))

    lines += ["", "## 检验（Fisher 精确检验，双侧）", "",
              "| 对比 | a 倒/n | b 倒/n | p 值 | 判定（α=0.05） |", "| --- | --- | --- | --- | --- |"]
    pairs = [(("-x", 60), ("-y", 60)), (("-x", 80), ("-y", 80)),
             (("-x", 60), ("-x", 80)), (("-y", 60), ("-y", 80))]
    tests = []
    for a, b in pairs:
        if a not in stats or b not in stats:
            continue
        sa, sb = stats[a], stats[b]
        if sa["n"] == 0 or sb["n"] == 0:
            continue
        table = [[sa["down"], sa["n"] - sa["down"]], [sb["down"], sb["n"] - sb["down"]]]
        try:
            _, p = fisher_exact(table)
        except Exception:
            p = float("nan")
        verdict = "差异显著" if p < 0.05 else "差异不显著"
        tests.append({"a": list(a), "b": list(b), "p": p, "table": table})
        lines.append("| %s %d vs %s %d | %d/%d | %d/%d | %.4g | %s |"
                     % (a[0], a[1], b[0], b[1], sa["down"], sa["n"], sb["down"], sb["n"], p, verdict))

    lines += ["",
              "## 读法",
              "",
              "1. **CI 重叠不等于没差异**，但 CI 完全不重叠已经是强证据；判断两向强弱看 Fisher p，而不是看百分比大小。",
              "2. 每格 N=20 时，p=0.5 附近的 95% CI 约 ±0.20；p=0.05 附近约 [0.00, 0.25]。所以 `20/20` 的 95% 下界约 0.83，"
              "仍不能说 100%。",
              "3. 若 80 档出现恢复（哪怕 1 次），“5/5 全倒从 80 起”必须改写成带 N 的比率（D25 第 3 条）。",
              "4. 与 5 种子基线的差异若在 CI 内，说明基线没问题、只是精度低；若跑到 CI 外，先怀疑环境差异而不是“新发现”。",
              ""]

    (run / "boundary20.md").write_text("\n".join(lines), encoding="utf-8")
    out_json = {"dir": str(run), "cells": {("%s_j%d" % k): v for k, v in stats.items()}, "tests": tests}
    (run / "boundary20.json").write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.figure:
        imp_vals = sorted({k[1] for k in stats})
        fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=160)
        for d in ("-x", "-y"):
            xs, ys, lo_e, hi_e = [], [], [], []
            ns_used = 0
            for imp in imp_vals:
                s = stats.get((d, imp))
                if not s or s["n"] == 0:
                    continue
                xs.append(imp)
                ns_used = s["n"]
                ys.append(100 * s["rate"])
                lo_e.append(100 * (s["rate"] - s["ci"][0]))
                hi_e.append(100 * (s["ci"][1] - s["rate"]))
            if xs:
                ax.errorbar(xs, ys, yerr=[lo_e, hi_e], marker=MARK[d], color=COLOR[d],
                            label="%s (N=%d)" % (d, ns_used), capsize=3, lw=1.6)
        ax.set_xlabel("impulse on torso_link (N*s; 0.2 s window)")
        ax.set_ylabel("share of seeds that ended on the floor (%)")
        ax.set_title("LAB15: boundary cells with 20 seeds (termination off, 3 s after push)")
        ax.set_ylim(-4, 104)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure)
        print("图：%s" % args.figure)

    print("\n".join(lines[:24]))
    print("\n写盘：%s、%s" % (run / "boundary20.md", run / "boundary20.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
