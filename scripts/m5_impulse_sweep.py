#!/usr/bin/env python
"""把"关掉终止"的两批实验（LAB12 + LAB13）画成 冲量 -> 最终倒地比例 的曲线。

只收录**用 --ignore-terminations 重跑过**的格（有 ground truth 的），
不用姿态审计的 collapse 替代（那是下限，口径不同）。

用法：
    environment/.venv/bin/python scripts/m5_impulse_sweep.py \
        --run runs/20261119-001-lab12-recovery \
        --run runs/20261119-002-lab13-mid-impulse \
        --out outputs/figures/m5_impulse_sweep.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CELL_RE = re.compile(r"^(p|n)(x|y)_j(\d+)$")
DIR_LABEL = {("p", "x"): "+x", ("n", "x"): "-x", ("p", "y"): "+y", ("n", "y"): "-y"}
COLOR = {"+x": "#1f77b4", "-x": "#ff7f0e", "+y": "#2ca02c", "-y": "#d62728"}


def load(runs):
    cells = {}
    for run in runs:
        f = Path(run) / "recovery.json"
        if not f.exists():
            raise SystemExit("找不到 %s" % f)
        local = {}
        for r in json.loads(f.read_text(encoding="utf-8")):
            m = CELL_RE.match(r["cell"])
            if not m:
                continue
            d = DIR_LABEL[(m.group(1), m.group(2))]
            imp = int(m.group(3))
            c = local.setdefault((d, imp), {"n": 0, "down": 0, "transient": 0, "stable": 0})
            c["n"] += 1
            c[r["cls"]] += 1
        # 同一 (方向, 冲量) 出现在多个 run 时，保留种子数更多的那个
        for key, c in local.items():
            if key not in cells or c["n"] > cells[key]["n"]:
                cells[key] = c
    return cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", default="outputs/figures/m5_impulse_sweep.png")
    args = ap.parse_args()

    cells = load(args.run)
    for (d, imp) in sorted(cells, key=lambda k: (k[0], k[1])):
        c = cells[(d, imp)]
        print("| %s | %d | %d/%d | %d | %d |" % (d, imp, c["down"], c["n"], c["transient"], c["stable"]))

    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
    # 主线：-x / -y（各有 4 个冲量点）
    for d in ("-x", "-y"):
        pts = sorted((imp, cells[(d, imp)]) for (dd, imp) in cells if dd == d)
        xs = [p[0] for p in pts]
        ys = [p[1]["down"] / p[1]["n"] for p in pts]
        name = {"-x": "-x (braking, against motion)", "-y": "-y (lateral)"}[d]
        ax.plot(xs, ys, "-o", color=COLOR[d], label=name, lw=1.8, ms=7, zorder=3)
        for x, y, p in zip(xs, ys, pts):
            dy = 9 if y > 0.02 else -14
            ax.annotate("%d/%d" % (p[1]["down"], p[1]["n"]), (x, y), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=7, color=COLOR[d])
    # 单点：+x / +y 只在 120 N·s 有 ground truth（+y 另有 60）
    for d in ("+x", "+y"):
        pts = sorted((imp, cells[(d, imp)]) for (dd, imp) in cells if dd == d)
        xs = [p[0] for p in pts]
        ys = [p[1]["down"] / p[1]["n"] for p in pts]
        ax.plot(xs, ys, "s", color=COLOR[d], ms=8, label="%s (single impulse tested)" % d, zorder=3)
        for x, y, p in zip(xs, ys, pts):
            ax.annotate("%d/%d" % (p[1]["down"], p[1]["n"]), (x, y), textcoords="offset points",
                        xytext=(10, -3), ha="left", fontsize=7, color=COLOR[d])

    ax.set_xscale("log")
    ax.set_xticks([40, 45, 60, 90, 120])
    ax.set_xticklabels(["40", "45", "60", "90", "120"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("impulse on torso_link (N·s; 0.2 s window)")
    ax.set_ylabel("fraction of seeds that ended on the floor")
    ax.set_ylim(-0.12, 1.12)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ns = sorted({c["n"] for c in cells.values()})
    ax.set_title("M5/LAB12/LAB13/LAB15: outcome with termination off (N = %s seeds per point)"
                 % "/".join(str(n) for n in ns), fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    big = [n for n in sorted({c["n"] for c in cells.values()}) if n > 5]
    note = ("60/80 N\u00b7s: N=%d (1 seed = %.0f%%); others: N=5 (1 seed = 20%%)"
            % (big[0], 100.0 / big[0]) if big else "1 seed = 20%")
    ax.text(0.99, 0.02, note, transform=ax.transAxes, ha="right", fontsize=7, color="0.4")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
