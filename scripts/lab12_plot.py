#!/usr/bin/env python
"""画 LAB12 恢复实验的分类堆叠图（每格 5 种子）。

用法：
    environment/.venv/bin/python scripts/lab12_plot.py \
        --run runs/20261119-001-lab12-recovery \
        --out outputs/figures/lab12_recovery.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["px_j120", "nx_j120", "py_j120", "ny_j120", "nx_j60", "ny_j60"]
LABELS = {
    "px_j120": "+x 600 N",
    "nx_j120": "-x 600 N",
    "py_j120": "+y 600 N",
    "ny_j120": "-y 600 N",
    "nx_j60": "-x 300 N",
    "ny_j60": "-y 300 N",
}
COLORS = {"stable": "#4c9f70", "transient": "#e2b33c", "down": "#c0392b"}
NAMES = {"stable": "stayed up", "transient": "wobbled, recovered", "down": "ended on the floor"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/20261119-001-lab12-recovery")
    ap.add_argument("--out", default="outputs/figures/lab12_recovery.png")
    ap.add_argument("--title", default="LAB12: what happens after the push (termination off, 5 seeds)")
    args = ap.parse_args()

    rows = json.loads((Path(args.run) / "recovery.json").read_text(encoding="utf-8"))
    cells = [c for c in ORDER if any(r["cell"] == c for r in rows)]
    cells += sorted(set(r["cell"] for r in rows) - set(cells))

    counts = {c: {k: 0 for k in COLORS} for c in cells}
    for r in rows:
        counts[r["cell"]][r["cls"]] += 1

    fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=150)
    bottom = [0] * len(cells)
    for k in ("stable", "transient", "down"):
        vals = [counts[c][k] for c in cells]
        ax.bar(range(len(cells)), vals, bottom=bottom, color=COLORS[k], label=NAMES[k], width=0.62)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v:
                ax.text(i, b + v / 2, str(v), ha="center", va="center", color="white", fontsize=9)
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([LABELS.get(c, c) for c in cells], fontsize=9)
    ax.set_ylabel("seeds (of 5)")
    ax.set_ylim(0, 5.4)
    ax.set_title(args.title, fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
