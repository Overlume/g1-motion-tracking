#!/usr/bin/env python
"""A1 审计配图：非 down rollout 的「偏离时长 vs 末尾 root 误差」散点。

横轴 = 推后窗口内最长连续偏离（s）；纵轴 = 末尾 0.5 s 平均 root 误差（cm）。
虚线 = 退让式判据（0.5 s / 30 cm）：右上角那一团就是「跟丢了但没倒」。
只读既有 recovery.json，不跑仿真。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab_transient_audit import REGISTRY  # noqa: E402

GROUPS = {
    "LAB16 (M4)": ("M4 -x/-y 60", "#1f77b4"),
    "LAB17 (M4)": ("M4 -y 45/50", "#2ca02c"),
    "LAB19 (M4)": ("M4 +x/+y", "#9467bd"),
    "LAB19b (M4)": ("M4 +x 100", "#8c564b"),
    "LAB20b (DR)": ("DR -x/-y 60", "#d62728"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", default="outputs/figures/lab_transient_audit.png")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    plt.rcParams["font.size"] = 11
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=130)
    for label, run, cells in REGISTRY:
        rows = []
        for f in sorted((root / run).glob("rep*/recovery.json")):
            rows += json.loads(f.read_text(encoding="utf-8"))
        rows = [r for r in rows if r["cell"] in cells and r["cls"] != "down"]
        if not rows:
            continue
        name, color = GROUPS.get(label, (label, "gray"))
        ax.scatter([r["longest_dev_s"] for r in rows],
                   [r["root_err_end_cm"] for r in rows],
                   s=26, alpha=0.75, color=color, edgecolors="none",
                   label="%s (n=%d)" % (name, len(rows)))

    ax.axvline(0.5, color="gray", ls="--", lw=1)
    ax.axhline(30.0, color="gray", ls="--", lw=1)
    ax.text(0.52, 32, "yield zone:\n>=0.5 s deviation and >=30 cm root error", fontsize=9, color="gray")
    ax.text(0.06, 120, "non-down rollouts only\n(down rollouts are off-chart by definition)", fontsize=9, color="#444")
    ax.set_xlabel("longest continuous deviation after push (s)")
    ax.set_ylabel("mean root error in last 0.5 s (cm)")
    ax.set_title("A1: are the k=0 cells stable, or just yielding without falling?")
    ax.set_xlim(-0.03, 1.35)
    ax.set_ylim(-3, 160)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    out = Path(args.figure)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
