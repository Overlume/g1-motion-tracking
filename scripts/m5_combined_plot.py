#!/usr/bin/env python
"""M5 combined figure: merge several perturbation runs (e.g. coarse sweep + fine +y sweep).

Usage:
  environment/.venv/bin/python scripts/m5_combined_plot.py \
      --runs runs/20261117-005-m5-perturb runs/20261118-001-m5-threshold-refine \
      --out outputs/figures/m5_perturbation.png
"""
import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

DIRS = ["px", "nx", "py", "ny"]
LABEL = {"px": "+x", "nx": "-x", "py": "+y", "ny": "-y"}
MASS_KG = 33.34
TERM_S = 15.0     # first reset earlier than this = terminated by the push
BASE_ROOT_CM = 3.88


def tier_impulse(tier):
    n = tier[1:] if tier[:1] == "j" else ""
    try:
        return float(n)
    except ValueError:
        return None


def collect(runs):
    cells = {}
    for run in runs:
        for d in sorted(Path(run).iterdir()):
            if not d.is_dir() or "_" not in d.name:
                continue
            direction, _, tier = d.name.partition("_")
            imp = tier_impulse(tier)
            if imp is None or direction not in LABEL:
                continue
            sums = [json.loads(f.read_text()) for f in sorted(d.glob("seed*/summary.json"))]
            if not sums:
                continue
            survived, terminated = [], 0
            for s in sums:
                p = s.get("perturb") or {}
                fr = p.get("first_reset_after_s")
                post = p.get("root_err_post_0_1s_cm")
                if fr is not None and fr < TERM_S:
                    terminated += 1
                elif post is not None:
                    survived.append(post)
            cells[(direction, imp)] = {
                "n": len(sums),
                "n_term": terminated,
                "post": survived,
                "src": str(d),
            }
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/figures/m5_perturbation.png")
    ap.add_argument("--dpi", type=int, default=130)
    args = ap.parse_args()

    cells = collect(args.runs)
    if not cells:
        raise SystemExit("no cells found")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    for d in DIRS:
        pts = sorted((imp, c) for (dd, imp), c in cells.items() if dd == d)
        xs, mid, lo, hi, hollow = [], [], [], [], []
        for imp, c in pts:
            if not c["post"]:
                continue
            xs.append(imp)
            mid.append(st.median(c["post"]))
            lo.append(st.median(c["post"]) - min(c["post"]))
            hi.append(max(c["post"]) - st.median(c["post"]))
            hollow.append(c["n_term"] > 0)
        if xs:
            axes[0].errorbar(np.array(xs), np.array(mid),
                             yerr=np.vstack([np.array(lo), np.array(hi)]),
                             marker="o", capsize=3, label=LABEL[d])
            for x, y, h in zip(xs, mid, hollow):
                if h:
                    axes[0].plot([x], [y], "o", mfc="white", mec="black", ms=9, mew=1.2)
        xs2 = [imp for imp, _ in pts]
        frac = [1.0 - c["n_term"] / c["n"] for _, c in pts]
        axes[1].plot(xs2, frac, marker="s", label=LABEL[d])

    axes[0].axhline(BASE_ROOT_CM, ls=":", color="gray", lw=1, label="no-push baseline 3.88 cm")
    axes[0].set_xscale("log")
    axes[0].set_xticks([15, 20, 25, 30, 60, 120])
    axes[0].set_xticklabels(["15", "20", "25", "30", "60", "120"])
    axes[0].set_xlabel("impulse on torso_link (N*s over 0.2 s)")
    axes[0].set_ylabel("root xy error 0-1 s after push (cm)")
    axes[0].set_title("recovery error (survivors only; hollow = some seeds terminated)")

    axes[1].set_xscale("log")
    axes[1].set_xticks([15, 20, 25, 30, 60, 120])
    axes[1].set_xticklabels(["15", "20", "25", "30", "60", "120"])
    axes[1].set_xlabel("impulse on torso_link (N*s over 0.2 s)")
    axes[1].set_ylabel("fraction of seeds that survived")
    axes[1].set_ylim(-0.05, 1.08)
    axes[1].set_title("survival fraction (5 seeds per cell)")
    for ax in axes:
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print("cells:", len(cells))
    for (d, imp), c in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        med = st.median(c["post"]) if c["post"] else float("nan")
        print("%-3s %5.0f N*s  n=%d  terminated=%d  median 0-1s=%.1f cm  (%s)"
              % (d, imp, c["n"], c["n_term"], med, c["src"].split("/")[-2]))
    print("wrote", out)


if __name__ == "__main__":
    main()
