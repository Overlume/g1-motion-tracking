#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LAB26：−y 的 40/45/50/55/60 N·s 曲线（f30 = 摩擦 0.2–3.0，零训练）——定位 3.0 档的 50% 交点。

读：
  * 本次 run：rep{1,2,3}/recovery.json（cell = ny_j50 / ny_j55）
  * M4：40 用 LAB13（20261119-002，5 种子）、45/50 用 LAB17（20261120-003）、60 用 LAB15 + LAB16 rep2/3
  * f30：45/60 用 LAB25（20261124-001）的 f30ny45 / f30ny60
输出：<run>/lab26_ymid.md|json、outputs/figures/lab26_ymid.png
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CELLS = [(40, "ny_j40"), (45, "ny_j45"), (50, "ny_j50"), (55, "ny_j55"), (60, "ny_j60")]
NEW_CELLS = (50, 55)


def base_paths() -> dict:
    lab13 = [RUNS / "20261119-002-lab13-mid-impulse/recovery.json"]
    lab17 = [RUNS / ("20261120-003-lab17-y-transition/rep%d/recovery.json" % i) for i in (1, 2, 3)]
    m4_60 = [RUNS / "20261120-001-lab15-boundary20/recovery.json",
             RUNS / "20261120-002-lab16-fliprate/rep2/recovery.json",
             RUNS / "20261120-002-lab16-fliprate/rep3/recovery.json"]
    f30 = {
        45: [RUNS / ("20261124-001-lab25-converge/arms/f30ny45/perturb/rep%d/recovery.json" % i) for i in (1, 2, 3)],
        60: [RUNS / ("20261124-001-lab25-converge/arms/f30ny60/perturb/rep%d/recovery.json" % i) for i in (1, 2, 3)],
    }
    return {"M4": {40: lab13, 45: lab17, 50: lab17, 60: m4_60}, "f30": f30}


def load(paths) -> list:
    out = []
    for p in paths:
        if not Path(p).exists():
            continue
        try:
            rows = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        out += rows if isinstance(rows, list) else []
    return out


def cell_reps(paths, cell) -> list:
    reps = []
    for p in paths:
        rows = [r for r in load([p]) if r.get("cell") == cell]
        if rows:
            reps.append(rows)
    return reps


def agg(rows: list) -> dict:
    if not rows:
        return {}
    k = sum(1 for r in rows if r.get("cls") == "down")
    nond = [r for r in rows if r.get("cls") != "down"]
    kinds = {"edge": 0, "drift": 0, "yield": 0, "mild": 0}
    for r in nond:
        dev = abs(r.get("longest_dev_s") or 0.0)
        root = r.get("root_err_end_cm") or 0.0
        if dev >= 0.5 and root >= 30:
            kinds["yield"] += 1
        elif dev < 0.5 and root >= 30:
            kinds["drift"] += 1
        elif dev <= 0.3:
            kinds["edge"] += 1
        else:
            kinds["mild"] += 1
    return {
        "n": len(rows), "k": k, "rate": k / len(rows),
        "root": round(st.median([r["root_err_end_cm"] for r in nond]), 1) if nond else None,
        "kinds": kinds, "n_nondown": len(nond),
        "pz": round(min(r["pelvis_z_min"] for r in rows), 3),
    }


def fmt_ks(reps: list) -> str:
    if not reps:
        return "—"
    ks = [sum(1 for r in rep if r.get("cls") == "down") for rep in reps]
    return "/".join(str(k) for k in ks)


def fmt_pool(reps: list) -> str:
    rows = [r for rep in reps for r in rep]
    if not rows:
        return "—"
    a = agg(rows)
    return "%d/%d，%d%%" % (a["k"], a["n"], round(100 * a["rate"]))


def cell_txt(reps: list) -> str:
    rows = [r for rep in reps for r in rep]
    if not rows:
        return "—"
    if len(reps) <= 1:
        a = agg(rows)
        return "%d/%d（%d%%）" % (a["k"], a["n"], round(100 * a["rate"]))
    return "%s（%s）" % (fmt_ks(reps), fmt_pool(reps))


def pool_rate(reps: list):
    rows = [r for rep in reps for r in rep]
    return agg(rows)["rate"] if rows else None


def crossing(points: list):
    """points = [(j, rate)]；线性插值找 50%（只在前一个点 <50% 时插值）。"""
    pts = sorted([(j, r) for j, r in points if r is not None])
    prev = None
    for j, r in pts:
        if r >= 0.5:
            if prev is None:
                return float(j)
            pj, pr = prev
            if r == pr:
                return float(j)
            return pj + (0.5 - pr) * (j - pj) / (r - pr)
        prev = (j, r)
    return None


def fisher_f30_vs_m4(f30_reps, m4_reps) -> str:
    out = []
    for a, b in zip(f30_reps, m4_reps):
        ka = sum(1 for r in a if r.get("cls") == "down")
        kb = sum(1 for r in b if r.get("cls") == "down")
        _, p = fisher_exact([[ka, len(a) - ka], [kb, len(b) - kb]])
        out.append("%.3g" % p)
    return "/".join(out) if out else "—"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--title", default="LAB26 −y 的 50/55 N·s：定位 3.0 档的交点")
    ap.add_argument("--figure-title", default="LAB26: -y 40/45/50/55/60 N s, f30 (0.2-3.0) vs M4")
    args = ap.parse_args()

    run = Path(args.run)
    if not run.is_absolute():
        run = ROOT / run
    if not run.is_dir():
        raise SystemExit("run 目录不存在：%s" % run)

    base = base_paths()
    new = {j: cell_reps([run / ("rep%d/recovery.json" % i) for i in (1, 2, 3)],
                        dict(CELLS)[j]) for j in NEW_CELLS}

    lines = [bt("# %s" % args.title), "",
             "- run：%s" % run.relative_to(ROOT),
             "- 口径：root 中位 = 三遍非倒地汇合后中位；退让 = dev ≥0.5 s 且 root ≥30 cm；漂移 = dev <0.5 s 且 root ≥30 cm。",
             ""]

    t1 = ["## 表 1 −y 倒地率曲线（k 逐遍 / 池化）", "",
          "| 策略 | " + " | ".join("%d N·s" % j for j, _ in CELLS) + " | 50% 交点（线性插值） |",
          "| --- |" + " --- |" * len(CELLS) + " --- |"]
    curves = {}
    for pol in ("M4", "f30"):
        reps_by_j, cells_txt, pts = {}, [], []
        for j, cell in CELLS:
            if pol == "f30" and j in NEW_CELLS:
                reps = new[j]
            else:
                reps = cell_reps(base[pol].get(j, []), cell)
            reps_by_j[j] = reps
            cells_txt.append(cell_txt(reps))
            pts.append((j, pool_rate(reps)))
        cr = crossing(pts)
        curves[pol] = {"reps": reps_by_j, "points": pts, "crossing": cr}
        t1.append("| %s | %s | %s |" % (pol, " | ".join(cells_txt), "%.1f N·s" % cr if cr else "—"))
    lines += t1 + [""]

    t2 = ["## 表 2 非倒地 root 中位 / 三型（退让占比）", "",
          "| 策略 | " + " | ".join("%d N·s" % j for j, _ in CELLS) + " |",
          "| --- |" + " --- |" * len(CELLS)]
    for pol in ("M4", "f30"):
        cells_txt = []
        for j, _cell in CELLS:
            reps = curves[pol]["reps"][j]
            rows = [r for rep in reps for r in rep]
            if not rows:
                cells_txt.append("—")
                continue
            a = agg(rows)
            if a["root"] is None:
                cells_txt.append("全倒")
                continue
            y = a["kinds"]["yield"]
            share = "%d/%d" % (y, a["n_nondown"]) if a["n_nondown"] else "—"
            cells_txt.append("root %s cm；退让 %s；漂移 %d" % (a["root"], share, a["kinds"]["drift"]))
        t2.append("| %s | %s |" % (pol, " | ".join(cells_txt)))
    lines += t2 + [""]

    tips = ["## 读数与判定", ""]
    for j in NEW_CELLS:
        reps = new[j]
        cell = dict(CELLS)[j]
        m4 = cell_reps(base["M4"].get(j, []), cell)
        if reps and m4:
            tips.append("- **%s**：f30 逐遍 %s（池化 %s）；M4 逐遍 %s（池化 %s）；逐遍 Fisher p = %s。"
                        % (cell, fmt_ks(reps), fmt_pool(reps), fmt_ks(m4), fmt_pool(m4),
                           fisher_f30_vs_m4(reps, m4)))
        elif reps:
            tips.append("- **%s**：f30 逐遍 %s（池化 %s）；M4 无同格数据（只和 40/45/60 的曲线比）。"
                        % (cell, fmt_ks(reps), fmt_pool(reps)))
        else:
            tips.append("- **%s**：本次 run 缺这一格（检查 LAB26_CELLS 与 params.txt）。" % cell)
    cr_m4, cr_f30 = curves["M4"]["crossing"], curves["f30"]["crossing"]
    if cr_m4 and cr_f30:
        tips.append("- 交点：M4 ≈ **%.1f N·s**（自检应 ≈44，LAB17 为 44.3）、f30 ≈ **%.1f N·s** → 3.0 档把 −y 边界推高 ≈ **%.1f N·s**（线性插值，仅作量级）。"
                    % (cr_m4, cr_f30, cr_f30 - cr_m4))
    if cr_f30:
        if cr_f30 < 50:
            tips.append("- 判读：落在讲义 §0.3 的 **C**（交点 45–50）→ LAB25 的「−y 有效」幅度需下调。")
        elif cr_f30 < 55:
            tips.append("- 判读：落在讲义 §0.3 的 **A**（交点 50–55）→ 3.0 档把 −y 边界推高 ≈7–10 N·s。")
        else:
            tips.append("- 判读：落在讲义 §0.3 的 **B**（交点 55–60）→ 曲线在 55→60 很陡，写「55 仍 0–5/20、60 全倒」。")
    lines += tips + [""]

    lines += ["## 方法", "",
              "- 两格 ny_j50（0,−250,0）与 ny_j55（0,−275,0），20 种子 101–120 × 3 遍；终止关、8.0 s 起推 0.2 s。",
              "- 只换冲量、不换 ckpt：全部用 LAB24 f30（摩擦 0.2–3.0）的 model_52998.pt，sha 1119f986…455ba0。",
              "- 交点 = 「第一个池化倒地率 ≥50% 的格」与其前一格之间的线性插值；20 种子的不确定性见逐遍 k（D39）。",
              "- M4 的 40 N·s 只有 5 个种子（LAB13，0/5），用于把交点锚在 40–45 之间；其余格均为 20 种子 × 3 遍。",
              ""]

    data = {"run": str(run),
            "series": {pol: {"crossing": curves[pol]["crossing"],
                             "points": curves[pol]["points"],
                             "cells": {str(j): curves[pol]["reps"][j] for j, _ in CELLS}}
                       for pol in ("M4", "f30")}}
    (run / "lab26_ymid.md").write_text(bt("\n".join(lines)), encoding="utf-8")
    (run / "lab26_ymid.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[lab26] 表：%s" % (run / "lab26_ymid.md"))

    if args.figure:
        js = [j for j, _ in CELLS]
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
        ax = axes[0]
        for pol, color, marker in (("M4", "#7f7f7f", "o"), ("f30", "#d62728", "s")):
            ys = [100 * r if r is not None else float("nan") for _, r in curves[pol]["points"]]
            ax.plot(js, ys, marker=marker, color=color, label=pol)
        ax.axhline(50, ls="--", lw=1, color="#555")
        ax.set_xticks(js)
        ax.set_xlabel("-y impulse (N s)")
        ax.set_ylabel("down rate (%)")
        ax.set_ylim(0, 105)
        ax.set_title("down rate vs impulse", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

        ax = axes[1]
        for pol, color, marker in (("M4", "#7f7f7f", "o"), ("f30", "#d62728", "s")):
            ys = []
            for j, _ in CELLS:
                rows = [r for rep in curves[pol]["reps"][j] for r in rep]
                root = agg(rows)["root"] if rows else None
                ys.append(root if root is not None else float("nan"))
            ax.plot(js, ys, marker=marker, color=color, label=pol)
        ax.set_xticks(js)
        ax.set_xlabel("-y impulse (N s)")
        ax.set_ylabel("non-down root error median (cm)")
        ax.set_title("tracking error of survivors", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.suptitle(args.figure_title, fontsize=11)
        fig.tight_layout()
        out = ROOT / "outputs/figures/lab26_ymid.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130)
        print("[lab26] 图：%s" % out)
    return 0


def bt(s: str) -> str:
    return s.replace("`", chr(96))


if __name__ == "__main__":
    raise SystemExit(main())
