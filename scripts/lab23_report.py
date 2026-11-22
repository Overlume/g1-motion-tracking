#!/usr/bin/env python
"""LAB23 报告：DR 单档消融（fric / com / bias）三数口径对比。

输入（由 scripts/lab23_ablate.sh 生成）：
  <run>/arms/<arm>/perturb/rep<N>/recovery.json      nx_j60 × 20 种子 × N 遍
  <run>/arms/<arm>/perturb_ny/rep1/recovery.json     ny_j60 × 20 种子 × 1 遍
  <run>/arms/<arm>/eval/heldout/seed<k>/summary.json held-out 完成度（10 种子）

对照（已有）：
  M4   nx: runs/20261120-001-lab15-boundary20/recovery.json（rep1）+ runs/20261120-002-lab16-fliprate/rep2,3
  LAB20 nx: runs/20261122-001-lab20b-perturb/rep1..3

输出：
  <run>/lab23_ablation.md / .json
  outputs/figures/lab23_dr_ablation.png（--figure）

口径（D35/D36）：一个臂必须同时报 ①倒地数 k ②非倒地 rollout 的末尾 root 误差 ③transient 退让型占比。
Fisher 按 D35：三遍写 20 vs 20 的 p，池化 p 只作参考。
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

ARM_CN = {"fric": "摩擦 0.2–1.4", "com": "质心 ±4/6/6 cm", "bias": "编码器 ±0.02 rad"}
ARM_EN = {"fric": "friction", "com": "COM", "bias": "encoder bias"}
M4_NX = ["runs/20261120-001-lab15-boundary20/recovery.json",
         "runs/20261120-002-lab16-fliprate/rep2/recovery.json",
         "runs/20261120-002-lab16-fliprate/rep3/recovery.json"]
L20_NX = ["runs/20261122-001-lab20b-perturb/rep1/recovery.json",
          "runs/20261122-001-lab20b-perturb/rep2/recovery.json",
          "runs/20261122-001-lab20b-perturb/rep3/recovery.json"]


def load_json(f: Path):
    if not f.exists():
        return []
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return []


def rows_of(paths, root: Path, cell: str):
    out = []
    for p in paths:
        out += [r for r in load_json(root / p) if r.get("cell") == cell]
    return out


def is_yield(r):
    return r["longest_dev_s"] >= 0.5 and r["root_err_end_cm"] >= 30.0


def stats(rows):
    if not rows:
        return None
    down = [r for r in rows if r["cls"] == "down"]
    nond = [r for r in rows if r["cls"] != "down"]
    tra = [r for r in rows if r["cls"] == "transient"]
    yield_ = [r for r in tra if is_yield(r)]
    return dict(
        n=len(rows), k=len(down), k_pct=round(100.0 * len(down) / len(rows), 1),
        n_transient=len(tra), n_stable=len(rows) - len(down) - len(tra),
        n_yield=len(yield_),
        root_nondown=round(st.median([r["root_err_end_cm"] for r in nond]), 1) if nond else None,
        pzmin=round(min(r["pelvis_z_min"] for r in rows), 3),
    )


def fisher_p(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0:
        return None
    _, p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    return p


def heldout_stats(arm_dir: Path):
    vals = []
    for f in sorted(arm_dir.glob("eval/heldout/seed*/summary.json")):
        try:
            vals.append(float(json.loads(f.read_text()).get("track_completion", 0.0)))
        except Exception:
            pass
    if not vals:
        return None
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    return {"n": len(vals), "mean": m, "sd": sd, "min": min(vals), "max": max(vals)}


def fmt(x, nd=2, dash="—"):
    return dash if x is None else ("%." + str(nd) + "f") % x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    run = (root / args.run).resolve() if not Path(args.run).is_absolute() else Path(args.run)
    if not run.is_dir():
        raise SystemExit("run 目录不存在：%s" % run)
    arms_dir = run / "arms"
    arms = sorted([d.name for d in arms_dir.iterdir() if d.is_dir()]) if arms_dir.is_dir() else []
    if not arms:
        raise SystemExit("找不到任何臂：%s/arms/*" % run)

    # 基线
    base = {}
    for name, paths in (("M4", M4_NX), ("LAB20", L20_NX)):
        reps, ks, ns = [], [], []
        for p in paths:
            rows = [r for r in load_json(root / p) if r.get("cell") == "nx_j60"]
            if not rows:
                continue
            reps.append(rows)
            ks.append(sum(1 for r in rows if r["cls"] == "down"))
            ns.append(len(rows))
        if reps:
            base[name] = dict(reps=reps, ks=ks, ns=ns, stat=stats([r for x in reps for r in x]))

    table = []
    for arm in arms:
        ad = arms_dir / arm
        nx_reps = []
        for f in sorted(ad.glob("perturb/rep*/recovery.json")):
            rows = [r for r in load_json(f) if r.get("cell") == "nx_j60"]
            if rows:
                nx_reps.append(rows)
        nx_all = [r for x in nx_reps for r in x]
        ny = [r for r in load_json(ad / "perturb_ny/rep1/recovery.json") if r.get("cell") == "ny_j60"]
        hd = heldout_stats(ad)
        row = dict(arm=arm, nx=stats(nx_all), nx_ks=[sum(1 for r in x if r["cls"] == "down") for x in nx_reps],
                   nx_ns=[len(x) for x in nx_reps], ny=stats(ny), heldout=hd, p={})
        for bname, b in base.items():
            ps = []
            for i, rep in enumerate(nx_reps):
                if i < len(b["reps"]):
                    p = fisher_p(sum(1 for r in rep if r["cls"] == "down"), len(rep), b["ks"][i], b["ns"][i])
                    if p is not None:
                        ps.append(p)
            row["p"][bname] = ps
        table.append(row)

    lines = [
        "# LAB23 DR 单档消融：谁造成了「退让式不倒地」",
        "",
        "- 生成：scripts/lab23_report.py（读 recovery.json + heldout summary.json，不跑仿真）",
        "- 训练口径：热启动 model_49999、3000 iter、4096 envs、seed 0、S3 0–20 s（与 M4 / LAB20 完全一致）",
        "- 扰动口径：nx_j60 / ny_j60，20 种子 101–120 × 3 遍（ny 1 遍），终止关、8.0 s 起推 0.2 s、推后看 3 s",
        "- 三数口径（D35/D36）：①倒地数 k ②非倒地末尾 root 误差 ③transient 退让型占比（偏离 ≥0.5 s 且 root ≥30 cm）",
        "",
        "| 臂 | nx k（逐遍） | nx 池化 k/n | nx 非倒地 root 中位 (cm) | nx 退让型 | vs M4 p（逐遍） | vs LAB20 p（逐遍） | ny k/20 | held-out 完成度（10 种子） |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def prow(name, nx_stat, nx_ks, nx_ns, root_cm, nyield, pM4, pL20, ny_stat, hd):
        ks = "/".join(str(x) for x in nx_ks) if nx_ks else "—"
        pool = "%d/%d" % (nx_stat["k"], nx_stat["n"]) if nx_stat else "—"
        pm4 = "/".join(("%.3g" % p) for p in pM4) if pM4 else "—"
        pl20 = "/".join(("%.3g" % p) for p in pL20) if pL20 else "—"
        nyk = "%d/%d" % (ny_stat["k"], ny_stat["n"]) if ny_stat else "—"
        if isinstance(hd, str):
            hds = hd
        elif hd:
            hds = "%s ± %s（n=%d）" % (fmt(hd["mean"], 3), fmt(hd["sd"], 4), hd["n"])
        else:
            hds = "—"
        return "| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            name, ks, pool, fmt(root_cm, 1), nyield, pm4, pl20, nyk, hds)

    for name, b in base.items():
        stt = b["stat"]
        hd = "0.100（LAB09，10 种子）" if name == "M4" else "0.018（LAB20，10 种子）"
        lines.append(prow("%s（已有）" % name, stt, b["ks"], b["ns"],
                          stt["root_nondown"] if stt else None, stt["n_yield"] if stt else 0,
                          [], [], None, hd))
    for row in table:
        lines.append(prow(ARM_CN.get(row["arm"], row["arm"]), row["nx"], row["nx_ks"], row["nx_ns"],
                          row["nx"]["root_nondown"] if row["nx"] else None,
                          row["nx"]["n_yield"] if row["nx"] else 0,
                          row["p"].get("M4", []), row["p"].get("LAB20", []),
                          row["ny"], row["heldout"]))

    lines += [
        "",
        "说明：",
        "",
        "1. M4 的 nx 用 LAB15（rep1）+ LAB16（rep2/3）合并；held-out 完成度 0.100 来自 LAB09（10 种子）。",
        "2. LAB20 的 nx 用 LAB20b（3 遍）；held-out 0.018 来自 LAB20（10 种子）。",
        "3. p 值按 D35：逐遍 20 vs 20 的 Fisher，三遍分别给；池化 p 会因重复测量偏小，不写进结论。",
        "4. 「退让型」= transient 里偏离 ≥0.5 s 且末尾 root ≥30 cm（D36）；k=0 但有大量退让型时，不能写「鲁棒性提升」。",
        "",
    ]

    (run / "lab23_ablation.md").write_text("\n".join(lines), encoding="utf-8")
    out = {"run": str(run), "arms": table, "base": {k: v["stat"] for k, v in base.items()}}
    (run / "lab23_ablation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", run / "lab23_ablation.md")
    print("\n".join(lines[6:]))

    if args.figure:
        names, ks, roots, hds = [], [], [], []
        for name, b in base.items():
            stt = b["stat"]
            names.append(name)
            ks.append(stt["k_pct"] if stt else 0.0)
            roots.append((stt["root_nondown"] or 0.0) if stt else 0.0)
            hds.append(0.100 if name == "M4" else 0.018)
        for row in table:
            names.append(ARM_EN.get(row["arm"], row["arm"]))
            ks.append(row["nx"]["k_pct"] if row["nx"] else 0.0)
            roots.append((row["nx"]["root_nondown"] or 0.0) if row["nx"] else 0.0)
            hds.append(row["heldout"]["mean"] if row["heldout"] else 0.0)
        fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), dpi=130)
        palette = {"M4": "#7f7f7f", "LAB20": "#d62728"}
        arm_colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
        colors = []
        ai = 0
        for nm in names:
            if nm in palette:
                colors.append(palette[nm])
            else:
                colors.append(arm_colors[ai % len(arm_colors)])
                ai += 1
        axs[0].bar(names, ks, color=colors)
        axs[0].set_title("nx_j60 down rate (%)")
        axs[0].set_ylim(0, max(ks + [1.0]) * 1.25 + 1)
        for i, v in enumerate(ks):
            axs[0].text(i, v + max(ks + [1.0]) * 0.02, "%.1f" % v, ha="center", fontsize=9)
        axs[1].bar(names, roots, color=colors)
        axs[1].set_title("nx_j60 root error, non-down (cm)")
        axs[1].set_ylim(0, max(roots + [1.0]) * 1.25)
        for i, v in enumerate(roots):
            axs[1].text(i, v + max(roots + [1.0]) * 0.02, "%.1f" % v, ha="center", fontsize=9)
        axs[2].bar(names, hds, color=colors)
        axs[2].set_title("held-out completion (S3 120 s)")
        axs[2].set_ylim(0, max(hds + [0.01]) * 1.3)
        for i, v in enumerate(hds):
            axs[2].text(i, v + max(hds + [0.01]) * 0.02, "%.3f" % v, ha="center", fontsize=9)
        for a in axs:
            a.tick_params(axis="x", rotation=25)
            a.grid(axis="y", alpha=0.25)
        fig.suptitle("LAB23: DR single-factor ablation (M4 = default DR, LAB20 = all three)")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        figpath = root / "outputs/figures/lab23_dr_ablation.png"
        figpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figpath)
        print("wrote", figpath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
