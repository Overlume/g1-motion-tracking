# -*- coding: utf-8 -*-
"""LAB24 报告：摩擦档强度扫描（f14 / f20 / f30）——k、漂移、退让。

输入（scripts/lab24_fric_scan.sh 生成）：
  <run>/arms/<arm>/perturb/rep<N>/recovery.json     nx_j60 × 20 种子 × N 遍
对照：
  M4    nx: LAB15 rep1 + LAB16 rep2/3
  LAB20 nx: LAB20b rep1..3
输出：
  <run>/lab24_scan.md / .json
  outputs/figures/lab24_fric_scan.png（--figure）
口径（D35/D36/D37）：倒地数 k、非倒地末尾 root 误差、transient 三型（擦边/漂移/退让）。
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

FRIC = {"f14": "0.2–1.4", "f20": "0.2–2.0", "f30": "0.2–3.0"}
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
        return


def cell_rows(paths, root: Path, cell: str):
    out = []
    for p in paths:
        out += [r for r in load_json(root / p) if r.get("cell") == cell]
    return out


def kind_of(r):
    dev = r["longest_dev_s"]
    root_cm = r["root_err_end_cm"]
    if dev >= 0.5 and root_cm >= 30.0:
        return "yield"
    if dev < 0.5 and root_cm >= 30.0:
        return "drift"
    if dev <= 0.3:
        return "edge"
    return "mild"


def stats(rows):
    if not rows:
        return None
    down = [r for r in rows if r["cls"] == "down"]
    nond = [r for r in rows if r["cls"] != "down"]
    kinds = {"edge": 0, "drift": 0, "yield": 0, "mild": 0}
    for r in nond:
        kinds[kind_of(r)] += 1
    return dict(
        n=len(rows), k=len(down), k_pct=round(100.0 * len(down) / len(rows), 1),
        n_stable=len(rows) - len(down) - sum(1 for r in rows if r["cls"] == "transient"),
        n_transient=sum(1 for r in rows if r["cls"] == "transient"),
        kinds=kinds,
        root_nondown=round(st.median([r["root_err_end_cm"] for r in nond]), 1) if nond else None,
        pzmin=round(min(r["pelvis_z_min"] for r in rows), 3),
    )


def fisher_p(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0:
        return None
    _, p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    return p


def fmt(x, nd=1):
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    if nd == 0:
        return "%d" % x
    return ("%." + str(nd) + "f") % x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--root", default=".")
    ap.add_argument("--title", default="LAB24 摩擦档强度扫描")
    ap.add_argument("--figure-title", default="LAB24: friction range scan (nx_j60, 20 seeds x 3 reps)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    run = (root / args.run).resolve() if not Path(args.run).is_absolute() else Path(args.run)
    if not run.is_dir():
        raise SystemExit("run 目录不存在：%s" % run)
    arms_dir = run / "arms"
    arms = sorted([d.name for d in arms_dir.iterdir() if d.is_dir()]) if arms_dir.is_dir() else []
    if not arms:
        raise SystemExit("找不到任何臂：%s/arms/*" % run)

    base = {}
    for name, paths in (("M4", M4_NX), ("LAB20", L20_NX)):
        reps = []
        for p in paths:
            rows = [r for r in load_json(root / p) if r.get("cell") == "nx_j60"]
            if rows:
                reps.append(rows)
        if reps:
            base[name] = dict(reps=reps,
                              ks=[sum(1 for r in x if r["cls"] == "down") for x in reps],
                              ns=[len(x) for x in reps],
                              stat=stats([r for x in reps for r in x]))

    table = []
    for arm in arms:
        ad = arms_dir / arm
        reps = []
        for f in sorted(ad.glob("perturb/rep*/recovery.json")):
            rows = [r for r in load_json(f) if r.get("cell") == "nx_j60"]
            if rows:
                reps.append(rows)
        if not reps:
            table.append(dict(arm=arm, reps=0, stat=None, ks=[], ns=[], p={}))
            continue
        row = dict(arm=arm, reps=len(reps),
                   stat=stats([r for x in reps for r in x]),
                   ks=[sum(1 for r in x if r["cls"] == "down") for x in reps],
                   ns=[len(x) for x in reps], p={})
        for bname, b in base.items():
            ps = []
            for i, rep in enumerate(reps):
                if i < len(b["reps"]):
                    p = fisher_p(sum(1 for r in rep if r["cls"] == "down"), len(rep),
                                 b["ks"][i], b["ns"][i])
                    if p is not None:
                        ps.append(p)
            row["p"][bname] = ps
        table.append(row)

    lines = [
        "# %s" % args.title,
        "",
        "- 生成：scripts/lab24_report.py（只读 recovery.json，不跑仿真）",
        "- 训练口径：热启动 model_49999、3000 iter、4096 envs、seed 0、S3 0–20 s；com/bias 回默认",
        "- 扰动口径：nx_j60（300 N × 0.2 s），20 种子 101–120 × 3 遍，终止关、8.0 s 起推、推后看 3 s",
        "- 三数口径（D35/D36/D37）：倒地数 k、非倒地末尾 root 误差、非倒地三型（擦边 / 漂移 / 退让）",
        "",
        "| 臂 | 摩擦区间 | k（逐遍） | 池化 k/n | 非倒地 root 中位 (cm) | 擦边/漂移/退让 | pz_min (m) | vs M4 p（逐遍） |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def prow(name, fric, stt, ks, ns, ps):
        kss = "/".join(str(x) for x in ks) if ks else "—"
        pool = "%d/%d（%s%%）" % (stt["k"], stt["n"], fmt(stt["k_pct"])) if stt else "—"
        kinds = stt["kinds"] if stt else {"edge": 0, "drift": 0, "yield": 0}
        kd = "%d/%d/%d" % (kinds["edge"], kinds["drift"], kinds["yield"]) if stt else "—"
        pss = "/".join("%.3g" % p for p in ps) if ps else "—"
        return "| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            name, fric, kss, pool,
            fmt(stt["root_nondown"], 1) if stt else "—", kd,
            fmt(stt["pzmin"], 3) if stt else "—", pss)

    for name, b in base.items():
        lines.append(prow("%s（已有）" % name,
                          "默认 0.3–1.2" if name == "M4" else "0.2–1.4 + 三档",
                          b["stat"], b["ks"], b["ns"], []))

    sanity = []
    for row in table:
        arm = row["arm"]
        fric = FRIC.get(arm, "?")
        lines.append(prow(arm, fric, row["stat"], row["ks"], row["ns"], row["p"].get("M4", [])))
        if arm == "f14" and row["stat"]:
            sanity.append(row["stat"]["k"])

    lines += [
        "",
        "说明：",
        "",
        "1. f14 = LAB23 的 fric 臂（同一 ckpt），用来核对本链路是否与 LAB23 一致（应为池化 6/60、2/2/2）。",
        "2. p 值按 D35：逐遍 20 vs 20 的 Fisher；池化 p 因重复测量偏小，不写进结论。",
        "3. 「漂移型」= 非倒地但 root ≥30 cm 且偏离 <0.5 s（D37）：没摔倒、也没跟住；「退让型」= 偏离 ≥0.5 s 且 root ≥30 cm。",
        "4. 判据（D37 决定 4）：失败率 ≤10% 且非倒地 root 中位从 81 cm 压到 ≤40 cm 才算「摩擦档变强」，否则只是换了一种失败方式。",
        "",
    ]
    if sanity:
        lines.append("自检：f14 池化 k = %d（LAB23 是 6；5–6 属 D28 可见翻转，差 ≥2 才排查）。" % sanity[0])
        lines.append("")

    (run / "lab24_scan.md").write_text("\n".join(lines), encoding="utf-8")
    out = {"run": str(run), "arms": table, "base": {k: v["stat"] for k, v in base.items()}}
    (run / "lab24_scan.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", run / "lab24_scan.md")
    print("\n".join(lines[6:]))

    if args.figure:
        names, ks, roots, cols = [], [], [], []
        for name, b in base.items():
            names.append(name)
            ks.append(b["stat"]["k_pct"] if b["stat"] else 0.0)
            roots.append((b["stat"]["root_nondown"] or 0.0) if b["stat"] else 0.0)
            cols.append("#7f7f7f" if name == "M4" else "#d62728")
        palette = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
        i = 0
        for row in table:
            arm = row["arm"]
            names.append((FRIC.get(arm, arm)).replace("0.2–", "0.2-"))
            ks.append(row["stat"]["k_pct"] if row["stat"] else 0.0)
            roots.append((row["stat"]["root_nondown"] or 0.0) if row["stat"] else 0.0)
            cols.append(palette[i % len(palette)])
            i += 1
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), dpi=130)
        axs[0].bar(names, ks, color=cols)
        axs[0].set_title("nx_j60 down rate (%)")
        axs[0].axhline(10.0, ls="--", lw=1, color="k", alpha=0.5)
        axs[0].set_ylim(0, max(ks + [1.0]) * 1.3 + 1)
        for j, v in enumerate(ks):
            axs[0].text(j, v + 1.0, "%.1f" % v, ha="center", fontsize=9)
        axs[1].bar(names, roots, color=cols)
        axs[1].set_title("root error after push, non-down (cm)")
        axs[1].axhline(40.0, ls="--", lw=1, color="k", alpha=0.5)
        axs[1].set_ylim(0, max(roots + [1.0]) * 1.3)
        for j, v in enumerate(roots):
            axs[1].text(j, v + 2.0, "%.1f" % v, ha="center", fontsize=9)
        axs[2].scatter(ks, roots, c=cols, s=70)
        for j, nm in enumerate(names):
            axs[2].annotate(nm, (ks[j], roots[j]), fontsize=8, xytext=(4, 3), textcoords="offset points")
        axs[2].set_xlabel("down rate (%)")
        axs[2].set_ylabel("non-down root error (cm)")
        axs[2].set_title("trade-off: fall vs yield")
        axs[2].grid(alpha=0.3)
        for a in axs[:2]:
            a.tick_params(axis="x", rotation=20)
            a.grid(axis="y", alpha=0.25)
        fig.suptitle(args.figure_title)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        figpath = root / "outputs/figures/lab24_fric_scan.png"
        figpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figpath)
        print("wrote", figpath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
