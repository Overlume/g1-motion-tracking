"""Summarize LAB09 generalization probe: group seed*/summary.json by combo.

Usage: environment/.venv/bin/python scripts/lab09_report.py --run runs/YYYYMMDD-NNN-lab09-generalization
"""
import argparse
import json
import statistics as st
from pathlib import Path

CLIP_KIND = {
    "train": "训练片段（in-distribution）",
    "heldout": "同片段未见窗口（held-out）",
    "unseen": "未见片段（held-out）",
}
CLIP_LABEL = {"train": "S3 000–20 s", "heldout": "S3 120–140 s", "unseen": "S2 134 s"}
RESET_GUARD = 5


def agg(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return None
    return {"median": st.median(vals), "min": min(vals), "max": max(vals), "n": len(vals)}


def fmt(a, nd=3):
    if a is None:
        return "n/a"
    return "%.*f (%.*f–%.*f)" % (nd, a["median"], nd, a["min"], nd, a["max"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    rows = []
    for d in sorted(p for p in run.iterdir() if p.is_dir() and "_" in p.name):
        policy, _, clip = d.name.partition("_")
        sums = []
        for sd in sorted(d.glob("seed*")):
            f = sd / "summary.json"
            if f.exists():
                sums.append(json.loads(f.read_text()))
        if not sums:
            continue
        rows.append({
            "combo": d.name,
            "policy": policy,
            "clip": clip,
            "n": len(sums),
            "completion": agg([s["track_completion"] for s in sums]),
            "resets": agg([s["reset_count"] for s in sums]),
            "longest_seg_s": agg([
                s["longest_segment"]["seconds"] if s.get("longest_segment") else 0.0
                for s in sums
            ]),
            "root_err_cm": agg([s["root_err_mean_cm"] for s in sums]),
            "r_mpkpe_cm": agg([s["r_mpkpe_mean_cm"] for s in sums]),
            "mpkpe_cm": agg([s["mpkpe_mean_cm"] for s in sums]),
            "falls": sum(1 for s in sums if s.get("fall_detected")),
            "terms": sorted({t for s in sums for pair in s.get("reset_terms", []) for t in pair}),
        })
    (run / "generalization.json").write_text(
        json.dumps({"run": str(run), "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = ["# LAB09 泛化探针：官方策略 vs 3000 iter 特化策略", ""]
    lines.append("口径：scripts/eval_tracking.py，终止检测开、启动 DR 关、999 步；每格 N 个种子。")
    lines.append("")
    lines.append("| 策略 | 片段 | 类型 | 完成度 | reset 中位 | 最长存活段 s | root 误差 cm | R-MPKPE cm | 摔/N | 终止原因 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        invalid = r["resets"] is not None and r["resets"]["median"] > RESET_GUARD
        root = "n/a†" if invalid else fmt(r["root_err_cm"], 2)
        rmp = "n/a†" if invalid else fmt(r["r_mpkpe_cm"], 2)
        lines.append(
            "| {p} | {clip} | {kind} | {comp} | {res:.1f} | {seg} | {root} | {rmp} | {f}/{n} | {terms} |".format(
                p=r["policy"],
                clip=CLIP_LABEL.get(r["clip"], r["clip"]),
                kind=CLIP_KIND.get(r["clip"], r["clip"]),
                comp=fmt(r["completion"], 3),
                res=r["resets"]["median"] if r["resets"] else float("nan"),
                seg=fmt(r["longest_seg_s"], 2),
                root=root,
                rmp=rmp,
                f=r["falls"],
                n=r["n"],
                terms=", ".join(r["terms"]) or "-",
            )
        )
    lines.append("")
    lines.append("† 该格中位 reset 次数 > 5：机器人几百毫秒就被重置一次，几乎没走出片头，root / pose 误差不可比（D17 规则 3）。此类格只看完成度、最长存活段与终止原因。")
    lines.append("")
    lines.append("## 读法")
    lines.append("")
    lines.append("1. 先比 **train 与 heldout 的完成度**：差得越大，说明越是“记住这个窗口”而不是“会跟踪”。")
    lines.append("2. 再看 **heldout / unseen 的最长存活段**：< 1 s 基本等于没学会跟踪新运动。")
    lines.append("3. root 误差只在 reset 少的格里读；reset 多的格写 n/a。")
    lines.append("4. 两条策略在 heldout 上都挂，不代表 m4 更差——它只是“对该窗口稍有残余泛化”。")
    (run / "generalization.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
