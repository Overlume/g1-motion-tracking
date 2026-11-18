"""Summarize M5 perturbation runs: group <dir>_<tier>/seed*/summary.json.

Usage:
  environment/.venv/bin/python scripts/m5_report.py \
      --run runs/YYYYMMDD-NNN-m5-perturb --baseline runs/YYYYMMDD-NNN-lab09-generalization/m4_train
"""
import argparse
import json
import math
import statistics as st
from pathlib import Path

TIERS = {"j30": 30.0, "j60": 60.0, "j120": 120.0}   # N*s
DIR_LABEL = {"px": "+x", "nx": "-x", "py": "+y", "ny": "-y"}
MASS_KG = 33.34
BASE_ROOT_CM = 3.88   # LAB09 m4_train 中位；仅用于“恢复慢”判定
TERM_S = 15.0         # 首个 reset 早于该秒数 → 视为“扰动后终止”


def agg(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return None
    return {"median": st.median(vals), "min": min(vals), "max": max(vals), "n": len(vals)}


def fmt(a, nd=2):
    if a is None:
        return "n/a"
    return "%.*f (%.*f-%.*f)" % (nd, a["median"], nd, a["min"], nd, a["max"])


def grade(row):
    if row["terminated_after_push"] is None:
        return "n/a"
    if row["terminated_after_push"]:
        return "失稳（扰动后终止）"
    post = row["root_post01_cm"]["median"] if row["root_post01_cm"] else None
    if post is not None and post > 3.0 * BASE_ROOT_CM:
        return "未终止但恢复慢"
    return "未失稳"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--baseline", default="")
    args = ap.parse_args()
    run = Path(args.run)

    base = None
    bdir = Path(args.baseline) if args.baseline else None
    if bdir is not None and bdir.exists():
        sums = [json.loads(f.read_text()) for f in sorted(bdir.glob("seed*/summary.json"))]
        base = {
            "n": len(sums),
            "completion": agg([s["track_completion"] for s in sums]),
            "root_err_cm": agg([s["root_err_mean_cm"] for s in sums]),
            "r_mpkpe_cm": agg([s["r_mpkpe_mean_cm"] for s in sums]),
            "resets": agg([s["reset_count"] for s in sums]),
        }

    rows = []
    for d in sorted(p for p in run.iterdir() if p.is_dir() and "_" in p.name):
        direction, _, tier = d.name.partition("_")
        sums = [json.loads(f.read_text()) for f in sorted(d.glob("seed*/summary.json"))]
        if not sums:
            continue
        first = agg([
            s["perturb"]["first_reset_after_s"] for s in sums
            if s.get("perturb") and s["perturb"].get("first_reset_after_s") is not None
        ])
        rows.append({
            "combo": d.name,
            "dir": direction,
            "dir_label": DIR_LABEL.get(direction, direction),
            "tier": tier,
            "impulse_ns": TIERS.get(tier),
            "dv_ms": round(TIERS.get(tier, 0.0) / MASS_KG, 2),
            "n": len(sums),
            "completion": agg([s["track_completion"] for s in sums]),
            "resets": agg([s["reset_count"] for s in sums]),
            "first_reset_after_s": first,
            "terminated_after_push": (first["median"] <= TERM_S) if first else None,
            "root_pre_cm": agg([s["perturb"]["root_err_pre_1s_cm"] for s in sums if s.get("perturb")]),
            "root_during_cm": agg([s["perturb"]["root_err_during_cm"] for s in sums if s.get("perturb")]),
            "root_post01_cm": agg([s["perturb"]["root_err_post_0_1s_cm"] for s in sums if s.get("perturb")]),
            "root_post12_cm": agg([s["perturb"]["root_err_post_1_2s_cm"] for s in sums if s.get("perturb")]),
            "root_mean_cm": agg([s["root_err_mean_cm"] for s in sums]),
            "r_mpkpe_cm": agg([s["r_mpkpe_mean_cm"] for s in sums]),
            "falls": sum(1 for s in sums if s.get("fall_detected")),
        })
    rows.sort(key=lambda r: (r["dir"], r["impulse_ns"] or 0))
    for r in rows:
        r["grade"] = grade(r)

    (run / "m5.json").write_text(
        json.dumps({"run": str(run), "mass_kg": MASS_KG, "baseline": base, "rows": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    missing = [d for d in ("px_j30", "px_j60", "px_j120", "nx_j30", "nx_j60", "nx_j120",
                           "py_j30", "py_j60", "py_j120", "ny_j30", "ny_j60", "ny_j120")
               if not (run / d / "seed101" / "summary.json").exists()]

    lines = ["# M5 受控外力扰动：torso_link，世界系 ±x/±y，3 档冲量 × 5 种子", ""]
    lines.append("对象：特化策略 model_52998（M4，3000 iter）。")
    lines.append("施加：torso_link 上世界系恒力，窗口 8.00-8.20 s（0.2 s）；整机 33.34 kg，dv = J / m。")
    lines.append("")
    if base:
        lines.append("基线（无扰动，LAB09 m4_train，N=%d）：完成度 %s，root 误差 %s cm，R-MPKPE %s cm。" % (
            base["n"], fmt(base["completion"], 3), fmt(base["root_err_cm"], 2), fmt(base["r_mpkpe_cm"], 2)))
        lines.append("")
    if missing:
        lines.append("**缺组合（未计入本表）**：%s —— 见 README 的补齐命令。" % ", ".join(missing))
        lines.append("")
    lines.append("| 方向 | 冲量 N*s | 等效 dv m/s | 判定 | 扰动后首个 reset s | 完成度 | 扰动中 root cm | 后 0-1 s root cm | 后 1-2 s root cm | R-MPKPE cm |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        fr = r["first_reset_after_s"]
        invalid_post = bool(fr) and fr["median"] <= 9.5
        mk = "*" if invalid_post else ""
        lines.append("| {d} | {j:.0f} | {dv:.2f} | {g} | {fr} | {comp} | {dur} | {p01}{mk} | {p12}{mk} | {rmp} |".format(
            d=r["dir_label"], j=r["impulse_ns"] or 0, dv=r["dv_ms"], g=r["grade"],
            fr=fmt(fr, 2), comp=fmt(r["completion"], 3), dur=fmt(r["root_during_cm"], 2),
            p01=fmt(r["root_post01_cm"], 2), p12=fmt(r["root_post12_cm"], 2), mk=mk,
            rmp=fmt(r["r_mpkpe_cm"], 2)))
    lines.append("")
    lines.append("## 读数规则（重要）")
    lines.append("")
    lines.append("1. **判定**看“扰动后首个 reset”：< 15 s 表示这一局被扰动打断（片子全长 19.98 s，无扰动时 reset 发生在 19.94 s 的 motion_end）。")
    lines.append("2. **完成度在“被打断”的格子里不是“跟踪了一半”**：reset 把 motion 相位归零后只剩 999-410≈590 步预算，最多只能走到第 590 帧 → 完成度天然落在 0.58-0.59。它只表示“被打断过”，不表示跟踪质量。")
    lines.append("3. 带 * 的格：扰动窗口内就 reset 了，后 0-1 s / 1-2 s 的 root 误差属于**新一回合**，不可比。")
    lines.append("4. “扰动中 root”含扰动作用瞬间；未终止的格看“后 0-1 s”是否远超基线 3.88 cm、以及“后 1-2 s”是否收敛回来。")
    lines.append("5. 每格 5 个种子；相邻档差异小于基线 min-max 时写“不可分辨”。")
    lines.append("6. 对象是**特化策略**（换窗口即崩，LAB09），抗扰结论只对这个片段、这个策略成立。")
    (run / "m5.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
        for d in ["px", "nx", "py", "ny"]:
            sub = sorted([r for r in rows if r["dir"] == d and r["impulse_ns"]], key=lambda r: r["impulse_ns"])
            if not sub:
                continue
            alive = [(r["impulse_ns"], r["root_post01_cm"]) for r in sub
                     if r["terminated_after_push"] is False and r["root_post01_cm"]]
            xs = [a[0] for a in alive]
            mid = [a[1]["median"] for a in alive]
            lo = [a[1]["median"] - a[1]["min"] for a in alive]
            hi = [a[1]["max"] - a[1]["median"] for a in alive]
            if xs:
                axes[0].errorbar(np.array(xs), np.array(mid),
                                 yerr=np.vstack([np.array(lo), np.array(hi)]),
                                 marker="o", capsize=3, label=DIR_LABEL[d])
            xs2 = [r["impulse_ns"] for r in sub]
            y2 = [0.0 if r["terminated_after_push"] else 1.0 for r in sub]
            axes[1].plot(xs2, y2, marker="s", label=DIR_LABEL[d])
        axes[0].axhline(BASE_ROOT_CM, ls=":", color="gray", lw=1, label="no-push baseline")
        axes[0].set_xlabel("impulse on torso_link (N*s over 0.2 s)")
        axes[0].set_ylabel("root xy error, 0-1 s after push (cm)")
        axes[0].set_title("recovery error (only rollouts that survived)")
        axes[1].set_xlabel("impulse on torso_link (N*s over 0.2 s)")
        axes[1].set_ylabel("survived push")
        axes[1].set_title("survived (1) vs terminated by push (0)")
        axes[1].set_ylim(-0.15, 1.15)
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run / "m5.png", dpi=130)
        lines.append("")
        lines.append("图：m5.png（左：**只画没有被扰动打断的回合**的 0-1 s root 误差，虚线为无扰动基线 3.88 cm；被打断的格不出现在左图，只出现在右图）")
    except Exception as e:
        lines.append("")
        lines.append("（画图失败：%s）" % e)

    print("\n".join(lines))


if __name__ == "__main__":
    main()
