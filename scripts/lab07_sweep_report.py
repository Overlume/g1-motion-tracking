#!/usr/bin/env python3
"""LAB07 汇总：motion_global_root_pos 权重扫描（多个单变量臂）→ comparison.md / sweep.csv / sweep.png

用法：
  python scripts/lab07_sweep_report.py --eval-dir runs/20261117-001-lab07-eval \
      --arm w0.5=runs/20261116-003-lab06-eval/A \
      --arm w1.0=runs/20261117-001-lab07-eval/w10 \
      --arm w1.5=runs/20261116-003-lab06-eval/B \
      --arm w2.5=runs/20261117-001-lab07-eval/w25

每个 --arm 指向“包含 seed*/summary.json 的目录”。所有臂必须用同一评估脚本、同一动作、同一批种子。
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

METRICS = [
    ("root_err_mean_cm", "root xy 误差 cm"),
    ("root_err_p95_cm", "root p95 cm"),
    ("mpkpe_mean_cm", "MPKPE cm"),
    ("r_mpkpe_mean_cm", "R-MPKPE cm"),
    ("joint_rmse_mean_rad", "joint_rmse rad"),
    ("joint_vel_rmse_mean", "joint_vel_rmse"),
    ("yaw_err_absmean_deg", "yaw 误差 deg"),
    ("tilt_err_mean_deg", "tilt 误差 deg"),
    ("slip_left_total_cm", "左脚滑移 cm"),
    ("track_completion", "完成度"),
]


def read_arm(path: Path):
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(path.glob("seed*/summary.json"))]


def agg(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    if not vals:
        return None
    return {"n": len(vals), "median": st.median(vals), "min": min(vals), "max": max(vals)}


def fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.2f}" if abs(v) >= 1 else f"{v:.{nd}g}"
    return str(v)


def cell(a):
    return "n/a" if a is None else f"{fmt(a['median'])} ({fmt(a['min'])}-{fmt(a['max'])})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, help="本次 LAB07 的 eval run 目录（comparison.md 写在这里）")
    ap.add_argument("--arm", action="append", default=[], required=True, metavar="LABEL=DIR",
                    help="例如 w1.0=runs/xxxx-lab07-eval/w10")
    ap.add_argument("--title", default="LAB07 权重扫描：只动 weight，std 固定 0.3")
    ap.add_argument("--xlabel", default=None, help="横轴含义；M4 用来把 weight 换成训练迭代数")
    ap.add_argument("--motion", default="data/processed/fight1_s3_000s_20s.npz")
    args = ap.parse_args()

    arms = []
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm 需要 LABEL=DIR 形式，收到 {spec!r}")
        label, path = spec.split("=", 1)
        rows = read_arm(Path(path))
        if not rows:
            raise SystemExit(f"{label}: {path} 下没有 seed*/summary.json")
        arms.append({"label": label, "dir": path, "rows": rows, "w": float(label.lstrip("w")), "n": len(rows)})
    def wfmt(w):
        return f"{w:g}"

    arms.sort(key=lambda a: a["w"])

    seeds = sorted({r["seed"] for a in arms for r in a["rows"] if r.get("seed") is not None})

    lines = []
    lines.append(f"# {args.title}")
    lines.append("")
    scales = ("一次独立的 300 iter 微调（4096 env、seed 0、从 scratch/model_49999.pt 续训）"
              if args.xlabel is None else "一次独立训练（规模见各臂 params.txt 与 cfg_snapshot.py）")
    lines.append(f"动作 `{args.motion}`；每个臂 = {scales}，"
                 f"评估种子 {', '.join(str(s) for s in seeds)}，同一脚本 scripts/eval_tracking.py（终止检测开、启动 DR 关）。")
    lines.append("")
    if args.xlabel is None:
        lines.append("唯一变量：`motion_global_root_pos` 的 weight；std 固定 0.3。")
    else:
        lines.append(f"横轴：{args.xlabel}；训练规模、种子与配置快照逐臂见下表与各 run 目录。")
    lines.append("")
    lines.append("| 臂 | " + (args.xlabel or "weight") + " | 训练 run | checkpoint | 种子数 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for a in arms:
        rp = Path(a["dir"])
        tr, ck = rp / "train_run.txt", rp / "ckpt.txt"
        trs = tr.read_text().strip() if tr.exists() else "(直接给的目录)"
        cks = Path(ck.read_text().strip()).name if ck.exists() else "(见 summary.json 的 checkpoint 字段)"
        lines.append(f"| {a['label']} | {wfmt(a['w'])} | `{trs}` | `{cks}` | {a['n']} |")
    lines.append("")

    lines.append("## 主表（中位，括号内 min-max）")
    lines.append("")
    lines.append("| 指标 | " + " | ".join(a["label"] for a in arms) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in arms) + " |")
    for key, label in METRICS:
        lines.append(f"| {label} | " + " | ".join(cell(agg(a["rows"], key)) for a in arms) + " |")
    falls = [f"{sum(1 for r in a['rows'] if r.get('fall_detected'))}/{a['n']}" for a in arms]
    lines.append("| 摔倒次数 / N | " + " | ".join(falls) + " |")
    lines.append("")

    lines.append("## 曲线：" + (args.xlabel or "weight") + " -> 中位 root xy 误差")
    lines.append("")
    lines.append("| " + (args.xlabel or "weight") + " | root 中位 cm | 相对上一档 | R-MPKPE cm | joint_rmse rad | yaw deg | 摔倒 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    prev = None
    for a in arms:
        root = agg(a["rows"], "root_err_mean_cm")
        rm, jr, yw = (agg(a["rows"], k) for k in ("r_mpkpe_mean_cm", "joint_rmse_mean_rad", "yaw_err_absmean_deg"))
        n_fall = sum(1 for r in a["rows"] if r.get("fall_detected"))
        if root is None:
            lines.append(f"| {wfmt(a['w'])} | n/a | n/a | n/a | n/a | n/a | {n_fall}/{a['n']} |")
            continue
        delta = "" if prev is None else f"{(root['median'] - prev) / prev * 100:+.1f}%"
        lines.append(f"| {wfmt(a['w'])} | {root['median']:.2f} | {delta} | {fmt(rm['median']) if rm else 'n/a'} | "
                     f"{fmt(jr['median']) if jr else 'n/a'} | {fmt(yw['median']) if yw else 'n/a'} | {n_fall}/{a['n']} |")
        prev = root["median"]
    lines.append("")

    lines.append("## 逐种子 root xy 误差 cm")
    lines.append("")
    lines.append("| 臂 | " + " | ".join(f"seed{s}" for s in seeds) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in seeds) + " |")
    for a in arms:
        byseed = {r.get("seed"): r.get("root_err_mean_cm") for r in a["rows"]}
        lines.append(f"| {a['label']} | " + " | ".join(
            (f"{byseed[s]:.2f}" if isinstance(byseed.get(s), (int, float)) else "n/a") for s in seeds) + " |")
    lines.append("")

    lines.append("## 读数提醒")
    lines.append("")
    lines.append("- 按 D11：fight 类片段只报分布（N、中位、范围、摔/不摔次数），不引用单次 fall_step 作结论。")
    lines.append("- 同配置重跑（LAB05-A 43.34 / LAB06-A 41.52）给出**评估级噪声底约 4%**；相邻两档差小于 4% 应写“不可分辨”，不要排序。")
    lines.append("- 每个臂只训练 **1 个 seed**，训练侧噪声未量化；小于约 5% 的相邻差异只能当趋势，不能当结论。")
    lines.append("- 本机 WSL2 实时钟会正负跳变（D14）；本表数字来自评估脚本的世界坐标，不受影响，但耗时读数不能跨会话比。")
    lines.append("- 权重最优点只在这个动作片段、这个 300 iter 预算上成立；外推到 M4 必须显式写出这个前提。")
    lines.append("")

    ev = Path(args.eval_dir)
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {"motion": args.motion, "seeds": seeds,
               "arms": [{"label": a["label"], "weight": a["w"], "dir": a["dir"],
                         "metrics": {k: agg(a["rows"], k) for k, _ in METRICS},
                         "falls": sum(1 for r in a["rows"] if r.get("fall_detected")), "n": a["n"]} for a in arms]}
    (ev / "comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv = ["weight,arm,metric,median,min,max,n"]
    for a in arms:
        for k, _ in METRICS:
            s = agg(a["rows"], k)
            if s:
                csv.append(f"{wfmt(a['w'])},{a['label']},{k},{s['median']},{s['min']},{s['max']},{s['n']}")
    (ev / "sweep.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs, ys, lo, hi, rm = [], [], [], [], []
        for a in arms:
            r = agg(a["rows"], "root_err_mean_cm")
            m = agg(a["rows"], "r_mpkpe_mean_cm")
            if r is None:
                continue
            xs.append(a["w"]); ys.append(r["median"]); lo.append(r["min"]); hi.append(r["max"])
            rm.append(m["median"] if m else float("nan"))
        if xs:
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
            axes[0].errorbar(xs, ys, yerr=[[y - l for y, l in zip(ys, lo)], [h - y for y, h in zip(ys, hi)]],
                             marker="o", capsize=4)
            axes[0].set_xlabel(args.xlabel or "motion_global_root_pos weight (std=0.3)")
            axes[0].set_ylabel("root xy error mean (cm)")
            axes[0].set_title("root position tracking")
            axes[0].grid(alpha=0.3)
            axes[1].plot(xs, rm, marker="s", color="tab:orange")
            axes[1].set_xlabel(args.xlabel or "weight")
            axes[1].set_ylabel("R-MPKPE (cm)")
            axes[1].set_title("pose quality (root-relative)")
            axes[1].grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(ev / "sweep.png", dpi=130)
    except Exception as e:  # 画不出图不影响表格
        print(f"[warn] 画图跳过：{e}")

    print("\n".join(lines))
    print(f"[ok] 写入 {ev}/comparison.md, comparison.json, sweep.csv"
          + (", sweep.png" if (ev / "sweep.png").exists() else ""))


if __name__ == "__main__":
    main()
