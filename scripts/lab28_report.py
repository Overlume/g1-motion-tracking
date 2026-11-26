#!/usr/bin/env python
"""LAB28 报告：窗口距离曲线——同一段动作（dance1）上，距训练窗口越远完成度掉多少。

输入（scripts/lab28_window_curve.sh 生成）：
  <run>/eval/<policy>/w<SSS>/seed<k>/summary.json
  policy: m4 = M4（只训练过 fight1 S3 0–20 s）/ lab27 = LAB27（多训练了 dance1 0–20 s）

输出：
  <run>/lab28_curve.md     人读（两张表 + 逐种子配对差 + 读数顺序）
  <run>/lab28_curve.json   机器读
  outputs/figures/lab28_window_curve.png   --figure（ASCII 标签）
"""

import argparse
import json
import math
from pathlib import Path

POLICIES = ("m4", "lab27")
POLICY_CN = {"m4": "M4（只练 fight1 S3）", "lab27": "LAB27（+dance1 0–20 s）"}
POLICY_EN = {"m4": "M4 (1 clip)", "lab27": "LAB27 (2 clips)"}
REF_LAB27_W060 = 0.108  # LAB27 评估 dance1 60–80 s 的实测（4 窗口那轮的 dance 格）


def load_cell(d: Path):
    f = d / "summary.json"
    if not f.exists():
        return None
    try:
        s = json.loads(f.read_text())
    except Exception:
        return None
    return {
        "completion": float(s.get("track_completion", 0.0)),
        "surv": int(s.get("motion_frame_furthest", 0) or 0),
        "resets": int(s.get("reset_count", 0) or 0),
        "fall": bool(s.get("fall_detected", False)),
        "mpkpe": float(s.get("mpkpe_cm", 0.0) or 0.0),
    }


def collect(run: Path, starts):
    data = {}
    for p in POLICIES:
        data[p] = {}
        for s in starts:
            d = run / "eval" / p / ("w%03d" % s)
            cells = {}
            if d.is_dir():
                for sd in sorted(d.glob("seed*")):
                    try:
                        seed = int(sd.name.replace("seed", ""))
                    except ValueError:
                        continue
                    c = load_cell(sd)
                    if c is not None:
                        cells[seed] = c
            data[p][s] = cells
    return data


def stats(vals):
    vals = list(vals)
    n = len(vals)
    if n == 0:
        return None
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return {"n": n, "mean": m, "sd": sd, "min": min(vals), "max": max(vals)}


def fmt(x, nd=3):
    return "—" if x is None else (("%." + str(nd) + "f") % x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--starts", default="0 20 40 60 80 100")
    ap.add_argument("--title", default="LAB28 window distance curve (dance1)")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--figure-out", default="outputs/figures/lab28_window_curve.png")
    args = ap.parse_args()

    run = Path(args.run).resolve()
    if not run.is_dir():
        raise SystemExit("run 目录不存在：%s" % run)
    starts = [int(x) for x in args.starts.split()]
    data = collect(run, starts)

    out = {"run": str(run), "starts": starts, "policies": {}, "paired": {}}
    for p in POLICIES:
        out["policies"][p] = {}
        for s in starts:
            cells = data[p][s]
            out["policies"][p]["%03d" % s] = {
                "seeds": {str(k): v for k, v in cells.items()},
                "completion": stats([c["completion"] for c in cells.values()]),
                "surv": stats([c["surv"] for c in cells.values()]),
                "resets": stats([c["resets"] for c in cells.values()]),
                "falls": sum(1 for c in cells.values() if c["fall"]),
            }

    lines = ["# " + args.title, "",
             "- run：%s" % run,
             "- 口径：scripts/eval_tracking.py，终止检测开、启动 DR 关、999 步；完成度 = track_completion（到过的最远帧 / 总帧）",
             "- 动作：data/processed/lafan_dance1_subject1.npz（6574 帧，50 fps）；窗口 = 起点 0/20/40/60/80/100 s 各 20 s，均过 D41 地面校正闸",
             "- 距离 = 窗口起点 − 训练窗口起点（LAB27 训练窗口 = dance1 0–20 s）",
             ""]

    lines += ["## 1. 完成度随距离", "",
              "| 窗口 | 距离 | M4 | LAB27 | Δ（LAB27 − M4） |",
              "|---|---|---|---|---|"]
    for s in starts:
        row = ["%d–%d s" % (s, s + 20), "%d s" % s]
        vals = []
        for p in POLICIES:
            st = out["policies"][p]["%03d" % s]["completion"]
            row.append("—" if not st else "%s ± %s（n=%d）" % (fmt(st["mean"]), fmt(st["sd"]), st["n"]))
            vals.append(st["mean"] if st else None)
        d = None if None in vals else vals[1] - vals[0]
        row.append("—" if d is None else "%+.3f" % d)
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## 2. 存活帧与 reset 次数", "",
              "| 窗口 | M4 存活帧 | LAB27 存活帧 | M4 reset | LAB27 reset |",
              "|---|---|---|---|---|"]
    for s in starts:
        row = ["%d–%d s" % (s, s + 20)]
        for p in POLICIES:
            st = out["policies"][p]["%03d" % s]["surv"]
            row.append("—" if not st else "%d ± %d" % (round(st["mean"]), round(st["sd"])))
        for p in POLICIES:
            st = out["policies"][p]["%03d" % s]["resets"]
            row.append("—" if not st else "%.1f" % st["mean"])
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## 3. 逐种子配对差（完成度，LAB27 − M4）", ""]
    for s in starts:
        a = data["m4"][s]
        b = data["lab27"][s]
        common = sorted(set(a) & set(b))
        if not common:
            lines.append("- %d–%d s：没有可配对的种子" % (s, s + 20))
            continue
        diffs = {k: b[k]["completion"] - a[k]["completion"] for k in common}
        better = sum(1 for v in diffs.values() if v > 0.01)
        worse = sum(1 for v in diffs.values() if v < -0.01)
        lines.append("- %d–%d s：n=%d，LAB27 更好 %d / 更差 %d / 差不多 %d；逐种子 %s" % (
            s, s + 20, len(common), better, worse, len(common) - better - worse,
            ", ".join("s%d %+0.3f" % (k, diffs[k]) for k in common)))
        out["paired"]["%03d" % s] = {str(k): diffs[k] for k in common}

    # 形状判读提示（只做机械描述，不替你下结论）
    lab27 = [out["policies"]["lab27"]["%03d" % s]["completion"] for s in starts]
    lines += ["", "## 4. 曲线形状（自动描述，供你判 (A)/(B)/(C)）", ""]
    if all(lab27):
        means = [c["mean"] for c in lab27]
        lines.append("- LAB27 逐窗口完成度：%s" % ", ".join("%d s=%s" % (s, fmt(m)) for s, m in zip(starts, means)))
        mono = all(means[i] >= means[i + 1] - 0.01 for i in range(len(means) - 1))
        lines.append("- 单调不增（容差 0.01）：%s" % ("是" if mono else "否"))
        lines.append("- 0 s → 20 s 的落差：%+.3f" % (means[1] - means[0]))
        far = means[-1]
        lines.append("- 最远窗口（%d s）完成度：%s" % (starts[-1], fmt(far)))
    else:
        lines.append("- 数据不全，无法描述")

    out_json = run / "lab28_curve.json"
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    (run / "lab28_curve.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # 自检：LAB27 的 60 s 窗口应与上一轮「dance 60–80 s」实测对得上
    st60 = out["policies"]["lab27"].get("060", {}).get("completion")
    if st60:
        print("[lab28] 自检：LAB27 60 s 窗口 = %.3f（上一轮 dance 60–80 s = %.3f）" % (st60["mean"], REF_LAB27_W060))
    else:
        print("[lab28] 自检：缺 lab27 的 60 s 窗口数据")

    if args.figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=140)
            colors = {"m4": "#8c8c8c", "lab27": "#1f77b4"}
            for p in POLICIES:
                xs, ys, es = [], [], []
                for s in starts:
                    st = out["policies"][p]["%03d" % s]["completion"]
                    xs.append(s)
                    ys.append(st["mean"] if st else 0.0)
                    es.append(st["sd"] if st else 0.0)
                axes[0].errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=POLICY_EN[p], color=colors[p])
                dy = 7 if p == "lab27" else -13
                for x, y in zip(xs, ys):
                    axes[0].annotate("%.3f" % y, (x, y), textcoords="offset points", xytext=(0, dy),
                                     ha="center", fontsize=7)
            axes[0].set_xlabel("window start in dance1 (s); training window = 0-20 s for LAB27")
            axes[0].set_ylabel("track completion")
            axes[0].set_xticks(starts)
            axes[0].set_ylim(0, 1.05)
            axes[0].set_title("completion vs distance from training window")
            axes[0].legend(loc="upper right", fontsize=8)
            axes[0].grid(alpha=0.3)

            for p in POLICIES:
                ys = [out["policies"][p]["%03d" % s]["surv"]["mean"] if out["policies"][p]["%03d" % s]["surv"] else 0.0 for s in starts]
                axes[1].plot(starts, ys, marker="o", label=POLICY_EN[p], color=colors[p])
                if p == "lab27":
                    for x, y in zip(starts, ys):
                        axes[1].annotate("%d" % round(y), (x, y), textcoords="offset points", xytext=(0, 9),
                                         ha="center", fontsize=7)
            axes[1].set_xlabel("window start in dance1 (s)")
            axes[1].set_ylabel("motion frame furthest (999 = finished)")
            axes[1].set_xticks(starts)
            axes[1].set_ylim(0, 1050)
            axes[1].set_title("survival frames")
            axes[1].legend(loc="lower left", fontsize=8)
            axes[1].grid(alpha=0.3)

            fig.suptitle(args.title)
            fig.tight_layout()
            png = Path(args.figure_out)
            png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(png)
            print("[lab28] 图：%s" % png)
        except Exception as e:
            print("[lab28] 画图失败：%r" % (e,))

    print("[lab28] 表：%s" % (run / "lab28_curve.md"))
    print("[lab28] json：%s" % out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
