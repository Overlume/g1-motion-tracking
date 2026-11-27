#!/usr/bin/env python
"""LAB29 报告：全段覆盖训练（7 段）vs M4（1 段）vs LAB27（2 段）。"""

import argparse
import json
import math
from pathlib import Path

BT = chr(96)
POLICIES = ("m4", "lab27", "lab29")
POLICY_EN = {"m4": "M4 (1 clip)", "lab27": "LAB27 (2 clips)", "lab29": "LAB29 (7 clips)"}
WINDOWS = ["train", "heldout", "unseen", "w000", "w020", "w040", "w060", "w080", "w100"]
WIN_CN = {
    "train": "fight1 S3 0 s（训练窗口）",
    "heldout": "fight1 S3 120 s（同片段未见）",
    "unseen": "fight1 S2 134 s（换片段）",
    "w000": "dance1 0–20 s",
    "w020": "dance1 20–40 s",
    "w040": "dance1 40–60 s",
    "w060": "dance1 60–80 s",
    "w080": "dance1 80–100 s",
    "w100": "dance1 100–120 s",
}
DANCE = ["w000", "w020", "w040", "w060", "w080", "w100"]


def load_cell(d):
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


def collect_dir(d):
    cells = {}
    if not d.is_dir():
        return cells
    for sd in sorted(d.glob("seed*")):
        try:
            seed = int(sd.name.replace("seed", ""))
        except ValueError:
            continue
        c = load_cell(sd)
        if c is not None:
            cells[seed] = c
    return cells


def collect(run, b28, b27):
    data = {p: {} for p in POLICIES}
    for w in WINDOWS:
        data["lab29"][w] = collect_dir(run / "eval" / "lab29" / w)
        for p in ("m4", "lab27"):
            base = b28 if w.startswith("w") else b27
            data[p][w] = collect_dir(base / "eval" / p / w)
    return data


def stats(vals):
    vals = list(vals)
    n = len(vals)
    if n == 0:
        return None
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return {"n": n, "mean": m, "sd": sd, "min": min(vals), "max": max(vals)}


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None

    def rank(vs):
        order = sorted(range(n), key=lambda i: vs[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--b28", default="runs/20261126-001-lab28-window-curve")
    ap.add_argument("--b27", default="runs/20261125-004-lab27-multiclip")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--figure-out", default="outputs/figures/lab29_cover.png")
    ap.add_argument("--title", default="LAB29 full coverage (7 clips) vs M4 / LAB27")
    args = ap.parse_args()

    run = Path(args.run).resolve()
    b28 = Path(args.b28).resolve()
    b27 = Path(args.b27).resolve()
    data = collect(run, b28, b27)

    st = {p: {w: stats([c["completion"] for c in data[p][w].values()]) for w in WINDOWS} for p in POLICIES}
    sv = {p: {w: stats([c["surv"] for c in data[p][w].values()]) for w in WINDOWS} for p in POLICIES}

    cell_delta = {}
    for w in WINDOWS:
        common = sorted(set(data["lab29"][w]) & set(data["lab27"][w]))
        ds = [data["lab29"][w][s]["completion"] - data["lab27"][w][s]["completion"] for s in common]
        cell_delta[w] = {
            "n": len(common),
            "mean": (sum(ds) / len(ds)) if ds else None,
            "up": sum(1 for d in ds if d > 0.005),
            "down": sum(1 for d in ds if d < -0.005),
            "flat": sum(1 for d in ds if abs(d) <= 0.005),
        }

    dmin = {}
    audit = b28 / "lab28_audit.json"
    if audit.exists():
        try:
            a = json.loads(audit.read_text())
            for row in a.get("rows", []):
                dmin["w%03d" % int(row["start"])] = float(row.get("dmin_mean", 0.0))
        except Exception:
            dmin = {}

    n_up = sum(1 for w in DANCE if (cell_delta[w]["mean"] or 0) >= 0.02)
    n_down = sum(1 for w in DANCE if (cell_delta[w]["mean"] or 0) <= -0.02)
    if n_up >= 5 and n_down == 0:
        verdict = "(A) 覆盖有效：六个窗口几乎一起被拉起来 → 覆盖率是主因（哪段练过就哪段会）"
    elif n_up == 0:
        verdict = "(B) 覆盖无效：训练窗口都没被拉起来 → 问题不在数据覆盖，得改方法"
    elif n_up <= 2:
        verdict = "(B') 局部提升：只有个别窗口变好 → 覆盖不是主因，仍要回到观测/参考/训练方式"
    else:
        verdict = "(C) 部分覆盖：一部分窗口好、一部分不动或变差 → 要逐窗口对照新颖度，别写「全覆盖解决记忆」"

    def wrap(p):
        return BT + str(p) + BT

    lines = []
    add = lines.append
    add("# LAB29 全段覆盖训练（dance1 六窗口 + fight1 S3 0–20 s，共 7 段）")
    add("")
    add("- run：" + wrap(run))
    add("- 基线：dance1 六窗口取自 " + wrap(b28) + "，fight1 三窗口取自 " + wrap(b27))
    add("- 说明：LAB29 = 在 LAB27 基础上再补 dance1 20–40/40–60/60–80/80–100/100–120 s 五段；训练其余全同 M4（3000 iter、4096 envs、seed 0、热启动 model_49999）。")
    add("")
    add("## 1. 完成度（track_completion，均值 ± 标准差）")
    add("")
    add("| 窗口 | M4（1 段） | LAB27（2 段） | LAB29（7 段） | LAB29 − LAB27 |")
    add("|---|---|---|---|---|")
    for w in WINDOWS:
        a_, b_, c_ = st["m4"][w], st["lab27"][w], st["lab29"][w]
        d = cell_delta[w]
        if d["mean"] is None:
            delta_txt = "—"
        else:
            delta_txt = "%+.3f（%d↑/%d↓/%d=）" % (d["mean"], d["up"], d["down"], d["flat"])
        add("| %s | %s | %s | %s | %s |" % (
            WIN_CN[w],
            "—" if a_ is None else "%.3f ± %.3f (n=%d)" % (a_["mean"], a_["sd"], a_["n"]),
            "—" if b_ is None else "%.3f ± %.3f (n=%d)" % (b_["mean"], b_["sd"], b_["n"]),
            "—" if c_ is None else "%.3f ± %.3f (n=%d)" % (c_["mean"], c_["sd"], c_["n"]),
            delta_txt,
        ))
    add("")
    add("## 2. 存活帧（motion_frame_furthest；999 = 跑完）")
    add("")
    add("| 窗口 | M4 | LAB27 | LAB29 |")
    add("|---|---|---|---|")
    for w in WINDOWS:
        row = []
        for p in POLICIES:
            s = sv[p][w]
            row.append("—" if s is None else "%.0f（%.0f–%.0f）" % (s["mean"], s["min"], s["max"]))
        add("| %s | %s |" % (WIN_CN[w], " | ".join(row)))
    add("")
    add("## 3. 判读")
    add("")
    add("- **判据（写预测时定死）**：dance1 六窗口里 LAB29 − LAB27 ≥ +0.02 的窗口数 ≥5 → (A)；≤2 → (B')；其余 → (C)。")
    add("- **实测**：≥+0.02 的窗口 %d 个，≤−0.02 的窗口 %d 个 → **%s**" % (n_up, n_down, verdict))
    dance_txt = []
    for w in DANCE:
        d = cell_delta[w]["mean"]
        dance_txt.append("%s %s" % (w, "—" if d is None else "%+.3f" % d))
    add("- dance1 六窗口 Δ（LAB29 − LAB27）：" + "，".join(dance_txt))
    if dmin:
        xs = [dmin.get(w, 0.0) for w in DANCE]
        ys = [cell_delta[w]["mean"] for w in DANCE]
        if all(y is not None for y in ys):
            rho = spearman(xs, ys)
            rho_txt = "—" if rho is None else "%.3f" % rho
            add("- 新颖度 dmin（LAB28 口径）vs 本轮 Δ 的 Spearman：**" + rho_txt + "**（n=6，探索性）")
            add("- dmin：" + "，".join("%s %.2f" % (w, dmin.get(w, 0.0)) for w in DANCE))
    add("")
    add("## 4. 怎么读这张表（顺序）")
    add("")
    add("1. 先看 LAB29 在 **train** 窗口有没有掉（掉 >0.1 = 训练被 7 段摊薄，后面所有对比都要打折）。")
    add("2. 再看 dance1 六窗口的 Δ：**一起变好** 还是 **只有个别窗口** —— 这直接决定写 (A) / (B') / (C)。")
    add("3. 再看 heldout / unseen：它们和 dance1 不是同一动作，**不该因为多练 dance1 而变好**；明显变差说明容量被占。")
    add("4. 最后才看存活帧与 MPKPE —— 完成度单独用会被「窗口自身难度」污染（LAB28 已证）。")
    add("")
    add("## 5. 自检")
    add("")
    add("- 本表 LAB29 的格子数应为 9 窗口 × 10 种子 = 90；" + wrap("lab29_cover.sh eval") + " 会按 summary.json 跳过已完成的。")
    add("- M4 / LAB27 的历史格子数：dance1 六窗口各 10，fight1 三窗口各 10（缺失会显示 n 更小）。")
    add("- 若 LAB29 的 w060 与 LAB28 的 LAB27 值（0.108）差 >0.05，先查是不是评估到了别的 ckpt（summary.json 的 checkpoint 字段）。")

    text = "\n".join(lines) + "\n"
    (run / "lab29_cover.md").write_text(text)

    payload = {
        "run": str(run),
        "verdict": verdict,
        "n_up": n_up,
        "n_down": n_down,
        "completion": {p: {w: st[p][w] for w in WINDOWS} for p in POLICIES},
        "survival": {p: {w: sv[p][w] for w in WINDOWS} for p in POLICIES},
        "delta_vs_lab27": cell_delta,
        "dmin": dmin,
    }
    (run / "lab29_cover.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print("[lab29] 写出 %s/lab29_cover.md|json" % run)
    print("[lab29] 自检：lab29 格子 %d 个（应 90）" % sum(len(data["lab29"][w]) for w in WINDOWS))
    print("[lab29] 判读：" + verdict)

    if args.figure:
        plot(st, cell_delta, args.figure_out, args.title)
    return 0


def plot(st, cell_delta, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    xs = list(range(len(DANCE)))
    ax = axes[0]
    for p, color in (("m4", "#7f7f7f"), ("lab27", "#1f77b4"), ("lab29", "#d62728")):
        ys = [(st[p][w]["mean"] if st[p][w] else float("nan")) for w in DANCE]
        ax.plot(xs, ys, "o-", color=color, label=POLICY_EN[p], linewidth=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([w.replace("w", "") + "s" for w in DANCE])
    ax.set_xlabel("dance1 window start (s)")
    ax.set_ylabel("track completion")
    ax.set_title("completion on the six dance1 windows")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    ds = [(cell_delta[w]["mean"] if cell_delta[w]["mean"] is not None else 0.0) for w in DANCE]
    colors = ["#2ca02c" if d > 0 else "#d62728" for d in ds]
    ax.bar(xs, ds, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(0.02, color="#2ca02c", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(-0.02, color="#d62728", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([w.replace("w", "") + "s" for w in DANCE])
    ax.set_xlabel("dance1 window start (s)")
    ax.set_ylabel("completion delta (LAB29 - LAB27)")
    ax.set_title("coverage gain per window (dashed = +-0.02)")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(title)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print("[lab29] 图：%s" % out)


if __name__ == "__main__":
    raise SystemExit(main())
