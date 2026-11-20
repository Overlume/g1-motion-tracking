#!/usr/bin/env python
"""LAB16 判定翻转率：同一批种子重复 N 遍，数"20 个种子里有几个分类会翻"。

输入：每遍一个目录，里面是 scripts/lab12_recovery_report.py 写的 recovery.json。
      第 1 遍通常是 LAB15 的 20 种子目录（--rep rep1=runs/...）。
输出：<dir>/fliprate.md、<dir>/fliprate.json，以及可选的逐种子 x 逐遍矩阵图。

两个翻转口径：
- 二进制翻转（主指标）：down / 非 down 之间变 -> k（倒地数）会变，直接影响 k/N。
- 三分类翻转：stable / transient / down 之间变 -> k 可能不变，但"崩了又恢复"的叙事会变。

用法：
    environment/.venv/bin/python scripts/lab16_report.py \
        --dir runs/2026xxxx-NNN-lab16-fliprate \
        --rep rep1=runs/20261120-001-lab15-boundary20 \
        --rep rep2=runs/2026xxxx-NNN-lab16-fliprate/rep2 \
        --rep rep3=runs/2026xxxx-NNN-lab16-fliprate/rep3 \
        --figure outputs/figures/lab16_flipmatrix.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.stats import beta

SHORT = {"stable": "S", "transient": "T", "down": "D"}
CLASSES = ["stable", "transient", "down"]
# 5 种子基线（LAB12），用于"101-105 子集是否复现"
LAB12_BASE = {"nx_j60": 1, "ny_j60": 4, "px_j120": 0, "nx_j120": 5, "py_j120": 0, "ny_j120": 5}
LAB12_DIR = "runs/20261119-001-lab12-recovery"


def cp_interval(k: int, n: int, alpha: float = 0.05):
    if n <= 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def parse_rep(spec: str):
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), Path(path)
    p = Path(spec)
    return p.name, p


def load_rep(path: Path):
    f = path / "recovery.json"
    if not f.exists():
        raise SystemExit("找不到 %s（先跑脚本 run，或先单独判分）" % f)
    table = {}
    for r in json.loads(f.read_text(encoding="utf-8")):
        table.setdefault(r["cell"], {})[int(r["seed"])] = r
    return table


def median(v):
    v = sorted(x for x in v if x == x)
    if not v:
        return float("nan")
    m = len(v) // 2
    return float(v[m]) if len(v) % 2 else (v[m - 1] + v[m]) / 2.0


def main() -> int:
    ap = argparse.ArgumentParser(description="LAB16 判定翻转率")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--rep", action="append", required=True,
                    help="label=path，按顺序；第 1 个是参照遍（通常是 LAB15）")
    ap.add_argument("--cells", default="nx_j60,ny_j60")
    ap.add_argument("--figure", default=None)
    args = ap.parse_args()

    run = Path(args.dir)
    reps = [parse_rep(s) for s in args.rep]
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    need_seeds = [101 + i for i in range(20)]

    tables = {}
    for label, path in reps:
        tables[label] = load_rep(path)

    # ckpt 一致性（不同 ckpt 混着比 = 无效实验）
    ckpts = {}
    for label, path in reps:
        pf = path / "params.txt"
        if pf.exists():
            for line in pf.read_text(encoding="utf-8").splitlines():
                if line.startswith("ckpt:"):
                    ckpts[label] = line.split(":", 1)[1].strip()

    # 逐格、逐种子
    per_cell = {}
    missing = []
    for cell in cells:
        seeds = sorted(set().union(*[set(tables[l].get(cell, {})) for l, _ in reps])) or need_seeds
        rows = {}
        for label, _ in reps:
            got = tables[label].get(cell, {})
            if not got:
                missing.append("%s/%s" % (label, cell))
            for s in seeds:
                if s in got:
                    rows.setdefault(s, {})[label] = got[s]["cls"]
        k_by_rep = {}
        for label, _ in reps:
            got = tables[label].get(cell, {})
            k_by_rep[label] = sum(1 for r in got.values() if r["cls"] == "down")
        complete = [s for s in seeds if len(rows[s]) == len(reps)]
        flip_bin = [s for s in complete
                    if len({rows[s][l] == "down" for l in rows[s]}) > 1]
        flip_3 = [s for s in complete if len({rows[s][l] for l in rows[s]}) > 1]
        per_cell[cell] = dict(
            seeds=seeds, complete=complete, rows=rows, k_by_rep=k_by_rep,
            flip_bin=flip_bin, flip_3=flip_3,
            k_values=[k_by_rep[l] for l, _ in reps],
        )

    total_seeds = sum(len(v["complete"]) for v in per_cell.values())
    total_bin = sum(len(v["flip_bin"]) for v in per_cell.values())
    total_3 = sum(len(v["flip_3"]) for v in per_cell.values())
    bin_lo, bin_hi = cp_interval(total_bin, total_seeds)

    L = []
    a = L.append
    a("# LAB16 边界格判定翻转率")
    a("")
    a("- 输入：%s" % run)
    a("- 参照遍：" + "；".join("%s = %s" % (l, p) for l, p in reps))
    a("- 判分：与 LAB12 相同（scripts/lab12_recovery_report.py；stable / transient / down）")
    a("- 目的：LAB14 测的是「远离阈值」的重跑噪声（=0），LAB15 在边界格观测到 1 例翻转（D27）。"
      "这里量清楚：20 个种子里一般有几个分类会翻，从而决定边界格 k/N 该带多宽的不确定度。")
    if len(set(ckpts.values())) > 1:
        a("- **警告：各遍 ckpt 不一致**：%s（结论无效，先查 params.txt）" % ckpts)
    if missing:
        a("- 缺数据的格：%s" % ", ".join(missing))
    a("")
    a("## 每格 x 每遍")
    a("")
    a("| 格 | 遍 | n | stable | transient | down | 倒地种子 | 最长偏离中位 (s) |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for cell in cells:
        for label, path in reps:
            got = tables[label].get(cell, {})
            if not got:
                continue
            rs = list(got.values())
            c = {k: sum(1 for r in rs if r["cls"] == k) for k in CLASSES}
            ds = sorted(s for s, r in got.items() if r["cls"] == "down")
            dev = median([r["longest_dev_s"] for r in rs])
            a("| %s | %s | %d | %d | %d | %d | %s | %.2f |" % (
                cell, label, len(rs), c["stable"], c["transient"], c["down"],
                ", ".join(str(x) for x in ds) if ds else "-", dev))
    a("")
    a("## 逐种子判定（S=stable / T=transient / D=down）")
    a("")
    head = "| 格 | 种子 | " + " | ".join(l for l, _ in reps) + " | 二进制翻转 | 三分类翻转 |"
    a(head)
    a("| --- | --- | " + " | ".join("---" for _ in reps) + " | --- | --- |")
    for cell in cells:
        info = per_cell[cell]
        for s in info["seeds"]:
            r = info["rows"].get(s, {})
            cv = [SHORT.get(r[l], "?") if l in r else "-" for l, _ in reps]
            bin_mark = "是" if s in info["flip_bin"] else ""
            tri_mark = "是" if s in info["flip_3"] else ""
            a("| %s | %d | %s | %s | %s |" % (cell, s, " | ".join(cv), bin_mark, tri_mark))
    a("")
    a("## 翻转统计（主指标）")
    a("")
    a("| 格 | 完整种子数 | 二进制翻转（down 变非 down 或反之） | 翻转率 95% CI | 三分类翻转 | 逐遍 k | k 极差 |")
    a("| --- | --- | --- | --- | --- | --- | --- |")
    for cell in cells:
        info = per_cell[cell]
        n = len(info["complete"])
        b = len(info["flip_bin"])
        lo, hi = cp_interval(b, n)
        a("| %s | %d | %d | %.0f%% [%.0f%%, %.0f%%] | %d | %s | %d |" % (
            cell, n, b, 100 * b / n if n else 0.0, 100 * lo, 100 * hi,
            len(info["flip_3"]), " / ".join(str(x) for x in info["k_values"]),
            max(info["k_values"]) - min(info["k_values"])))
    a("| **合计** | %d | %d | %.0f%% [%.0f%%, %.0f%%] | %d | — | — |" % (
        total_seeds, total_bin, 100 * total_bin / total_seeds if total_seeds else 0.0,
        100 * bin_lo, 100 * bin_hi, total_3))
    a("")
    a("## 翻转明细")
    a("")
    any_flip = False
    for cell in cells:
        info = per_cell[cell]
        for s in info["flip_bin"]:
            any_flip = True
            r = info["rows"][s]
            seq = " -> ".join("%s=%s" % (l, r[l]) for l, _ in reps if l in r)
            a("- **%s seed %d**：%s" % (cell, s, seq))
    if not any_flip:
        a("- 无二进制翻转：所有完整种子的 down / 非 down 判定在各遍之间一致。")
    same_k_only = [cell for cell in cells
                   if per_cell[cell]["flip_3"] and not per_cell[cell]["flip_bin"]]
    for cell in same_k_only:
        a("- %s：只有 stable <-> transient 的换位，k（倒地数）不受影响。" % cell)
    a("")
    a("## 101-105 子集复核（与 LAB12 5 种子对齐）")
    a("")
    a("| 格 | 遍 | 101-105 倒地数 | LAB12 基线 | 一致 |")
    a("| --- | --- | --- | --- | --- |")
    for cell in cells:
        base = LAB12_BASE.get(cell)
        for label, _ in reps:
            got = tables[label].get(cell, {})
            sub = [got[s]["cls"] for s in range(101, 106) if s in got]
            if len(sub) < 5:
                continue
            k5 = sum(1 for c in sub if c == "down")
            ok = "—" if base is None else ("是" if k5 == base else "**否**")
            a("| %s | %s | %d/5 | %s | %s |" % (
                cell, label, k5, "-" if base is None else "%d/5" % base, ok))
    a("")
    a("对照：LAB12 的 5 种子是另一个独立 run（%s）；参照遍与它的差异就是 D27 那类翻转，" % LAB12_DIR)
    a("不是环境问题——查环境请比 ckpt sha 与 params.txt。")
    a("")
    a("## 判定（按 docs/learning/LAB16-fliprate.md §0 的三档规则）")
    a("")
    a("规则：每格二进制翻转 ≤1/20 -> k/N 可直接用（注明 ±1 个种子的不可复现成分）；"
      "=2/20 -> 边界格只能写「每次运行 k ± 2」，k/N 当中心估计；≥3/20 -> 改为区间叙事，每格 ≥3 遍取中位。")
    a("")
    for cell in cells:
        info = per_cell[cell]
        b = len(info["flip_bin"])
        if b <= 1:
            v = "≤1/20：k/N 可直接用，加一句「±1 个种子的判定不可复现」"
        elif b == 2:
            v = "2/20（10%）：k/N 只能当中心估计，边界格必须写「每次重跑 k 会差 ±2」"
        else:
            v = "≥3/20：翻转不可忽略，边界格改「区间叙事」（逐遍 k 的极差 + 中位），且每格 ≥3 遍"
        a("- **%s**：二进制翻转 %d/%d（三分类 %d/%d）；逐遍 k = %s -> %s" % (
            cell, b, len(info["complete"]), len(info["flip_3"]), len(info["complete"]),
            " / ".join(str(x) for x in info["k_values"]), v))
    a("")
    a("## 口径与限制")
    a("")
    a("- N 遍只采样 N 次：观测到的翻转率是「至少翻一次」的下界估计，不是每遍独立的翻转概率。")
    a("- 翻转的物理来源是数值抖动（LAB14：骨盆最低 ±1.1 cm、最长偏离 ±0.34 s）跨过硬阈值"
      "（骨盆低 0.08 m 或倾角差 13 deg、连续 10 帧）；离阈值越近的种子越容易翻。")
    a("- 仍是单一 ckpt（model_52998）、单一时段（8.0 s）、单一波形（0.2 s 方波）、单一作用点（torso_link）。")
    a("- 翻转率不等于「实验做错了」：它说明这套判据在边界处只能给区间，不能给单点。")
    a("")
    (run / "fliprate.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    out = dict(
        dir=str(run),
        reps=[dict(label=l, path=str(p), ckpt=ckpts.get(l)) for l, p in reps],
        cells={cell: dict(
            seeds=per_cell[cell]["complete"],
            k_by_rep=per_cell[cell]["k_by_rep"],
            k_range=max(per_cell[cell]["k_values"]) - min(per_cell[cell]["k_values"]),
            flip_binary=per_cell[cell]["flip_bin"],
            flip_three_way=per_cell[cell]["flip_3"],
            classes={str(s): per_cell[cell]["rows"][s] for s in per_cell[cell]["seeds"]},
        ) for cell in cells},
        pooled=dict(seeds=total_seeds, flip_binary=total_bin, flip_three_way=total_3,
                    ci=[bin_lo, bin_hi]),
        warning=("ckpt 不一致" if len(set(ckpts.values())) > 1 else None),
    )
    (run / "fliprate.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")

    if args.figure:
        plot(run, reps, tables, cells, per_cell, args.figure)

    print("\n".join(L[:26]))
    print("")
    for cell in cells:
        info = per_cell[cell]
        print("%s: k=%s  二进制翻转 %d/%d  三分类翻转 %d/%d 种子 %s" % (
            cell, " / ".join(str(x) for x in info["k_values"]),
            len(info["flip_bin"]), len(info["complete"]),
            len(info["flip_3"]), len(info["complete"]),
            info["flip_bin"] or "-"))
    print("写盘：%s、%s" % (run / "fliprate.md", run / "fliprate.json"))
    return 0


def plot(run, reps, tables, cells, per_cell, figure):
    cmap = ListedColormap(["#d9d9d9", "#ffb347", "#d62728"])
    ncell = len(cells)
    fig, axes = plt.subplots(ncell, 1, figsize=(9.0, 1.5 + 1.5 * ncell), dpi=160, squeeze=False)
    for idx, cell in enumerate(cells):
        ax = axes[idx][0]
        seeds = per_cell[cell]["seeds"]
        M = np.full((len(reps), len(seeds)), np.nan)
        for j, (label, _) in enumerate(reps):
            got = tables[label].get(cell, {})
            for k_, s in enumerate(seeds):
                if s in got:
                    M[j, k_] = CLASSES.index(got[s]["cls"])
        ax.imshow(M, aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5, interpolation="nearest")
        ax.set_yticks(range(len(reps)))
        ax.set_yticklabels([l for l, _ in reps], fontsize=8)
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([str(s) for s in seeds], fontsize=7, rotation=90)
        ks = "/".join(str(x) for x in per_cell[cell]["k_values"])
        ax.set_title("%s  down per rep = %s   (binary flips: %d/%d)" % (
            cell, ks, len(per_cell[cell]["flip_bin"]), len(per_cell[cell]["complete"])),
            fontsize=9)
        for k_ in range(len(seeds) + 1):
            ax.axvline(k_ - 0.5, color="white", lw=0.4)
        for j in range(len(reps) + 1):
            ax.axhline(j - 0.5, color="white", lw=0.6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#d9d9d9", "#ffb347", "#d62728"]]
    fig.legend(handles, ["stable", "transient", "down"], loc="lower center",
               ncol=3, fontsize=8, frameon=False)
    fig.suptitle("LAB16: per-seed verdict across repeated runs (same seeds, same config)", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    Path(figure).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure)
    print("图：%s" % figure)


if __name__ == "__main__":
    raise SystemExit(main())
