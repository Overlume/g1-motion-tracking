#!/usr/bin/env python
"""A1 判定口径审计：把各 LAB 的 recovery.json 统一成「退让式 transient」指标。

背景（D35）：LAB20b 发现 -x 60 N·s 的 20/20 全是 transient，但骨盆从未触地
（0.447-0.466 m）、末尾 root 误差中位 92.9 cm —— 这是「顺力退让」而不是「抗推」。
本脚本把同一套派生指标算到已有跑过的所有格上，用来区分：
  - 擦边 transient：偏离刚过 10 帧（约 0.2 s）、root 误差接近基线 -> 判定阈值噪声
  - 退让 transient：偏离 >= 0.5 s 且末尾 root 误差 >= 30 cm、骨盆未触地 -> 跟踪丢失
  - down：真的倒地或 3 s 内没恢复

用法：
  environment/.venv/bin/python scripts/lab_transient_audit.py --out runs/20261122-002-transient-audit
只读 runs/*/rep*/recovery.json，不跑仿真。
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

# (标签, run 目录, 格子列表, 权重说明)
REGISTRY = [
    ("LAB16 (M4)", "runs/20261120-002-lab16-fliprate", ["nx_j60", "ny_j60"]),
    ("LAB17 (M4)", "runs/20261120-003-lab17-y-transition", ["ny_j45", "ny_j50"]),
    ("LAB19 (M4)", "runs/20261121-001-lab19-symmetry", ["px_j60", "px_j80", "py_j60", "py_j80"]),
    ("LAB19b (M4)", "runs/20261121-002-lab19b-px100", ["px_j100"]),
    ("LAB20b (DR)", "runs/20261122-001-lab20b-perturb", ["nx_j60", "ny_j60"]),
]

YIELD_DEV_S = 0.5
YIELD_ROOT_CM = 30.0
EDGE_DEV_S = 0.3


def load(root: Path, run: str):
    rows = []
    for f in sorted((root / run).glob("rep*/recovery.json")):
        rows += json.loads(f.read_text(encoding="utf-8"))
    return rows


def med(xs):
    return round(st.median(xs), 2) if xs else None


def audit(rows, cell: str):
    rs = [r for r in rows if r["cell"] == cell]
    if not rs:
        return None
    down = [r for r in rs if r["cls"] == "down"]
    tra = [r for r in rs if r["cls"] == "transient"]
    sta = [r for r in rs if r["cls"] == "stable"]
    ys = [r for r in tra if r["longest_dev_s"] >= YIELD_DEV_S and r["root_err_end_cm"] >= YIELD_ROOT_CM]
    edge = [r for r in tra if r["longest_dev_s"] <= EDGE_DEV_S]
    return dict(
        cell=cell,
        n=len(rs),
        k_down=len(down),
        n_transient=len(tra),
        n_stable=len(sta),
        pct_down=round(100.0 * len(down) / len(rs), 1),
        # transient 的派生指标
        t_dev_s=med([r["longest_dev_s"] for r in tra]),
        t_root_cm=med([r["root_err_end_cm"] for r in tra]),
        t_root_max_cm=round(max([r["root_err_end_cm"] for r in tra]), 1) if tra else None,
        t_pzmin=round(min([r["pelvis_z_min"] for r in tra]), 3) if tra else None,
        t_maxdrop=med([r["max_drop"] for r in tra]),
        n_yield=len(ys),
        n_edge=len(edge),
        pzmin_all=round(min(r["pelvis_z_min"] for r in rs), 3),
        root_all=med([r["root_err_end_cm"] for r in rs]),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="A1 判定口径审计（退让式 transient）")
    ap.add_argument("--out", required=True, help="输出目录（会在其下写 audit.md/json）")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.out)
    if not outdir.is_absolute():
        outdir = root / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    table = []
    for label, run, cells in REGISTRY:
        rows = load(root, run)
        for c in cells:
            r = audit(rows, c)
            if r is None:
                print("skip (no data):", run, c)
                continue
            table.append(dict(lab=label, run=run, **r))

    with open(outdir / "audit.json", "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=1)

    ph = [
        "# A1 判定口径审计：transient 的两种类型（退让式 vs 擦边）",
        "",
        "- 生成：scripts/lab_transient_audit.py（只读已有 recovery.json，不跑仿真）",
        "- 数据：LAB16/17/19/19b 都是 M4 权重，LAB20b 是 LAB20（DR 加强）权重",
        "- 判据沿用 lab12_recovery_report.py：偏离 >= 0.08 m 或倾角差 >= 13°，连续 >= 10 帧 = 崩溃；末尾 0.5 s 平均下移 < 0.05 m 且倾角差 < 8° = 恢复",
        "- 本表新增派生指标（都在 transient 子集上算）：",
        "  - %s：偏离 >= %.1f s 且末尾 root 误差 >= %.0f cm = 退让式（跟丢了但没倒）" % ("n_yield", YIELD_DEV_S, YIELD_ROOT_CM),
        "  - %s：偏离 <= %.1f s = 擦边（刚过 10 帧门槛，跟踪基本还在）" % ("n_edge", EDGE_DEV_S),
        "",
        "| 权重 | 格 | 次数 | down | transient | stable | transient 偏离中位 (s) | transient 末尾 root 中位 (cm) | transient 骨盆最低 (m) | 退让式 | 擦边 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table:
        ph.append("| %s | %s | %d | %d (%.1f%%) | %d | %d | %s | %s | %s | %d | %d |" % (
            r["lab"], r["cell"], r["n"], r["k_down"], r["pct_down"], r["n_transient"], r["n_stable"],
            r["t_dev_s"], r["t_root_cm"], r["t_pzmin"], r["n_yield"], r["n_edge"]))
    ph += [
        "",
        "读法：",
        "",
        "1. 先看 down 列：那是「真倒地或 3 s 内没恢复」的失败率。",
        "2. 再看 transient 的偏离中位与 root 中位：偏离只有 0.2-0.3 s、root 只剩几 cm 的，属于判定门槛擦边，本质是稳的；",
        "3. 偏离 >= 0.5 s 且 root 中位 >= 30 cm 的，属于「被打退、跟丢、但没倒」——失败率会被 k=0 掩盖。",
        "",
        "生成时间：见 audit.json 的 mtime；口径见 D35/D36。",
        "",
    ]
    (outdir / "audit.md").write_text("\n".join(ph), encoding="utf-8")
    print("wrote", outdir / "audit.md")
    for r in table:
        print("%-12s %-9s k=%2d/%2d  yield=%2d edge=%2d  dev=%s root=%s pz=%s" % (
            r["lab"], r["cell"], r["k_down"], r["n"], r["n_yield"], r["n_edge"],
            r["t_dev_s"], r["t_root_cm"], r["t_pzmin"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
