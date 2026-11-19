#!/usr/bin/env python
"""LAB14 重复实验比对：同一配置重跑 N 遍，看倒地数与逐种子判定是否稳定。

读 <dir>/rep*/recovery.json（由 scripts/lab12_recovery_report.py 生成），写：
- <dir>/replication.md   每格 x 每遍的计数、逐种子矩阵、噪声判定、与 LAB13 基线对照
- <dir>/replication.json 同一内容的机器可读版

判读规则（与 docs/learning/LAB14-replication.md 一致）：
- 倒地数极差 0 且种子集合不变 -> 各遍完全一致（60/80 两端可重复，进 LAB15）
- 极差 0 但种子互换 -> 倒地数一致、边界摆动
- 极差 1 -> 差 1 个种子（20%），正好在 5 种子分辨率上，不能判定
- 极差 >= 2 -> 重跑噪声主导，改概率叙事并补种子到 >= 20/格
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

BASELINE = {
    "ny_j60": dict(down=4, seeds=[101, 103, 104, 105],
                   source="runs/20261119-001-lab12-recovery",
                   label="300 N x 0.2 s = 60 N*s"),
    "ny_j40": dict(down=0, seeds=[],
                   source="runs/20261119-002-lab13-mid-impulse",
                   label="200 N x 0.2 s = 40 N*s (legacy)"),
    "ny_j80": dict(down=5, seeds=[101, 102, 103, 104, 105],
                   source="runs/20261119-001-lab13b-true-mid",
                   label="400 N x 0.2 s = 80 N*s"),
}
SHORT = {"stable": "S", "transient": "T", "down": "D"}


def rep_no(p: Path) -> int:
    s = p.name[3:]
    return int(s) if s.isdigit() else 0


def verdict(down_counts, flips):
    spread = max(down_counts) - min(down_counts)
    if spread == 0 and not flips:
        return "各遍完全一致（倒地数与倒地种子集合都不变）", spread
    if spread == 0:
        return "倒地数一致，但个别种子在 stable/transient/down 间互换（边界摆动）", spread
    if spread == 1:
        return "差 1 个种子（20%）——正好在 5 种子分辨率上，不能判定真实或噪声", spread
    return "摆动 >= 2 个种子——重跑噪声主导", spread


def main() -> int:
    ap = argparse.ArgumentParser(description="LAB14 重复实验比对")
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    run = Path(args.dir)
    reps = sorted([d for d in run.iterdir()
                   if d.is_dir() and d.name.startswith("rep") and d.name[3:].isdigit()],
                  key=rep_no)
    if not reps:
        raise SystemExit(f"没有 rep* 目录: {run}")
    rep_names = [d.name for d in reps]

    table = {}
    missing = []
    for rep in reps:
        f = rep / "recovery.json"
        if not f.exists():
            missing.append(rep.name)
            continue
        for r in json.loads(f.read_text(encoding="utf-8")):
            table.setdefault(r["cell"], {}).setdefault(rep.name, {})[int(r["seed"])] = r
    if not table:
        raise SystemExit("没有可汇总的 recovery.json（先跑 lab14_replicate.sh run）")

    cells = sorted(table)
    out_cells = {}
    for cell in cells:
        per_rep = table[cell]
        present = [rn for rn in rep_names if rn in per_rep]
        down_counts = [sum(1 for r in per_rep[rn].values() if r["cls"] == "down") for rn in present]
        all_seeds = sorted(set().union(*[set(per_rep[rn]) for rn in present]))
        flips = [s for s in all_seeds
                 if len({per_rep[rn][s]["cls"] for rn in present if s in per_rep[rn]}) > 1]
        v, spread = verdict(down_counts, flips)
        out_cells[cell] = dict(
            reps=present,
            down_counts=down_counts,
            down_by_rep={rn: sorted(s for s, r in per_rep[rn].items() if r["cls"] == "down")
                         for rn in present},
            down_seeds=sorted(set(s for rn in present for s, r in per_rep[rn].items() if r["cls"] == "down")),
            flips=flips,
            spread=spread,
            verdict=v,
        )

    L = []
    add = L.append
    add("# LAB14 重复实验比对")
    add("")
    add(f"- 输入：{run}（{len(rep_names)} 遍：{', '.join(rep_names)}）")
    add("- 判分：与 LAB12 相同（scripts/lab12_recovery_report.py；stable / transient / down）")
    add("- 目的：同种子、同配置、重新起进程重跑，看结局是否稳定（测的是重跑噪声，不是种子分布）")
    if missing:
        add(f"- 缺 recovery.json 的遍：{', '.join(missing)}")
    add("")
    add("## 每格 x 每遍")
    add("")
    add("| 格 | 冲量 | 遍 | n | stable | transient | down | 倒地种子 | 最长偏离中位 (s) |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for cell in cells:
        for rn in out_cells[cell]["reps"]:
            rs = list(table[cell][rn].values())
            c = {k: sum(1 for r in rs if r["cls"] == k) for k in SHORT}
            ds = out_cells[cell]["down_by_rep"][rn]
            dev = statistics.median([r["longest_dev_s"] for r in rs]) if rs else float("nan")
            label = BASELINE.get(cell, {}).get("label", "?")
            add(f"| {cell} | {label} | {rn} | {len(rs)} | {c['stable']} | {c['transient']} | {c['down']} | "
                f"{', '.join(str(x) for x in ds) if ds else '-'} | {dev:.2f} |")
    add("")
    add("## 逐种子判定（S=stable / T=transient / D=down）")
    add("")
    present0 = out_cells[cells[0]]["reps"]
    add("| 格 | 种子 | " + " | ".join(present0) + " |")
    add("| --- | --- | " + " | ".join("---" for _ in present0) + " |")
    for cell in cells:
        present = out_cells[cell]["reps"]
        seeds = sorted(set().union(*[set(table[cell][rn]) for rn in present]))
        for s in seeds:
            cells_v = [SHORT.get(table[cell][rn][s]["cls"], "?") if s in table[cell][rn] else "-"
                       for rn in present]
            add(f"| {cell} | {s} | " + " | ".join(cells_v) + " |")
    add("")
    add("## 噪声判定")
    add("")
    for cell in cells:
        info = out_cells[cell]
        counts = " / ".join(str(x) for x in info["down_counts"])
        flip_txt = ", ".join(str(x) for x in info["flips"]) if info["flips"] else "无"
        add(f"- **{cell}**：倒地数 {counts}（极差 {info['spread']}）；种子互换 {flip_txt} -> {info['verdict']}")
    add("")
    add("## 与 LAB13 基线对照")
    add("")
    add("| 格 | 基线（倒地/5） | 本次三遍（倒地/5） | 极差 | 来源 |")
    add("| --- | --- | --- | --- | --- |")
    for cell in cells:
        b = BASELINE.get(cell, {})
        counts = " / ".join(str(x) for x in out_cells[cell]["down_counts"])
        add(f"| {cell} | {b.get('down', '?')} | {counts} | {out_cells[cell]['spread']} | {b.get('source', '?')} |")
    add("")
    add("## 口径与限制")
    add("")
    add("- 5 个种子 = 20% 分辨率；三遍只测重跑噪声，不测种子分布。")
    add("- 单一施加时刻（8.0 s）、单一波形（0.2 s 方波）、单一作用点（torso_link）、固定 ckpt（model_52998）。")
    add("- 三遍一致只说明'同配置可复现'；要谈冲量效应仍需 LAB15（更多种子或相位扫描）。")
    add("- 若某遍缺数据，表中 n 会小于 5；不要用缺数据的遍下结论。")
    add("")
    (run / "replication.md").write_text(chr(10).join(L) + chr(10), encoding="utf-8")
    (run / "replication.json").write_text(
        json.dumps(dict(dir=str(run), reps=rep_names, missing=missing, cells=out_cells),
                   ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")

    print(f"写盘：{run}/replication.md / replication.json")
    for cell in cells:
        info = out_cells[cell]
        print(f"{cell}: 倒地数 {' / '.join(str(x) for x in info['down_counts'])} -> {info['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
