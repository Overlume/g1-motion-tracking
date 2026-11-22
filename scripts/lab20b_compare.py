#!/usr/bin/env python
"""LAB20b 对比：LAB20（DR 加强）ckpt 与 M4 在 60 N·s 两向的倒地数直接对比。

输入：
- <run>/rep<N>/recovery.json   （scripts/lab12_recovery_report.py 的判分结果）
- 基线 recovery.json           （默认 runs/20261120-001-lab15-boundary20，M4 的 20 种子单遍）

输出：<run>/vs_m4.md，含
- 每格逐遍倒地 k/n 与 Fisher 精确检验 p 值（每遍 vs M4，只用在两边都有的共同种子上）
- 逐种子的配对变化：M4 倒而 LAB20b 站住（改进）、M4 站而 LAB20b 倒（变差）

注意：Fisher 是「两组 × 两种结局」的边际比较，不利用种子配对；配对信息放在种子清单里看。
缺失的格/种子显示为 -，不会伪装成 0。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import fisher_exact

M4_SHA = "fc8f975b7f6cbb223bccaa27a4edd63901a91518e69d2df9e94710d0fe8ceed9"


def load(path: Path):
    f = path / "recovery.json"
    if not f.exists():
        raise SystemExit("找不到 %s（先判分）" % f)
    cells = {}
    for r in json.loads(f.read_text(encoding="utf-8")):
        cells.setdefault(r["cell"], {})[int(r["seed"])] = r["cls"]
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description="LAB20b vs M4 扰动对比")
    ap.add_argument("--run", required=True)
    ap.add_argument("--baseline", default="runs/20261120-001-lab15-boundary20")
    ap.add_argument("--cells", default="nx_j60,ny_j60")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    base_dir = Path(args.baseline)
    base = load(base_dir)
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]

    reps = sorted((d for d in run.iterdir() if d.is_dir() and d.name.startswith("rep")),
                  key=lambda d: int(d.name[3:]))
    if not reps:
        raise SystemExit("%s 下没有 rep<N> 目录" % run)
    tabs = {d.name: load(d) for d in reps}

    sha = ""
    pf = run / "params.txt"
    if pf.exists():
        for line in pf.read_text(encoding="utf-8").splitlines():
            if line.startswith("sha256:"):
                sha = line.split(":", 1)[1].strip()

    L = []
    a = L.append
    a("# LAB20b：DR 加强 ckpt 的 60 N·s 抗推对比（vs M4）")
    a("")
    a("- LAB20b：%s（ckpt sha %s）" % (run, sha or "未知"))
    a("- 基线：%s（M4，LAB15 第 1 遍，种子 101-120）" % base_dir)
    a("- 判分：与 LAB15/LAB16 相同（scripts/lab12_recovery_report.py；stable / transient / down）")
    if sha and sha == M4_SHA:
        a("- **警告：sha 与 M4 相同——这不是 LAB20 的权重，对比无意义（查 params.txt）**")
    a("")
    a("## 每格倒地数（分母 = 该遍与基线的共同种子数）")
    a("")
    a("| 格 | M4 | " + " | ".join(d.name for d in reps) + " | Fisher p（各遍 vs M4） |")
    a("| --- | --- | " + " | ".join("---" for _ in reps) + " | --- |")
    per_cell = {}
    for cell in cells:
        b = base.get(cell, {})
        per_rep = []
        for d in reps:
            got = tabs[d.name].get(cell, {})
            seeds = sorted(set(b) & set(got))
            if not seeds:
                per_rep.append(None)
                continue
            k = sum(1 for s in seeds if got[s] == "down")
            km = sum(1 for s in seeds if b[s] == "down")
            p = fisher_exact([[k, len(seeds) - k], [km, len(seeds) - km]])[1]
            per_rep.append(dict(k=k, n=len(seeds), kb=km, p=p, improved=[s for s in seeds if b[s] == "down" and got[s] != "down"],
                                worsened=[s for s in seeds if b[s] != "down" and got[s] == "down"]))
        if not b:
            a("| %s | - | %s | 基线缺此格 |" % (cell, " | ".join("-" for _ in reps)))
            per_cell[cell] = per_rep
            continue
        nb = len(b)
        kb = sum(1 for c in b.values() if c == "down")
        cells_txt = ["%d/%d" % (r["k"], r["n"]) if r else "-" for r in per_rep]
        ps_txt = ["%.3g" % r["p"] if r else "-" for r in per_rep]
        a("| %s | %d/%d | %s | %s |" % (cell, kb, nb, " | ".join(cells_txt), " / ".join(ps_txt)))
        per_cell[cell] = per_rep

    a("")
    a("## 逐种子配对变化（同一 batch 种子，只换了权重）")
    a("")
    for cell in cells:
        for d, r in zip(reps, per_cell.get(cell, [])):
            if not r:
                a("- **%s %s**：该遍没有这一格的数据" % (cell, d.name))
                continue
            a("- **%s %s**（%d 个共同种子）：M4 倒→站住 %s；M4 站→倒 %s" % (
                cell, d.name, r["n"],
                ", ".join(str(s) for s in r["improved"]) if r["improved"] else "无",
                ", ".join(str(s) for s in r["worsened"]) if r["worsened"] else "无"))

    a("")
    a("## 池化（所有格合计，分母随之调整）")
    a("")
    allb = sum(len(base.get(c, {})) for c in cells)
    allkb = sum(1 for c in cells for v in base.get(c, {}).values() if v == "down")
    a("- M4：%d/%d" % (allkb, allb))
    for d, i in zip(reps, range(len(reps))):
        got = [per_cell.get(c, [None] * len(reps))[i] for c in cells]
        n = sum(r["n"] for r in got if r)
        k = sum(r["k"] for r in got if r)
        a("- %s：%d/%d" % (d.name, k, n))
    ks = [sum(r["k"] for r in [per_cell.get(c, [None] * len(reps))[i] for c in cells] if r) for i in range(len(reps))]
    if len(ks) > 1:
        a("- 逐遍极差：%d" % (max(ks) - min(ks)))
    a("")
    a("## 怎么读（不替你下结论）")
    a("")
    a("- 若某向 k 明显低于 M4 且 Fisher p 小（<0.05），说明加强 DR 在这一向确实更抗推；")
    a("- 若 k 落在 M4 的三遍区间内，就写「没有可分辨差异」，不要说「略有改善」；")
    a("- 若某向变差，先查 params.txt 的 sha 与 seeds 是否与基线一致，再看翻转（fliprate.md）；")
    a("- 逐遍极差 >= 3 时按 D28/D30 写区间，不写单点。")
    a("")
    out = Path(args.out) if args.out else run / "vs_m4.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("写盘：%s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
