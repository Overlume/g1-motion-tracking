#!/usr/bin/env python
"""M5 姿态审计（零 GPU）：用已有的 metrics.csv 回答"终止的格子到底是倒了，还是站着被误差判罚"。

只读 rollout 日志，不跑仿真：
- 每个 rollout 取第一次 reset 之前的帧；
- 看复位前最后 0.4 s 的姿态相对参考动作的偏移（骨盆低于参考多少、比参考歪多少度）；
- 分成三类：survived（跑完动作）/ collapse（复位瞬间姿态已崩）/ upright_trip（复位了但姿态还站得住，属于误差判罚，结局未知）。

口径、阈值依据与局限见 docs/learning/LAB11-posture-audit.md。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---- 口径（改这里要同步改 LAB11 文档与 docs/DECISIONS.md） ----
WINDOW_START_S = 7.5      # 受推前 0.5 s 起算，避开正常的蹲伏动作
TAIL_S = 0.4              # 复位前最后 0.4 s 视为"复位瞬间"
COLLAPSE_DROP_M = 0.08    # 骨盆比参考低 >= 8 cm 判为姿态崩
COLLAPSE_TILT_DEG = 13.0  # 或比参考更歪 >= 13 deg
PUSH_S = 8.0

# 视频核对过的 4 例，脚本自动校验分类是否与录像一致
VIDEO_CASES = [
    ("py_j30", 101, "collapse"),
    ("py_j60", 101, "collapse"),
    ("py_j27", 103, "collapse"),
    ("py_j27", 101, "survived"),
]


def load_rollout(d: Path):
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    m = np.genfromtxt(d / "metrics.csv", delimiter=",", names=True)
    return s, m


def features(s, m):
    rs = list(s.get("reset_steps") or [])
    rts = list(s.get("reset_terms") or [])
    end = int(rs[0]) if rs else len(m["step"])
    t = np.asarray(m["time_s"][:end], dtype=float)
    pz = np.asarray(m["pelvis_z"][:end], dtype=float)
    rpz = np.asarray(m["ref_pelvis_z"][:end], dtype=float)
    ti = np.asarray(m["tilt_deg"][:end], dtype=float)
    rti = np.asarray(m["ref_tilt_deg"][:end], dtype=float)

    drop = rpz - pz
    tiltex = ti - rti
    tail = t >= (t[-1] - TAIL_S)
    win = t >= WINDOW_START_S

    first_terms = list(rts[0]) if rts else []
    is_survived = all(x == "motion_end" for x in first_terms) if first_terms else True

    end_drop = float(drop[tail].mean())
    end_tiltex = float(tiltex[tail].mean())

    if is_survived:
        lab = "survived"
    elif end_drop >= COLLAPSE_DROP_M or end_tiltex >= COLLAPSE_TILT_DEG:
        lab = "collapse"
    else:
        lab = "upright_trip"

    return dict(
        reset_terms=",".join(first_terms) or "-",
        t_reset=float(t[-1]),
        dt_after_push=float(t[-1] - PUSH_S),
        end_drop=end_drop,
        end_tiltex=end_tiltex,
        drop_max=float(drop[win].max()),
        tiltex_max=float(tiltex[win].max()),
        pelvis_z_min=float(pz.min()),
        ref_pelvis_z_min=float(rpz.min()),
        label=lab,
    )


def discover(runs_dir: Path, patterns):
    out = []
    for run in sorted(runs_dir.glob("*")):
        if not run.is_dir() or not any(p in run.name for p in patterns):
            continue
        for sp in sorted(run.rglob("summary.json")):
            d = sp.parent
            if (d / "metrics.csv").exists():
                out.append((run.name, str(d.relative_to(run)), d))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="M5 姿态审计（零 GPU，只读 metrics.csv）")
    ap.add_argument("--runs-dir", default="runs", help="runs 根目录（默认 runs）")
    ap.add_argument("--patterns", nargs="+", default=["m5-perturb", "m5-threshold-refine"],
                    help="run 目录名要包含的关键字")
    ap.add_argument("--out", default=None, help="输出目录（默认 runs/posture-audit）")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    items = discover(runs_dir, args.patterns)
    if not items:
        print("没找到可审计的 rollout（需要同时有 summary.json 与 metrics.csv）", file=sys.stderr)
        return 2

    rows = []
    for run_name, rel, d in items:
        try:
            s, m = load_rollout(d)
            f = features(s, m)
        except Exception as e:  # 保留失败案例，不中断整批
            print("跳过 %s/%s: %s: %s" % (run_name, rel, type(e).__name__, e), file=sys.stderr)
            continue
        cell, _, seed_s = rel.partition("/")
        try:
            seed = int(seed_s.replace("seed", ""))
        except ValueError:
            seed = int(s.get("seed", -1))
        rows.append(dict(run=run_name, cell=cell, seed=seed, **f))

    if not rows:
        print("所有 rollout 都解析失败", file=sys.stderr)
        return 2

    out = Path(args.out) if args.out else runs_dir / "posture-audit"
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "posture_audit.csv"
    cols = ["run", "cell", "seed", "label", "reset_terms", "t_reset", "dt_after_push",
            "end_drop", "end_tiltex", "drop_max", "tiltex_max", "pelvis_z_min", "ref_pelvis_z_min"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(
                ("%.4f" % r[c] if isinstance(r[c], float) else str(r[c])) for c in cols) + "\n")

    checks = []
    for cell, seed, expect in VIDEO_CASES:
        hit = [r for r in rows if r["cell"] == cell and r["seed"] == seed]
        got = hit[0]["label"] if hit else "missing"
        checks.append((cell, seed, expect, got, got == expect))
    check_ok = all(c[4] for c in checks)

    counts = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1

    cells = {}
    for r in rows:
        cells.setdefault(r["cell"], []).append(r)

    def med(v):
        v = [x for x in v if x == x]
        return float(np.median(v)) if v else float("nan")

    L = []
    add = L.append
    add("# M5 姿态审计（零 GPU）")
    add("")
    add("- 输入：`%s` 下 %d 个 run、%d 次 rollout（只读 `metrics.csv` + `summary.json`，未跑仿真）"
        % (runs_dir, len(set(r["run"] for r in rows)), len(rows)))
    add("- 判据：复位前最后 %.2f s 的均值——骨盆比参考低 >= %.2f m 或比参考歪 >= %.1f deg 判 `collapse`；"
        "复位了但两项都没到判 `upright_trip`；跑到动作结束判 `survived`"
        % (TAIL_S, COLLAPSE_DROP_M, COLLAPSE_TILT_DEG))
    add("")
    add("## 总计")
    add("")
    add("| 类别 | 次数 | 含义 |")
    add("| --- | --- | --- |")
    add("| `survived` | %d | 跑完动作，没有被终止 |" % counts.get("survived", 0))
    add("| `collapse` | %d | 复位瞬间姿态已崩（比参考低 >= 8 cm 或歪 >= 13 deg）|" % counts.get("collapse", 0))
    add("| `upright_trip` | %d | 提前复位但复位瞬间姿态还站得住：属于误差判罚，之后会不会倒未知 |" % counts.get("upright_trip", 0))
    add("")
    add("## 视频校验（4 例已知答案）")
    add("")
    add("| 案例 | 录像结论 | 本脚本 | 一致 |")
    add("| --- | --- | --- | --- |")
    for cell, seed, expect, got, ok in checks:
        add("| `%s` seed %d | %s | %s | %s |" % (cell, seed, expect, got, "一致" if ok else "不一致"))
    add("")
    add("## 分格")
    add("")
    add("| 格 | n | survived | collapse | upright_trip | 复位时刻中位 (s) | 末段骨盆低 (m) 中位 | 末段歪 (deg) 中位 |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for cell in sorted(cells):
        rs = cells[cell]
        c = dict((k, sum(1 for r in rs if r["label"] == k)) for k in ("survived", "collapse", "upright_trip"))
        add("| `%s` | %d | %d | %d | %d | %.2f | %+.3f | %+.1f |" % (
            cell, len(rs), c["survived"], c["collapse"], c["upright_trip"],
            med([r["t_reset"] for r in rs if r["label"] != "survived"]),
            med([r["end_drop"] for r in rs]), med([r["end_tiltex"] for r in rs])))
    add("")
    add("## 逐次明细")
    add("")
    add("| run | cell | seed | 类别 | 复位项 | t_reset (s) | 推后 (s) | 末段骨盆低 (m) | 末段歪 (deg) | 窗口最大低 (m) | 窗口最大歪 (deg) |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in sorted(rows, key=lambda x: (x["run"], x["cell"], x["seed"])):
        add("| %s | `%s` | %d | %s | %s | %.2f | %+.2f | %+.3f | %+.1f | %.3f | %.1f |" % (
            r["run"][:26], r["cell"], r["seed"], r["label"], r["reset_terms"],
            r["t_reset"], r["dt_after_push"], r["end_drop"], r["end_tiltex"],
            r["drop_max"], r["tiltex_max"]))
    add("")
    add("## 这个统计能说什么、不能说什么")
    add("")
    add("- 能说：在复位那一刻，姿态是否已经偏离参考动作（骨盆高度、躯干倾角）。这是从日志读到的、与倒地最接近的量。")
    add("- 能说：`upright_trip` 这一类说明终止不等于倒地——强推力下策略在 0.1-0.3 s 内就被误差阈值判罚，那时它还站着。")
    add("- 不能说：`upright_trip` 之后会不会倒（回合已结束，没有后续数据）。")
    add("- 不能说：跪坐 vs 深蹲——两者骨盆高度重叠，真正的区分需要膝/踝高度与接触点，日志里没有。")
    add("")
    add("## 局限")
    add("")
    add("1. 阈值（8 cm / 13 deg）是从录像核对的 4 例上定的，样本小；改阈值会移动 collapse 与 upright_trip 的边界。")
    add("2. 末段窗口只有 0.4 s，抖动大；结论用中位与逐次明细并列，不单点下结论。")
    add("3. 只覆盖 +x/-x/+y/-y 的 12 格粗扫与 +y 的 5 档细化，不含无扰动基线对照。")
    add("")
    md = out / "posture_audit.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")

    print("写盘：%s" % csv_path)
    print("写盘：%s" % md)
    print("\n%d 次 rollout：survived=%d collapse=%d upright_trip=%d" % (
        len(rows), counts.get("survived", 0), counts.get("collapse", 0), counts.get("upright_trip", 0)))
    print("视频校验：" + ("全部一致" if check_ok else "有不一致"))
    for cell, seed, expect, got, ok in checks:
        if not ok:
            print("  %s/%d: 期望 %s 实得 %s" % (cell, seed, expect, got))
    return 0 if check_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
