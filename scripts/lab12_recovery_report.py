#!/usr/bin/env python
"""LAB12 恢复实验判分：读 --ignore-terminations 的 metrics.csv + summary.json，
判定推后 3 s 内"没崩 / 崩了又恢复 / 崩了没起来"。

判据（与 scripts/m5_posture_audit.py 同一套阈值）：
- 偏离帧：ref_pelvis_z - pelvis_z >= 0.08 m，或 tilt_deg - ref_tilt_deg >= 13 deg
- 崩溃：窗口内连续 >= 10 帧（0.2 s）偏离
- 窗口：[8.0, 11.2] s（推力 8.0-8.2 s，之后 3 s）
  - down：summary 的 fall_detected 为真（骨盆 <0.15 m 连续 5 步，eval 会在此处 break），或崩过且末尾仍偏离
  - transient：崩过，但数据最后 0.5 s 回到参考附近（平均下移 < 0.05 m 且倾角差 < 8 deg）
  - stable：窗口内没有连续 10 帧偏离
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PUSH_S = 8.0
WINDOW_END_S = 11.2
TAIL_S = 0.5
DROP_M = 0.08
TILT_DEG = 13.0
RUN_FRAMES = 10
RECOVER_DROP_M = 0.05
RECOVER_TILT_DEG = 8.0


def analyse(m, summary):
    t = np.asarray(m["time_s"], dtype=float)
    pz = np.asarray(m["pelvis_z"], dtype=float)
    rpz = np.asarray(m["ref_pelvis_z"], dtype=float)
    ti = np.asarray(m["tilt_deg"], dtype=float)
    rti = np.asarray(m["ref_tilt_deg"], dtype=float)
    root = np.asarray(m["root_err_cm"], dtype=float)

    drop = rpz - pz
    tiltex = ti - rti
    dev = (drop >= DROP_M) | (tiltex >= TILT_DEG)

    win = (t >= PUSH_S) & (t <= WINDOW_END_S)
    idx = np.where(win)[0]
    if idx.size == 0:
        raise ValueError("窗口 8.0-11.2 s 内没有数据（data ends at %.2f s）" % t[-1])

    best = cur = 0
    best_end = -1
    for k in idx:
        if dev[k]:
            cur += 1
            if cur > best:
                best = cur
                best_end = k
        else:
            cur = 0
    first = int(idx[np.argmax(dev[idx])]) if dev[idx].any() else -1

    # 末尾窗口：数据可能因 fall 提前 break，所以取"可用数据的最后 0.5 s"
    t_tail = max(t[idx[-1]] - TAIL_S, t[idx[0]])
    tail_idx = idx[t[idx] >= t_tail]
    end_drop = float(drop[tail_idx].mean()) if tail_idx.size else float("nan")
    end_tiltex = float(tiltex[tail_idx].mean()) if tail_idx.size else float("nan")
    recovered = (end_drop < RECOVER_DROP_M) and (end_tiltex < RECOVER_TILT_DEG)
    fell = bool(summary.get("fall_detected", False))

    if fell:
        cls = "down"
    elif best < RUN_FRAMES:
        cls = "stable"
    elif recovered:
        cls = "transient"
    else:
        cls = "down"

    return dict(
        cls=cls,
        fell=fell,
        t_end=float(t[-1]),
        longest_dev_frames=int(best),
        longest_dev_s=round(best * 0.02, 2),
        t_first_dev=round(float(t[first]), 2) if first >= 0 else None,
        end_drop=end_drop,
        end_tiltex=end_tiltex,
        max_drop=float(drop[idx].max()),
        max_tiltex=float(tiltex[idx].max()),
        pelvis_z_min=float(pz.min()),
        root_err_end_cm=float(root[tail_idx].mean()) if tail_idx.size else float("nan"),
        resets=int(summary.get("reset_count", 0)),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="LAB12 恢复实验判分")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    run = Path(args.dir)
    rows, skipped = [], []
    for cell in sorted(d for d in run.iterdir() if d.is_dir() and d.name.startswith(("px_", "nx_", "py_", "ny_"))):
        for sd in sorted(cell.iterdir()):
            f = sd / "metrics.csv"
            if not sd.is_dir() or not f.exists():
                continue
            try:
                m = np.genfromtxt(f, delimiter=",", names=True)
                sj = json.loads((sd / "summary.json").read_text(encoding="utf-8"))
                r = analyse(m, sj)
            except Exception as e:
                skipped.append((cell.name, sd.name, "%s: %s" % (type(e).__name__, e)))
                continue
            rows.append(dict(cell=cell.name, seed=int(sd.name.replace("seed", "")), **r))

    if not rows:
        raise SystemExit("没有可判分的 rollout")

    prefix = Path(args.out_prefix) if args.out_prefix else run / "recovery"
    with open(str(prefix) + ".json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    def med(v):
        v = [x for x in v if x == x]
        return float(np.median(v)) if v else float("nan")

    cells = {}
    for r in rows:
        cells.setdefault(r["cell"], []).append(r)

    L = []
    add = L.append
    add("# LAB12 恢复实验判分")
    add("")
    add("- 输入：%s 下 %d 个格、%d 次 rollout" % (run, len(cells), len(rows)))
    add("- 判据：偏离 = 骨盆比参考低 >= %.2f m 或倾角比参考歪 >= %.1f deg；窗口 %.1f–%.1f s；"
        "连续 >= %d 帧（%.1f s）偏离记崩溃；数据最后 %.1f s 回到参考附近（平均下移 < %.2f m 且倾角差 < %.0f deg）记恢复"
        % (DROP_M, TILT_DEG, PUSH_S, WINDOW_END_S, RUN_FRAMES, RUN_FRAMES * 0.02, TAIL_S, RECOVER_DROP_M, RECOVER_TILT_DEG))
    add("- down 包括 eval 的 fall 判据命中（骨盆 <0.15 m 连续 5 步，脚本在此 break，t_end 就是倒下时刻）")
    add("")
    add("## 汇总")
    add("")
    add("| 格 | n | stable | transient（崩了又恢复）| down（崩了没起来）| 倒下种子 | 数据结束时刻中位 (s) | 最长偏离中位 (s) | 末尾下移中位 (m) |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for cell in sorted(cells):
        rs = cells[cell]
        c = dict((k, sum(1 for r in rs if r["cls"] == k)) for k in ("stable", "transient", "down"))
        add("| %s | %d | %d | %d | %d | %d | %.2f | %.2f | %+.3f |" % (
            cell, len(rs), c["stable"], c["transient"], c["down"], c["down"],
            med([r["t_end"] for r in rs]), med([r["longest_dev_s"] for r in rs]),
            med([r["end_drop"] for r in rs])))
    add("")
    add("## 逐次")
    add("")
    add("| 格 | 种子 | 判定 | fall 判据 | 数据结束 (s) | 最长连续偏离 (s) | 首次偏离 (s) | 骨盆最低 (m) | 最大倾角差 (deg) | 末尾下移 (m) | 末尾 root 误差 (cm) |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in sorted(rows, key=lambda x: (x["cell"], x["seed"])):
        add("| %s | %d | %s | %s | %.2f | %.2f | %s | %.3f | %.1f | %+.3f | %.1f |" % (
            r["cell"], r["seed"], r["cls"], "是" if r["fell"] else "否", r["t_end"],
            r["longest_dev_s"], ("%.2f" % r["t_first_dev"]) if r["t_first_dev"] is not None else "-",
            r["pelvis_z_min"], r["max_tiltex"], r["end_drop"], r["root_err_end_cm"]))
    if skipped:
        add("")
        add("## 跳过")
        add("")
        for cell, seed, err in skipped:
            add("- %s/%s：%s" % (cell, seed, err))
    add("")
    Path(str(prefix) + ".md").write_text("\n".join(L) + "\n", encoding="utf-8")

    tot = dict((k, sum(1 for r in rows if r["cls"] == k)) for k in ("stable", "transient", "down"))
    print("写盘：%s.{md,json}" % prefix)
    print("%d 次 rollout：stable=%d transient=%d down=%d" % (len(rows), tot["stable"], tot["transient"], tot["down"]))
    if skipped:
        print("跳过 %d 个 rollout" % len(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
