#!/usr/bin/env python
"""LAB28 追加分析（零 GPU）：窗口难度、内容新颖度、首段对齐误差。

读 <run>/eval/<policy>/w<SSS>/seed*/ 下的 summary.json 与 metrics.csv，以及
data/processed/dance1_<SSS>s_20s_gfix.npz，不训练、不评估。

输出：
  <run>/lab28_audit.md    人读表 + 相关 + 口径说明
  <run>/lab28_audit.json  机器读

口径：
  参考关节速度 = joint_pos 逐帧差分 × fps 的 L2 均值（rad/s）
  参考根速度   = root_pos_w 逐帧差分 × fps 的 L2 均值（m/s）
  新颖度 dmin  = 该窗每帧到「训练窗口 0-20 s」所有帧的最小距离（29 维关节空间，
                 按训练窗口的均值/标准差归一化），取全窗均值
  首段对齐误差 = 两策略都还没 reset 的共同前缀 K = min(各自最小首段长度, cap) 上的 MPKPE 均值
"""

import argparse
import json
import pathlib

import numpy as np
from scipy.stats import spearmanr

POLICIES = ("m4", "lab27")


def first_seg_len(metrics_csv: pathlib.Path) -> int:
    n = 0
    with open(metrics_csv) as f:
        next(f)
        for line in f:
            n += 1
            if line.split(",")[3] in ("1", "True", "true"):
                break
    return n


def prefix_mean(metrics_csv: pathlib.Path, k: int, col: int) -> float:
    vals = []
    with open(metrics_csv) as f:
        next(f)
        for i, line in enumerate(f):
            if i >= k:
                break
            vals.append(float(line.rstrip("\n").split(",")[col]))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--starts", default="0 20 40 60 80 100")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--cap", type=int, default=200, help="首段对齐 K 的上限（帧）")
    ap.add_argument("--seeds", default="101 102 103 104 105 106 107 108 109 110")
    args = ap.parse_args()

    run = pathlib.Path(args.run).resolve()
    if not run.is_dir():
        raise SystemExit("run 目录不存在：%s" % run)
    starts = [int(x) for x in args.starts.split()]
    seeds = [int(x) for x in args.seeds.split()]

    # 1) 参考难度 + 关节位置
    ref, jpos = {}, {}
    for s in starts:
        p = pathlib.Path("data/processed/dance1_%03ds_20s_gfix.npz" % s)
        if not p.exists():
            raise SystemExit("缺窗口数据：%s" % p)
        d = np.load(p)
        jp = d["joint_pos"].astype(np.float64)
        jv = np.diff(jp, axis=0) * args.fps
        rv = np.diff(d["root_pos_w"].astype(np.float64), axis=0) * args.fps
        jpos[s] = jp
        ref[s] = {
            "joint_speed": float(np.linalg.norm(jv, axis=1).mean()),
            "root_speed": float(np.linalg.norm(rv, axis=1).mean()),
        }

    # 2) 内容新颖度：到训练窗口（第一个起点）的最小距离
    base = jpos[starts[0]]
    mu = base.mean(0, keepdims=True)
    sd = base.std(0, keepdims=True) + 1e-6
    bn = (base - mu) / sd
    novel = {}
    for s in starts:
        q = (jpos[s] - mu) / sd
        d2 = ((q[:, None, :] - bn[None, :, :]) ** 2).sum(-1)
        dmin = np.sqrt(d2.min(1))
        novel[s] = {"dmin_mean": float(dmin.mean()), "dmin_p10": float(np.percentile(dmin, 10))}

    # 3) 每窗口汇总
    rows = []
    for s in starts:
        comp, mpk, root = {}, {}, {}
        for p in POLICIES:
            cs, ms, rs = [], [], []
            for seed in seeds:
                f = run / "eval" / p / ("w%03d" % s) / ("seed%d" % seed) / "summary.json"
                if not f.exists():
                    continue
                d = json.loads(f.read_text())
                cs.append(d["track_completion"])
                ms.append(d["mpkpe_mean_cm"])
                rs.append(d["root_err_mean_cm"])
            if not cs:
                raise SystemExit("缺 cell：%s w%03d" % (p, s))
            comp[p] = float(np.mean(cs))
            mpk[p] = float(np.mean(ms))
            root[p] = float(np.mean(rs))

        k_policy = {}
        for p in POLICIES:
            lens = []
            for seed in seeds:
                f = run / "eval" / p / ("w%03d" % s) / ("seed%d" % seed) / "metrics.csv"
                lens.append(first_seg_len(f))
            k_policy[p] = min(lens)
        K = max(1, min(k_policy["m4"], k_policy["lab27"], args.cap))

        pre = {}
        for p in POLICIES:
            f = run / "eval" / p / ("w%03d" % s) / ("seed%d" % seeds[0]) / "metrics.csv"
            pre[p] = {"mpkpe": prefix_mean(f, K, 5), "root": prefix_mean(f, K, 7)}

        rows.append({
            "start": s,
            "m4_comp": comp["m4"], "l27_comp": comp["lab27"], "delta": comp["lab27"] - comp["m4"],
            "m4_mpkpe": mpk["m4"], "l27_mpkpe": mpk["lab27"],
            "m4_root": root["m4"], "l27_root": root["lab27"],
            "K": K, "m4_mpkpe_pre": pre["m4"]["mpkpe"], "l27_mpkpe_pre": pre["lab27"]["mpkpe"],
            "m4_root_pre": pre["m4"]["root"], "l27_root_pre": pre["lab27"]["root"],
            **ref[s], **novel[s],
        })

    d = np.array([r["delta"] for r in rows])
    dist = np.array([r["start"] for r in rows], dtype=float)
    m4c = np.array([r["m4_comp"] for r in rows])
    nov = np.array([r["dmin_mean"] for r in rows])
    js = np.array([r["joint_speed"] for r in rows])

    corr = {
        "distance_vs_delta": spearmanr(dist, d),
        "novelty_vs_delta": spearmanr(nov, d),
        "m4comp_vs_joint_speed": spearmanr(m4c, js),
        "novelty_vs_m4comp": spearmanr(nov, m4c),
    }
    corr = {k: {"rho": float(v.statistic), "p": float(v.pvalue)} for k, v in corr.items()}

    out = {"run": str(run), "starts": starts, "rows": rows, "spearman": corr}
    (run / "lab28_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")

    L = ["# LAB28 追加分析（零 GPU）：窗口难度、内容新颖度、首段对齐误差", "",
         "- run：%s" % run,
         "- 脚本：scripts/lab28_audit.py（读 6 个 npz + 120 个 summary.json / metrics.csv，不训练）", "",
         "## 1. 每窗口一览", "",
         "| 窗口 | M4 完成 | LAB27 完成 | Δ | 参考关节速度 rad/s | 参考根速度 m/s | 新颖度 dmin | 整段 MPKPE M4→L27 | 首段 K | 首段 MPKPE M4→L27 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| %d–%d s | %.3f | %.3f | %+.3f | %.2f | %.2f | %.2f | %.1f → %.1f | %d | %.1f → %.1f |" % (
            r["start"], r["start"] + 20, r["m4_comp"], r["l27_comp"], r["delta"],
            r["joint_speed"], r["root_speed"], r["dmin_mean"],
            r["m4_mpkpe"], r["l27_mpkpe"], r["K"], r["m4_mpkpe_pre"], r["l27_mpkpe_pre"]))
    L += ["", "## 2. 口径说明", "",
          "- **参考关节速度** = 该窗口 npz 的 joint_pos 逐帧差分 × fps 的 L2 范数均值（rad/s）。",
          "- **参考根速度** = root_pos_w 逐帧差分 × fps 的 L2 范数均值（m/s）。",
          "- **新颖度 dmin** = 该窗每帧到「训练窗口（第一个起点，dance1 0–20 s）」所有帧的最小距离（按训练窗口归一化的 29 维关节空间），取全窗均值；0 = 与训练窗口完全相同。",
          "- **整段 MPKPE** = summary.json 的 mpkpe_mean_cm（含 reset 之后的所有帧，跨策略不可直接比：存活长度不同）。",
          "- **首段 MPKPE** = 两策略都还没 reset 的共同前缀 K 帧（K = min(两策略各自最小首段长度, cap)）上的均值；同一参考帧、同一时刻，可跨策略比。",
          "", "## 3. Spearman（探索性，n=%d）" % len(rows), ""]
    for k, cn in (("distance_vs_delta", "距离 vs Δ完成度"),
                  ("novelty_vs_delta", "新颖度 vs Δ完成度"),
                  ("m4comp_vs_joint_speed", "M4 完成度 vs 参考关节速度"),
                  ("novelty_vs_m4comp", "新颖度 vs M4 完成度")):
        c = corr[k]
        L.append("- %s：**%+.3f（p = %.3f）**" % (cn, c["rho"], c["p"]))
    L += ["", "> n 很小、窗口间不独立（同一段动作相邻 20 s），只能当提示，不能当定论。", ""]
    (run / "lab28_audit.md").write_text("\n".join(L) + "\n")

    print("\n".join(L))
    print("[lab28-audit] %s" % (run / "lab28_audit.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
