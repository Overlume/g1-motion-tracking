#!/usr/bin/env python3
"""LAB05 / M3-b 汇总：把 A/B 两组的训练日志与评估小结拼成一张对比表。

用法:
  environment/.venv/bin/python scripts/lab05_ab_report.py --eval-dir runs/YYYYMMDD-NNN-m3b-eval

输入:
  <eval-dir>/A/seed*/summary.json           评估小结（scripts/eval_tracking.py 产生）
  <eval-dir>/A/train_run.txt               该组训练 run 目录
  <train_run>/train.log                    训练日志

输出:
  <eval-dir>/comparison.json, <eval-dir>/comparison.md，并打印表格。

注意：按 D11，fight 类片段只报分布（N、中位数、范围、摔倒次数），不引用单次 fall_step 作结论。
"""
import argparse
import json
import re
import statistics as st
from pathlib import Path

METRIC_KEYS = [
    ("track_completion", "完成度"),
    ("root_err_mean_cm", "root xy 误差均值 cm"),
    ("drift_rate_cm_s", "漂移速率 cm/s"),
    ("r_mpkpe_mean_cm", "R-MPKPE 均值 cm"),
    ("mpkpe_mean_cm", "MPKPE 均值 cm"),
]

RE_ITER = re.compile(r"Iteration time:\s*(-?[\d.]+)s")
RE_REWARD = re.compile(r"Mean reward:\s*(-?[\d.]+)")
RE_EPLEN = re.compile(r"Mean episode length:\s*(-?[\d.]+)")
RE_ANCHOR_POS = re.compile(r"Metrics/motion/error_anchor_pos:\s*(-?[\d.]+)")
RE_TERM_ANCHOR = re.compile(r"Episode_Termination/anchor_pos:\s*(-?[\d.]+)")


def read_eval(eval_dir: Path, policy: str):
    rows = []
    for sp in sorted((eval_dir / policy).glob("seed*/summary.json")):
        with open(sp, encoding="utf-8") as f:
            s = json.load(f)
        seg = s.get("longest_segment") or {}
        rows.append(
            {
                "seed": s.get("seed"),
                "fall_detected": bool(s.get("fall_detected")),
                "fall_step": s.get("fall_step"),
                "reset_count": s.get("reset_count"),
                "seconds_run": s.get("seconds_run"),
                "track_completion": s.get("track_completion"),
                "root_err_mean_cm": s.get("root_err_mean_cm"),
                "drift_rate_cm_s": seg.get("drift_rate_cm_s"),
                "r_mpkpe_mean_cm": s.get("r_mpkpe_mean_cm"),
                "mpkpe_mean_cm": s.get("mpkpe_mean_cm"),
            }
        )
    return rows


def summarize(rows):
    out = {"n": len(rows), "falls": sum(1 for r in rows if r["fall_detected"])}
    fall_steps = [r["fall_step"] / 50.0 for r in rows if r["fall_detected"] and r["fall_step"]]
    if fall_steps:
        out["fall_time_s"] = {
            "median": round(st.median(fall_steps), 2),
            "min": round(min(fall_steps), 2),
            "max": round(max(fall_steps), 2),
        }
    for key, _ in METRIC_KEYS:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if vals:
            out[key] = {
                "median": round(st.median(vals), 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
            }
    return out


def tail_metrics(log: Path, iters: int = 30):
    """训练日志里最后 iters 次迭代的平均值（跳过负值等异常迭代时间）。"""
    if not log.exists():
        return {}
    txt = log.read_text(encoding="utf-8", errors="replace")
    acc = {"reward": [], "eplen": [], "anchor_pos": [], "term_anchor": []}
    for m in RE_REWARD.finditer(txt):
        acc["reward"].append(float(m.group(1)))
    for m in RE_EPLEN.finditer(txt):
        acc["eplen"].append(float(m.group(1)))
    for m in RE_ANCHOR_POS.finditer(txt):
        acc["anchor_pos"].append(float(m.group(1)))
    for m in RE_TERM_ANCHOR.finditer(txt):
        acc["term_anchor"].append(float(m.group(1)))
    times = [float(m.group(1)) for m in RE_ITER.finditer(txt)]
    times = [t for t in times[1:] if t > 0]
    out = {}
    for k, v in acc.items():
        if v:
            out[k + "_tail"] = round(st.mean(v[-iters:]), 4)
    if times:
        out["iter_time_median_s"] = round(st.median(times), 3)
    out["iter_time_negative_samples"] = sum(1 for m in RE_ITER.finditer(txt) if float(m.group(1)) <= 0)
    return out


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--title", default="M3-b A/B 对比（LAB05）", help="comparison.md 的标题")
    ap.add_argument("--labels", default="A = 原始奖励（weight 0.5 / std 0.3）；B = 修改后（weight 1.0 / std 0.15）。", help="标题下的口径说明，LAB06 用它写清唯变量")
    args = ap.parse_args()
    ed = Path(args.eval_dir)

    data = {}
    for policy in ("A", "B"):
        rows = read_eval(ed, policy)
        if not rows:
            raise SystemExit(f"[ERR] {ed}/{policy} 下没有 seed*/summary.json")
        train_run = (ed / policy / "train_run.txt").read_text(encoding="utf-8").strip()
        tr = tail_metrics(Path(train_run) / "train.log")
        data[policy] = {"rows": rows, "summary": summarize(rows), "train": tr, "train_run": train_run}

    lines = [f"# {args.title}", ""]
    lines.append(args.labels)
    lines.append("")
    for policy in ("A", "B"):
        d = data[policy]
        lines.append(f"- {policy} 组训练：{d['train_run']}")
    lines.append("")
    lines.append("## 评估分布（同一动作、同一评估脚本；N 见各行）")
    lines.append("")
    head = "| 指标 | A 中位 (min–max) | B 中位 (min–max) |"
    lines.append(head)
    lines.append("| --- | --- | --- |")

    def cell(policy, key):
        s = data[policy]["summary"].get(key)
        if not s:
            return "n/a"
        return f"{fmt(s['median'])} ({fmt(s['min'])}–{fmt(s['max'])})"

    lines.append(f"| 摔倒次数 / N | {data['A']['summary']['falls']}/{data['A']['summary']['n']} | {data['B']['summary']['falls']}/{data['B']['summary']['n']} |")
    for key, label in METRIC_KEYS:
        lines.append(f"| {label} | {cell('A', key)} | {cell('B', key)} |")
    for policy in ("A", "B"):
        ft = data[policy]["summary"].get("fall_time_s")
        if ft:
            lines.append(f"| 摔倒时间 s（{policy}） | {ft['median']} ({ft['min']}–{ft['max']}) | |")
    lines.append("")
    lines.append("## 逐种子")
    lines.append("")
    lines.append("| 组 | seed | 摔倒 | fall_step | 完成度 | root_err cm | 漂移 cm/s |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for policy in ("A", "B"):
        for r in data[policy]["rows"]:
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    policy, r["seed"], "是" if r["fall_detected"] else "否", r["fall_step"],
                    fmt(r["track_completion"]), fmt(r["root_err_mean_cm"]), fmt(r["drift_rate_cm_s"]),
                )
            )
    lines.append("")
    lines.append("## 训练侧（最后 30 次迭代均值）")
    lines.append("")
    lines.append("| 项 | A | B |")
    lines.append("| --- | --- | --- |")
    for key, label in [
        ("reward_tail", "Mean reward"),
        ("eplen_tail", "Mean episode length"),
        ("anchor_pos_tail", "Metrics error_anchor_pos"),
        ("term_anchor_tail", "Episode_Termination/anchor_pos"),
        ("iter_time_median_s", "迭代耗时中位 s"),
        ("iter_time_negative_samples", "负值迭代耗时样本数（WSL 时钟跳变）"),
    ]:
        lines.append(f"| {label} | {fmt(data['A']['train'].get(key, 'n/a'))} | {fmt(data['B']['train'].get(key, 'n/a'))} |")
    lines.append("")
    lines.append("## 读数提醒")
    lines.append("")
    lines.append("- 按 D11：fight 类片段只报分布（N、中位、范围、摔/不摔次数），不得拿单次 fall_step 当结论。")
    lines.append("- 训练侧 Mean reward 的绝对量在 A/B 之间不可直接比（奖励函数本身变了）；可比的是 error_anchor_pos / anchor_pos 终止次数等物理量。")
    lines.append("- 若两组评估分布重叠，结论应写“在 300 iter 尺度上不可分辨”，而不是“B 更好”。")
    lines.append("- 负值迭代耗时是本机 WSL2 实时钟 ±7 s 跳变造成的（见 D14），不是训练异常；中位数已剔除非正值，但正值的绝对值同样被时钟漂移影响（约 -10%），跨会话比较耗时只能比同机同时段的相对值。")

    md = "\n".join(lines) + "\n"
    (ed / "comparison.md").write_text(md, encoding="utf-8")
    with open(ed / "comparison.json", "w", encoding="utf-8") as f:
        json.dump({p: {"summary": data[p]["summary"], "train": data[p]["train"]} for p in ("A", "B")}, f, ensure_ascii=False, indent=2)
    print(md)


if __name__ == "__main__":
    main()
