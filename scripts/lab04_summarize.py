#!/usr/bin/env python3
"""Summarize a LAB04 smoke-training run: iteration time, VRAM, checkpoints.

Usage: lab04_summarize.py RUN_DIR NUM_ENVS ITERS SEED MOTION_FILE [TRAIN_RC]
"""

import glob
import json
import os
import re
import statistics
import sys
import time


def quantile(values, q):
    if not values:
        return None
    ys = sorted(values)
    pos = (len(ys) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def pick(pattern, text):
    return [float(m) for m in re.findall(pattern, text)]


def main():
    run, num_envs, iters, seed, motion = sys.argv[1:6]
    train_rc = int(sys.argv[6]) if len(sys.argv) > 6 else 0

    log_path = os.path.join(run, "train.log")
    text = ""
    if os.path.exists(log_path):
        with open(log_path, "r", errors="replace") as fh:
            text = fh.read()

    iter_s = pick(r"Iteration time:\s*(-?\d+(?:\.\d+)?)s", text)
    rewards = pick(r"Mean reward:\s*(-?\d+(?:\.\d+)?)", text)
    ep_lens = pick(r"Mean episode length:\s*(-?\d+(?:\.\d+)?)", text)
    stds = pick(r"Mean action std:\s*(-?\d+(?:\.\d+)?)", text)

    vram = []
    vram_path = os.path.join(run, "vram_mib.txt")
    if os.path.exists(vram_path):
        with open(vram_path, "r", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line.isdigit():
                    vram.append(int(line))

    ckpt_paths = sorted(
        glob.glob(os.path.join(run, "logs", "*", "*", "model_*.pt")),
        key=lambda p: int(re.search(r"model_(\d+)", os.path.basename(p)).group(1)),
    )
    onnx_paths = sorted(glob.glob(os.path.join(run, "logs", "*", "*", "*.onnx")))

    delta = None
    if len(ckpt_paths) >= 2:
        try:
            import torch

            a = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)
            b = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=False)
            sa = a.get("actor_state_dict", {})
            sb = b.get("actor_state_dict", {})
            total = 0.0
            count = 0
            changed = 0
            for key, va in sa.items():
                if not torch.is_tensor(va) or not va.is_floating_point():
                    continue
                if key.endswith("_mean") or key.endswith("_var"):
                    continue
                vb = sb[key]
                diff = (vb.float() - va.float()).abs()
                total += float(diff.sum())
                count += va.numel()
                if float(diff.max()) > 1e-6:
                    changed += 1
            delta = {
                "mean_abs_delta": (total / count) if count else None,
                "tensors": len(sa),
                "tensors_changed": changed,
                "iter_from": a.get("iter"),
                "iter_to": b.get("iter"),
            }
        except Exception as exc:
            delta = {"error": repr(exc)}

    steady = iter_s[1:] if len(iter_s) > 1 else iter_s
    # WSL2 墙上时钟会偶发向后跳，导致个别 "Iteration time" 为负；
    # 这些样本是日志伪影，不参与稳态统计，但保留计数以便核对。
    nonpositive = [t for t in iter_s if t is not None and t <= 0]
    steady_ok = [t for t in steady if t is not None and t > 0]
    iter_stats = None
    if iter_s:
        iter_stats = {
            "first": iter_s[0],
            "median": statistics.median(steady_ok) if steady_ok else None,
            "p95": quantile(steady_ok, 0.95) if steady_ok else None,
            "min": min(steady_ok) if steady_ok else None,
            "max": max(iter_s),
            "n": len(iter_s),
            "nonpositive_samples": len(nonpositive),
        }

    with open(os.path.join(run, "iter_times.csv"), "w") as fh:
        fh.write("iter,seconds\n")
        for i, value in enumerate(iter_s):
            fh.write("%d,%s\n" % (i, value))

    errors = []
    for line in text.splitlines():
        low = line.lower()
        if "traceback" in low or "error" in low or "oom" in low:
            errors.append(line.strip()[:200])
    errors = errors[-10:]

    summary = {
        "purpose": "LAB04 smoke training: mechanics check (iteration time / VRAM / checkpoint reload), NOT a learning run",
        "run_dir": run,
        "num_envs": int(num_envs),
        "iterations_requested": int(iters),
        "iterations_captured": len(iter_s),
        "samples_per_iter": int(num_envs) * 24,
        "seed": int(seed),
        "motion_file": motion,
        "train_exit_code": train_rc,
        "iter_time_s": iter_stats,
        "vram_mib": {
            "peak": max(vram) if vram else None,
            "median": statistics.median(vram) if vram else None,
            "samples": len(vram),
        },
        "checkpoints": [os.path.relpath(p, run) for p in ckpt_paths],
        "checkpoint_bytes": [os.path.getsize(p) for p in ckpt_paths],
        "onnx": [os.path.relpath(p, run) for p in onnx_paths],
        "actor_delta": delta,
        "mean_reward": {
            "first": rewards[0] if rewards else None,
            "last": rewards[-1] if rewards else None,
        },
        "mean_episode_length": {
            "first": ep_lens[0] if ep_lens else None,
            "last": ep_lens[-1] if ep_lens else None,
        },
        "mean_action_std": {
            "first": stds[0] if stds else None,
            "last": stds[-1] if stds else None,
        },
        "log_error_lines": errors,
        "summarized_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }

    with open(os.path.join(run, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    def fmt(value, spec):
        return "n/a" if value is None else format(value, spec)

    print()
    print("[lab04] ---------------- 摘要 ----------------")
    print("run: %s" % run)
    print("规模: %s env x %s iter, seed=%s, %s samples/iter" % (num_envs, iters, seed, int(num_envs) * 24))
    print("迭代: 记录到 %d/%s 条迭代时间" % (len(iter_s), iters))
    if iter_stats:
        print(
            "耗时: 首次 %s s | 稳态中位 %s s | p95 %s s | 最慢 %s s"
            % (
                fmt(iter_stats["first"], ".2f"),
                fmt(iter_stats["median"], ".2f"),
                fmt(iter_stats["p95"], ".2f"),
                fmt(iter_stats["max"], ".2f"),
            )
        )
    if vram:
        print("显存: 峰值 %d MiB | 中位 %.0f MiB | 采样 %d 点" % (max(vram), statistics.median(vram), len(vram)))
    print("检查点: %s" % (", ".join(os.path.basename(p) for p in ckpt_paths) or "无"))
    if delta and "mean_abs_delta" in delta:
        print(
            "actor 参数: mean |dw| = %.3e（%d/%d 个张量有变化，iter %s -> %s）"
            % (
                delta["mean_abs_delta"] or 0.0,
                delta["tensors_changed"],
                delta["tensors"],
                delta["iter_from"],
                delta["iter_to"],
            )
        )
    if rewards:
        print("mean reward: 首次 %.2f -> 末次 %.2f" % (rewards[0], rewards[-1]))
    else:
        print("mean reward: 日志里没抓到（迭代太少或训练失败）")
    if len(iter_s) < int(iters):
        print("[warn] 只抓到 %d/%s 次迭代：训练可能没跑完，看 train.log 末尾" % (len(iter_s), iters))
    if train_rc != 0:
        print("[warn] 训练退出码 %d，看 train.log 末尾 40 行" % train_rc)
    print("[lab04] ------------------------------------")
    return 0


if __name__ == "__main__":
    sys.exit(main())
