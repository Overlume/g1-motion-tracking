#!/usr/bin/env bash
# LAB27 —— 多片段联合训练策略 vs M4 基线：4 个窗口 × N 种子 × 2 策略
#
# 用法（项目根目录）:
#   LAB27_RUN=runs/20261126-001-lab27-multiclip bash scripts/lab27_eval.sh run   # 80 次 ≈ 50 min
#   LAB27_RUN=runs/20261126-001-lab27-multiclip bash scripts/lab27_eval.sh report
#
# 口径与 LAB09/M4/LAB20 一致：scripts/eval_tracking.py，终止检测开、启动 DR 关（默认）、999 步。
# 四个窗口：
#   train   = 训练片段（S3 0 s）
#   heldout = 同片段未见窗口（S3 120 s）——主指标（M4 = 0.100）
#   unseen  = 未见片段（S2 134 s，M4 = 0.025）
#   dance   = 新片段未见窗口（dance1 60–80 s 地面校正版）——训练里出现的是 dance1 0–20 s
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN="${LAB27_RUN:-}"
SEEDS="$(echo "${LAB27_SEEDS:-101 102 103 104 105 106 107 108 109 110}")"
POLICIES="$(echo "${LAB27_POLICIES:-m4 lab27}")"
WINDOWS="$(echo "${LAB27_WINDOWS:-train heldout unseen dance}")"
FORCE="${LAB27_FORCE:-0}"

die() { echo "[ERR] $*" >&2; exit 1; }
say() { echo "[lab27] $*"; }

if [ -z "$RUN" ]; then
  RUN="$(ls -d runs/*-lab27-multiclip 2>/dev/null | sort | tail -1 || true)"
fi
[ -n "$RUN" ] && [ -d "$RUN" ] || die "找不到 LAB27 run 目录（用 LAB27_RUN=runs/... 指定）"
[ "$(basename "$RUN")" != "runs" ] || die "RUN 不能是 runs 本身"

m4_ckpt() {
  local run ck
  run="$(ls -d runs/*-m4-w15-finetune 2>/dev/null | sort | tail -1 || true)"
  [ -n "$run" ] || die "找不到 M4 训练 run（runs/*-m4-w15-finetune）"
  ck="$(ls "$run"/logs/g1_tracking/*/model_52998.pt 2>/dev/null | head -1 || true)"
  [ -n "$ck" ] || ck="$(ls "$run"/logs/g1_tracking/*/model_*.pt 2>/dev/null | sort -V | tail -1 || true)"
  [ -n "$ck" ] || die "找不到 M4 ckpt（$run）"
  echo "$ck"
}

lab27_ckpt() {
  local ck
  # 必须排除 pretrained/（热启动权重副本，2026-11-22 在 LAB20 踩过）
  ck="$(ls "$RUN"/logs/g1_tracking/*/model_*.pt 2>/dev/null | grep -v "/pretrained/" | sort -V | tail -1 || true)"
  [ -n "$ck" ] || die "找不到 LAB27 ckpt（$RUN/logs/g1_tracking/*/model_*.pt）；训练是否跑完？"
  echo "$ck"
}

motion_file() {
  case "$1" in
    train)   echo "data/processed/fight1_s3_000s_20s.npz" ;;
    heldout) echo "data/processed/fight1_s3_120s_20s.npz" ;;
    unseen)  echo "data/processed/fight1_s2_134s_20s.npz" ;;
    dance)   echo "data/processed/dance1_060s_20s_gfix.npz" ;;
    *) die "未知窗口 $1（只能是 train / heldout / unseen / dance）" ;;
  esac
}

run_stage() {
  local m4 l27 ck mo
  m4="$(m4_ckpt)"; l27="$(lab27_ckpt)"
  say "M4    ckpt: $m4（sha $(sha256sum "$m4" | cut -d' ' -f1)）"
  say "LAB27 ckpt: $l27（sha $(sha256sum "$l27" | cut -d' ' -f1)）"
  say "种子: $SEEDS"

  local up0 up1
  up0=$(cut -d' ' -f1 /proc/uptime)
  for p in $POLICIES; do
    case "$p" in m4) ck="$m4" ;; lab27) ck="$l27" ;; *) die "未知策略 $p" ;; esac
    for w in $WINDOWS; do
      mo="$(motion_file "$w")"
      [ -f "$mo" ] || die "缺数据 $mo（dance 窗口需先跑 scripts/lab27_ground_fix.py）"
      for s in $SEEDS; do
        local out="$RUN/eval/$p/$w/seed${s}"
        if [ -f "$out/summary.json" ] && [ "$FORCE" != "1" ]; then
          say "skip $p/$w/seed$s（已有 summary.json）"
          continue
        fi
        mkdir -p "$out"
        say "$p $w seed $s"
        MUJOCO_GL=egl environment/.venv/bin/python scripts/eval_tracking.py \
          --checkpoint-file "$ck" --motion-file "$mo" --seed "$s" --out "$out" \
          > "$RUN/eval/$p/$w/seed${s}.log" 2>&1 || echo "[ERR] $p/$w/seed$s 失败"
      done
    done
  done
  up1=$(cut -d' ' -f1 /proc/uptime)
  say "评估墙钟（/proc/uptime）：$(awk -v a="$up0" -v b="$up1" 'BEGIN{printf "%.0f", b-a}') s"
}

report_stage() {
  environment/.venv/bin/python scripts/lab27_report.py --run "$RUN" --title "LAB27 multiclip vs M4" --figure
}

case "${1:-}" in
  run)    run_stage; report_stage ;;
  report) report_stage ;;
  *)      sed -n '2,13p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
