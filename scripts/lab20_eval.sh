#!/usr/bin/env bash
# LAB20 —— DR 加强策略 vs M4 基线：3 个窗口 × N 种子 × 2 策略
#
# 用法（项目根目录）:
#   LAB20_RUN=runs/20261121-003-lab20-dr bash scripts/lab20_eval.sh run      # 60 次 ≈ 40 min
#   LAB20_RUN=runs/20261121-003-lab20-dr bash scripts/lab20_eval.sh report   # 只重算表
#
# 口径与 LAB09/M4 一致：scripts/eval_tracking.py，终止检测开、启动 DR 关（默认）、999 步。
# 三个窗口：train = 训练片段（S3 0 s）/ heldout = 同片段未见窗口（S3 120 s）/ unseen = 未见片段（S2 134 s）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN="${LAB20_RUN:-}"
SEEDS="$(echo "${LAB20_SEEDS:-101 102 103 104 105 106 107 108 109 110}")"
POLICIES="$(echo "${LAB20_POLICIES:-m4 dr}")"
WINDOWS="$(echo "${LAB20_WINDOWS:-train heldout unseen}")"
FORCE="${LAB20_FORCE:-0}"

die() { echo "[ERR] $*" >&2; exit 1; }
say() { echo "[lab20] $*"; }

if [ -z "$RUN" ]; then
  RUN="$(ls -d runs/*-lab20-dr 2>/dev/null | sort | tail -1 || true)"
fi
[ -n "$RUN" ] && [ -d "$RUN" ] || die "找不到 LAB20 run 目录（用 LAB20_RUN=runs/... 指定）"
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

dr_ckpt() {
  local ck
  ck="$(ls "$RUN"/logs/g1_tracking/*/model_*.pt 2>/dev/null | sort -V | tail -1 || true)"
  [ -n "$ck" ] || die "找不到 LAB20 ckpt（$RUN/logs/g1_tracking/*/model_*.pt）；训练是否跑完？"
  echo "$ck"
}

motion_file() {
  case "$1" in
    train)   echo "data/processed/fight1_s3_000s_20s.npz" ;;
    heldout) echo "data/processed/fight1_s3_120s_20s.npz" ;;
    unseen)  echo "data/processed/fight1_s2_134s_20s.npz" ;;
    *) die "未知窗口 $1（只能是 train / heldout / unseen）" ;;
  esac
}

run_stage() {
  local m4 dr ck mo
  m4="$(m4_ckpt)"; dr="$(dr_ckpt)"
  say "M4  ckpt: $m4（sha $(sha256sum "$m4" | cut -d' ' -f1)）"
  say "DR  ckpt: $dr（sha $(sha256sum "$dr" | cut -d' ' -f1)）"
  say "种子: $SEEDS"

  local up0 up1
  up0=$(cut -d' ' -f1 /proc/uptime)
  for p in $POLICIES; do
    case "$p" in m4) ck="$m4" ;; dr) ck="$dr" ;; *) die "未知策略 $p" ;; esac
    for w in $WINDOWS; do
      mo="$(motion_file "$w")"
      [ -f "$mo" ] || die "缺数据 $mo"
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
  environment/.venv/bin/python scripts/lab20_report.py --run "$RUN" --title "LAB20 (DR-strengthened) vs M4" --figure
}

case "${1:-}" in
  run)    run_stage; report_stage ;;
  report) report_stage ;;
  *)      sed -n '2,9p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
