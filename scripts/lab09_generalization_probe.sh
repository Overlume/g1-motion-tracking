#!/usr/bin/env bash
# LAB09 —— 泛化探针正式化：2 策略 × 3 片段 × N 种子
#
# 用法（项目根目录）:
#   bash scripts/lab09_generalization_probe.sh run              # 默认 5 种子（101-105），约 20 min
#   bash scripts/lab09_generalization_probe.sh report
#
# 目的：区分 "在训练片段上是专家" 与 "会跟踪参考运动"。
# 三条片段：train = 参与训练；heldout = 同片段未见窗口；unseen_s2 = 未见片段。
# 评估口径与 M4 一致：scripts/eval_tracking.py，终止检测开、启动 DR 关、999 步。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUNS_BASE="$(echo "${LAB09_RUNS_DIR:-runs}")"
SEEDS="$(echo "${LAB09_SEEDS:-101 102 103 104 105}")"
COMBOS_FILTER="$(echo "${LAB09_COMBOS:-}")"

OFFICIAL="scratch/model_49999.pt"
TRAIN_CLIP="data/processed/fight1_s3_000s_20s.npz"
HELDOUT_CLIP="data/processed/fight1_s3_120s_20s.npz"
UNSEEN_CLIP="data/processed/fight1_s2_134s_20s.npz"

die() { echo "[ERR] $*" >&2; exit 1; }
say() { echo "[lab09] $*"; }

next_run_dir() {
  local tag="$1" day idx
  day=$(date +%Y%m%d); idx=1
  while compgen -G "$RUNS_BASE/${day}-$(printf "%03d" "$idx")-*" >/dev/null; do idx=$((idx + 1)); done
  echo "$RUNS_BASE/${day}-$(printf "%03d" "$idx")-${tag}"
}
newest_run() { ls -d "$RUNS_BASE"/*-"${1}" 2>/dev/null | sort | tail -1; }

m4_ckpt() {
  local run
  run="$(ls -d runs/*-m4-w15-finetune 2>/dev/null | sort | tail -1 || true)"
  [ -n "$run" ] || die "找不到 M4 训练 run（runs/*-m4-w15-finetune）"
  local ck
  ck=$(ls "$run"/logs/g1_tracking/*/model_52998.pt 2>/dev/null | head -1 || true)
  [ -n "$ck" ] || ck=$(ls "$run"/logs/g1_tracking/*/model_*.pt | sort -V | tail -1)
  [ -n "$ck" ] || die "找不到 ckpt（$run）"
  echo "$ck"
}

run_stage() {
  for f in "$TRAIN_CLIP" "$HELDOUT_CLIP" "$UNSEEN_CLIP"; do
    [ -f "$f" ] || die "缺数据 $f"
  done
  local m4; m4="$(m4_ckpt)"
  say "官方 ckpt: $OFFICIAL"
  say "M4 ckpt : $m4"

  RUN="$(next_run_dir lab09-generalization)"; mkdir -p "$RUN"
  {
    echo "# LAB09 泛化探针"
    echo ""
    echo "- 官方 ckpt：${OFFICIAL}（sha $(sha256sum "$OFFICIAL" | cut -d' ' -f1)）"
    echo "- M4 ckpt：${m4}（sha $(sha256sum "$m4" | cut -d' ' -f1)）"
    echo "- 种子：${SEEDS}"
    echo "- 片段：train=${TRAIN_CLIP} / heldout=${HELDOUT_CLIP} / unseen_s2=${UNSEEN_CLIP}"
    echo "- 口径：scripts/eval_tracking.py，终止检测开、启动 DR 关、999 步"
    echo "- 起始：$(date -Is)（D14：墙钟不可跨会话比，等 /proc/uptime）"
  } > "$RUN/params.txt"

  local combos=(official_train official_heldout official_unseen m4_train m4_heldout m4_unseen)
  local up0 up1
  up0=$(cut -d' ' -f1 /proc/uptime)
  for combo in "${combos[@]}"; do
    if [ -n "$COMBOS_FILTER" ] && ! echo " $COMBOS_FILTER " | grep -q " $combo "; then continue; fi
    local policy="${combo%%_*}" clip="${combo##*_}" ck mo
    case "$policy" in
      official) ck="$OFFICIAL" ;;
      m4) ck="$m4" ;;
      *) die "未知策略 ${policy}" ;;
    esac
    case "$clip" in
      train) mo="$TRAIN_CLIP" ;;
      heldout) mo="$HELDOUT_CLIP" ;;
      unseen) mo="$UNSEEN_CLIP" ;;
      *) die "未知片段 ${clip}" ;;
    esac
    for s in $SEEDS; do
      local out="$RUN/${combo}/seed${s}"
      mkdir -p "$out"
      say "${combo} seed ${s}"
      MUJOCO_GL=egl environment/.venv/bin/python scripts/eval_tracking.py \
        --checkpoint-file "$ck" --motion-file "$mo" --seed "$s" --out "$out" \
        > "$RUN/${combo}/seed${s}.log" 2>&1 || echo "[ERR] ${combo} seed ${s} 失败"
    done
  done
  up1=$(cut -d' ' -f1 /proc/uptime)
  echo "- 真实墙钟（/proc/uptime）：$(awk -v a="$up0" -v b="$up1" 'BEGIN{printf "%.0f", b-a}') s" >> "$RUN/params.txt"
  say "run dir: $RUN"
  LAB09_RUN="$RUN"
}

report_stage() {
  RUN="$(newest_run lab09-generalization)"
  [ -n "$RUN" ] || die "找不到 LAB09 run（$RUNS_BASE/*-lab09-generalization）"
  environment/.venv/bin/python scripts/lab09_report.py --run "$RUN"
  say "report: $RUN/generalization.md"
}

case "${1:-}" in
  run)
    run_stage
    ;;
  report)
    report_stage
    ;;
  *)
    sed -n '2,9p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
