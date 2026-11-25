#!/usr/bin/env bash
# LAB21 —— 世界系根位置/速度观测策略 vs M4 基线：3 个窗口 × N 种子
#
# 用法（项目根目录）:
#   LAB21_RUN=runs/20261125-003-lab21-world-obs bash scripts/lab21_eval.sh run     # 30 次 ≈ 20 min
#   LAB21_RUN=runs/20261125-003-lab21-world-obs bash scripts/lab21_eval.sh report  # 只重算表与图
#
# 口径与 LAB20 / LAB09 完全一致：scripts/eval_tracking.py，终止检测开、startup DR 关、跑满动作帧数。
# 三个窗口：train = 训练片段（S3 0 s）/ heldout = 同片段未见窗口（S3 120 s）/ unseen = 未见片段（S2 134 s）。
# 只有 LAB21 的 ckpt 需要 --obs-world-root；M4 基线不传该开关（默认策略 = obs，m4 仅用于复核对齐）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN="$(echo "${LAB21_RUN:-}")"
SEEDS="$(echo "${LAB21_SEEDS:-101 102 103 104 105 106 107 108 109 110}")"
POLICIES="$(echo "${LAB21_POLICIES:-obs}")"
WINDOWS="$(echo "${LAB21_WINDOWS:-train heldout unseen}")"
FORCE="${LAB21_FORCE:-0}"
BASELINE="$(echo "${LAB21_BASELINE:-runs/20261121-003-lab20-dr/eval/m4}")"

die() { echo "[ERR] $*" >&2; exit 1; }
say() { echo "[lab21] $*"; }

if [ -z "$RUN" ]; then
  RUN="$(ls -d runs/*-lab21-world-obs 2>/dev/null | sort | tail -1 || true)"
fi
[ -n "$RUN" ] && [ -d "$RUN" ] || die "找不到 LAB21 run 目录（用 LAB21_RUN=runs/... 指定）"
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

obs_ckpt() {
  local ck
  # 必须排除 pretrained/（那是热启动权重副本，不是训练产物）
  ck="$(ls "$RUN"/logs/g1_tracking/*/model_*.pt 2>/dev/null | grep -v "/pretrained/" | sort -V | tail -1 || true)"
  [ -n "$ck" ] || die "找不到 LAB21 ckpt（$RUN/logs/g1_tracking/*/model_*.pt）；训练是否跑完？"
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
  local ck_obs ck_m4 mo extra ck
  ck_obs="$(obs_ckpt)"; ck_m4="$(m4_ckpt)"
  say "LAB21 ckpt: $ck_obs（sha $(sha256sum "$ck_obs" | cut -d' ' -f1)）"
  say "M4    ckpt: $ck_m4（sha $(sha256sum "$ck_m4" | cut -d' ' -f1)）"
  say "基线（M4 已算好的 3 窗口 × 10 种子）：$BASELINE"
  say "种子: $SEEDS"

  local up0 up1
  up0=$(cut -d' ' -f1 /proc/uptime)
  for p in $POLICIES; do
    for w in $WINDOWS; do
      mo="$(motion_file "$w")"
      [ -f "$mo" ] || die "缺数据 $mo"
      for s in $SEEDS; do
        local out="$RUN/eval/$p/$w/seed${s}"
        case "$p" in
          obs) ck="$ck_obs"; extra="--obs-world-root" ;;
          m4)  ck="$ck_m4";  extra="" ;;
          *) die "未知策略 $p" ;;
        esac
        if [ -f "$out/summary.json" ] && [ "$FORCE" != "1" ]; then
          say "skip $p/$w/seed$s（已有 summary.json）"
          continue
        fi
        mkdir -p "$out"
        say "$p $w seed $s"
        MUJOCO_GL=egl environment/.venv/bin/python scripts/eval_tracking.py \
          --checkpoint-file "$ck" --motion-file "$mo" --seed "$s" --out "$out" $extra \
          > "$RUN/eval/$p/$w/seed${s}.log" 2>&1 || echo "[ERR] $p/$w/seed$s 失败"
      done
    done
  done
  up1=$(cut -d' ' -f1 /proc/uptime)
  say "评估墙钟（/proc/uptime）：$(awk -v a="$up0" -v b="$up1" 'BEGIN{printf "%.0f", b-a}') s"
}

report_stage() {
  environment/.venv/bin/python scripts/lab21_report.py --run "$RUN" --baseline "$BASELINE" \
    --title "LAB21: world-frame root pos/vel observation vs M4" --figure
}

case "${1:-}" in
  run)    run_stage; report_stage ;;
  report) report_stage ;;
  *)      sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
