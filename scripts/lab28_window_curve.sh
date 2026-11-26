#!/usr/bin/env bash
# LAB28 —— 窗口距离曲线：同一段动作（dance1）上，距训练窗口越远，完成度掉多少？
#
# 零训练，只评估：2 策略（M4 单片段 / LAB27 双片段）× 6 个 20 s 窗口 × 10 种子 = 120 次 ≈75 min。
#
# 用法（项目根目录）:
#   LAB28_RUN=runs/20261126-001-lab28-window-curve bash scripts/lab28_window_curve.sh run     # 建窗口 + 评估 + 出表
#   bash scripts/lab28_window_curve.sh build        # 只做地面校正（6 个窗口，2 个已存在会跳过）
#   bash scripts/lab28_window_curve.sh report       # 只出表出图
#
# 窗口：dance1 的 0/20/40/60/80/100 s 各 20 s（= data/processed/dance1_SSSs_20s_gfix.npz）
# 策略：m4    = M4（只训练过 fight1 S3 0–20 s，dance1 完全没见过）
#       lab27 = LAB27（训练过 fight1 S3 0–20 s + dance1 0–20 s）
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN="$LAB28_RUN"
SEEDS="$(echo $LAB28_SEEDS)"
POLICIES="$(echo $LAB28_POLICIES)"
STARTS="$(echo $LAB28_STARTS)"
FORCE="$LAB28_FORCE"
L27_RUN="$LAB28_L27_RUN"

[ -n "$RUN" ] || RUN="$(ls -d runs/*-lab28-window-curve 2>/dev/null | sort | tail -1 || true)"
[ -n "$RUN" ] || RUN="runs/20261126-001-lab28-window-curve"
[ -n "$SEEDS" ] || SEEDS="101 102 103 104 105 106 107 108 109 110"
[ -n "$POLICIES" ] || POLICIES="m4 lab27"
[ -n "$STARTS" ] || STARTS="0 20 40 60 80 100"
[ -n "$FORCE" ] || FORCE=0
[ -n "$L27_RUN" ] || L27_RUN="runs/20261125-004-lab27-multiclip"

SRC="data/processed/lafan_dance1_subject1.npz"
PY="environment/.venv/bin/python"
L27_SHA="8745f96e2136e9187b8c2441812d72d95de125f846b147af14c76d680125dac"

say() { echo "[lab28] $*"; }
die() { echo "[ERR] $*" >&2; exit 1; }

[ "$(basename "$RUN")" != "runs" ] || die "RUN 不能是 runs 本身（会污染 runs 根目录）"

clip_of() { printf 'data/processed/dance1_%03ds_20s_gfix.npz' "$1"; }
win_of()  { printf 'w%03d' "$1"; }

build_stage() {
  [ -f "$SRC" ] || die "缺源数据 $SRC"
  say "地面校正（D41 闸门：max <= 20 mm 且 p99 <= 20 mm）"
  for s in $STARTS; do
    local out
    out="$(clip_of "$s")"
    if [ -f "$out" ]; then
      say "已有 $out（sha $(sha256sum "$out" | cut -d' ' -f1 | cut -c1-8)），跳过"
      continue
    fi
    say "生成 dance1 $s–$((s+20)) s → $out"
    "$PY" scripts/lab27_ground_fix.py --src "$SRC" --start-s "$s" --dur-s 20 --out "$out" \
      || die "地面校正失败或闸门 FAIL：$out（见上面的 [gate] 行）"
  done
}

m4_ckpt() {
  local run ck
  run="$(ls -d runs/*-m4-w15-finetune 2>/dev/null | sort | tail -1 || true)"
  [ -n "$run" ] || die "找不到 M4 训练 run（runs/*-m4-w15-finetune）"
  ck="$(ls "$run"/logs/g1_tracking/*/model_52998.pt 2>/dev/null | head -1 || true)"
  [ -n "$ck" ] || die "找不到 M4 ckpt（$run/logs/g1_tracking/*/model_52998.pt）"
  echo "$ck"
}

lab27_ckpt() {
  local ck sha
  ck="$(ls "$L27_RUN"/logs/g1_tracking/*/model_52998.pt 2>/dev/null | grep -v "/pretrained/" | head -1 || true)"
  [ -n "$ck" ] || die "找不到 LAB27 ckpt（$L27_RUN/logs/g1_tracking/*/model_52998.pt）"
  sha="$(sha256sum "$ck" | cut -d' ' -f1)"
  [ "$sha" = "$L27_SHA" ] || die "LAB27 ckpt sha 不符：$sha（期望 $L27_SHA）"
  echo "$ck"
}

run_stage() {
  build_stage
  local m4 l27 ck mo out up0 up1
  m4="$(m4_ckpt)"
  l27="$(lab27_ckpt)"
  say "M4    ckpt: $m4（sha $(sha256sum "$m4" | cut -d' ' -f1 | cut -c1-8)…）"
  say "LAB27 ckpt: $l27（sha $(sha256sum "$l27" | cut -d' ' -f1 | cut -c1-8)…）"
  say "run: $RUN；种子: $SEEDS；窗口起点: $STARTS"
  up0=$(cut -d' ' -f1 /proc/uptime)
  for p in $POLICIES; do
    case "$p" in
      m4)    ck="$m4" ;;
      lab27) ck="$l27" ;;
      *) die "未知策略 $p（只能是 m4 / lab27）" ;;
    esac
    for s in $STARTS; do
      mo="$(clip_of "$s")"
      [ -f "$mo" ] || die "缺窗口数据 $mo"
      for seed in $SEEDS; do
        out="$RUN/eval/$p/$(win_of "$s")/seed$seed"
        if [ -f "$out/summary.json" ] && [ "$FORCE" != "1" ]; then
          say "skip $p $(win_of "$s") seed$seed（已有 summary.json）"
          continue
        fi
        mkdir -p "$out"
        say "$p $(win_of "$s") seed$seed"
        MUJOCO_GL=egl "$PY" scripts/eval_tracking.py \
          --checkpoint-file "$ck" --motion-file "$mo" --seed "$seed" --out "$out" \
          > "$RUN/eval/$p/$(win_of "$s")/seed$seed.log" 2>&1 || echo "[ERR] $p $(win_of "$s") seed$seed 失败"
      done
    done
  done
  up1=$(cut -d' ' -f1 /proc/uptime)
  say "评估墙钟（/proc/uptime）：$(awk -v a="$up0" -v b="$up1" 'BEGIN{printf "%.0f", b-a}') s"
}

report_stage() {
  "$PY" scripts/lab28_report.py --run "$RUN" --starts "$STARTS" --figure \
    --title "LAB28 window distance curve (dance1, 20 seeds x 10)"
}

case "$1" in
  build)  build_stage ;;
  run)    run_stage; report_stage ;;
  report) report_stage ;;
  show)   cat "$RUN/lab28_curve.md" 2>/dev/null || sed -n '2,16p' "$0" ;;
  *)      sed -n '2,16p' "$0"; exit 1 ;;
esac
