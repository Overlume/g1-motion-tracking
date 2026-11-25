#!/usr/bin/env bash
# LAB26：−y 的 50/55 N·s（f30 = 摩擦 0.2–3.0 的权重，零训练）——把 −y 曲线从
# 45/60 两个端点补成 45/50/55/60 四点，定位 3.0 档的 50% 交点。
#
# 用法：
#   LAB26_RUN=runs/20261125-001-lab26-y-f30-mid bash scripts/lab26_ymid.sh run
#   LAB26_RUN=runs/20261125-001-lab26-y-f30-mid bash scripts/lab26_ymid.sh status
#   LAB26_RUN=runs/20261125-001-lab26-y-f30-mid bash scripts/lab26_ymid.sh report
#   bash scripts/lab26_ymid.sh show
#
# 覆盖变量：LAB26_RUN（必须，run 目录）/ LAB26_CELLS / LAB26_SEEDS / LAB26_REPS /
#           LAB26_F30_DIR / LAB26_F30_SHA / LAB26_SKIP_SHA_CHECK
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PY=environment/.venv/bin/python
RUN=${LAB26_RUN:-}
CELLS=${LAB26_CELLS:-ny_j50 ny_j55}
SEEDS=${LAB26_SEEDS:-101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120}
REPS=${LAB26_REPS:-3}
F30_DIR=${LAB26_F30_DIR:-runs/20261123-001-lab24-fric-scan/arms/f30}
F30_SHA=${LAB26_F30_SHA:-1119f986fd1a33054eb584c9865dae7c499390f8f48a41229ef8c560b2455ba0}
SKIP_SHA=${LAB26_SKIP_SHA_CHECK:-0}

say() { printf '\n== %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

require_run() {
  [[ -n $RUN ]] || die "先设置 LAB26_RUN=runs/<YYYYMMDD>-NNN-lab26-y-f30-mid（不要写 runs 本身）"
  case $RUN in
    runs|runs/|./runs|./runs/|.|./)
      die "$RUN 会把产物摊在 runs 根目录；先建 runs/<YYYYMMDD>-NNN-lab26-y-f30-mid 再传进来" ;;
  esac
}

find_ckpt() {
  ls "$F30_DIR"/logs/g1_tracking/*/model_*.pt 2>/dev/null | grep -v "/pretrained/" | sort -V | tail -1
}

do_run() {
  require_run
  local ckpt sha rep
  ckpt=$(find_ckpt)
  [[ -n $ckpt && -f $ckpt ]] || die "找不到 f30 的 ckpt（$F30_DIR/logs/g1_tracking/*/model_*.pt）"
  sha=$(sha256sum "$ckpt" | cut -d' ' -f1)
  say "ckpt：$ckpt"
  say "sha256：$sha"
  if [[ $SKIP_SHA != 1 && $sha != "$F30_SHA" ]]; then
    die "ckpt sha 与 LAB24 f30 不符（期望 ${F30_SHA}）：很可能选到了 pretrained 热启动副本"
  fi
  mkdir -p "$RUN"
  for ((rep=1; rep<=REPS; rep++)); do
    say "rep$rep：$CELLS × $(( $(wc -w <<< "$SEEDS") )) 个种子"
    LAB12_CKPT=$ckpt \
    LAB12_RUN_DIR="$RUN/rep$rep" \
    LAB12_TAG=lab26-y-mid \
    LAB12_CELLS="$CELLS" \
    LAB12_SEEDS="$SEEDS" \
    LAB12_SKIP_EXISTING=1 \
      bash scripts/lab12_recovery.sh run
  done
  say "汇总"
  $PY scripts/lab26_report.py --run "$RUN" --figure
}

do_report() {
  require_run
  $PY scripts/lab26_report.py --run "$RUN" --figure
}

do_status() {
  require_run
  local rep n
  for ((rep=1; rep<=REPS; rep++)); do
    n=$(ls "$RUN"/rep$rep/seed*/summary.json 2>/dev/null | wc -l)
    printf 'rep%d: %s/%d 个种子完成\n' "$rep" "$n" "$(( $(wc -w <<< "$SEEDS") * $(wc -w <<< "$CELLS") ))"
  done
  [[ -f $RUN/lab26_ymid.md ]] && sed -n '1,30p' "$RUN/lab26_ymid.md" || true
}

do_show() {
  local d
  d=$(ls -dt runs/*-lab26-y-f30-mid 2>/dev/null | head -1)
  [[ -n $d && -f $d/lab26_ymid.md ]] || die "还没有报告（先 run）"
  sed -n '1,40p' "$d/lab26_ymid.md"
}

case ${1:-} in
  run) do_run ;;
  report) do_report ;;
  status) do_status ;;
  show) do_show ;;
  *) die "用法：bash scripts/lab26_ymid.sh {run|report|status|show}" ;;
esac
