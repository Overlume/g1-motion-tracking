#!/usr/bin/env bash
# LAB16 边界格的判定翻转率：nx_j60 / ny_j60 各 20 个种子，再跑 2 遍，
# 与 LAB15 的第 1 遍合成 3 遍，统计"20 个种子里有几个的分类会翻"。
#
# 用法：
#   bash scripts/lab16_fliprate.sh run     # 2 格 x 20 种子 x 2 遍 = 80 次评估，约 40-50 min
#   bash scripts/lab16_fliprate.sh report  # 只重算判分 + 翻转率表 + 图（不重跑评估）
#   bash scripts/lab16_fliprate.sh show    # 打印翻转率表
#
# 复用 LAB12 的评估与判分（同 ckpt / 动作 / 推力窗口 / 判据），
# 新跑的每遍写进 <run>/rep<N>/；第 1 遍直接引用 LAB15 的 20 种子结果（LAB16_REP1_DIR）。
#
# 覆盖变量：LAB16_RUN_DIR / LAB16_TAG / LAB16_CELLS / LAB16_SEEDS / LAB16_REPS / LAB16_REP1_DIR
#   LAB16_CELLS / LAB16_SEEDS 只用于卡住时试一格，正式 run 不要带。
set -euo pipefail

ROOT=${BASH_SOURCE[0]}
ROOT=$(cd "$(dirname "$ROOT")/.." && pwd)
cd "$ROOT"

PY=environment/.venv/bin/python
TAG=${LAB16_TAG:-lab16-fliprate}
CELLS=${LAB16_CELLS:-nx_j60 ny_j60}
SEEDS=${LAB16_SEEDS:-101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120}
REPS=${LAB16_REPS:-3}
BASE=${LAB16_REP1_DIR:-runs/20261120-001-lab15-boundary20}
FIG=${LAB16_FIG:-outputs/figures/lab16_flipmatrix.png}

say() { printf "\n== %s\n" "$*"; }
die() { printf "ERROR: %s\n" "$*" >&2; exit 1; }

new_run_dir() {
  local stamp n
  stamp=$(date +%Y%m%d)
  n=1
  while [[ -d runs/$(printf "%s-%03d-%s" "$stamp" "$n" "$TAG") ]]; do n=$((n+1)); done
  printf "runs/%s-%03d-%s" "$stamp" "$n" "$TAG"
}

rep_args() {
  local run=$1 rep
  REP_ARGS=("--rep" "rep1=$BASE")
  for ((rep=2; rep<=REPS; rep++)); do
    if [[ -d $run/rep$rep ]]; then REP_ARGS+=("--rep" "rep$rep=$run/rep$rep"); fi
  done
}

judge_reps() {
  local run=$1 rep
  for ((rep=2; rep<=REPS; rep++)); do
    if [[ -d $run/rep$rep ]]; then "$PY" scripts/lab12_recovery_report.py --dir "$run/rep$rep"; fi
  done
}

do_run() {
  local run rep ncell nseed total new_reps
  read -r -a _c <<< "$CELLS"; ncell=${#_c[@]}
  read -r -a _s <<< "$SEEDS"; nseed=${#_s[@]}
  [[ -f $BASE/recovery.json ]] || die "第 1 遍缺 $BASE/recovery.json（先跑 LAB15）"
  new_reps=$(( REPS - 1 ))
  total=$(( ncell * nseed * new_reps ))

  if [[ -n "${LAB16_RUN_DIR:-}" ]]; then run="${LAB16_RUN_DIR}"; else run=$(new_run_dir); fi
  mkdir -p "$run"
  printf 'tag: %s\ncells: %s\nseeds: %s\nreps: %s（rep1 引用 %s，本次新跑 rep2..rep%s）\ngrid: <run>/rep<N>/<cell>/seed<S>\nnote: 同一组种子重复；每遍是新进程；判分与 LAB12 完全一致\n' \
    "$TAG" "$CELLS" "$SEEDS" "$REPS" "$BASE" "$REPS" > "$run/params.txt"

  say "run 目录：$run"
  say "第 1 遍：$BASE（LAB15，不再重跑）"
  say "本次：$ncell 格 x $nseed 种子 x $new_reps 遍 = $total 次评估，约 $(( total * 35 / 60 )) min"
  say "注意：这期间不要让别的任务占 GPU；判定层的抖动不会消失，这正是要量的东西"

  for ((rep=2; rep<=REPS; rep++)); do
    say "重复 $rep/$REPS -> $run/rep$rep"
    LAB12_TAG="$TAG" LAB12_CELLS="$CELLS" LAB12_SEEDS="$SEEDS" \
      LAB12_RUN_DIR="$run/rep$rep" bash scripts/lab12_recovery.sh run
  done

  say "判分"
  judge_reps "$run"
  say "翻转率统计"
  rep_args "$run"
  "$PY" scripts/lab16_report.py --dir "$run" "${REP_ARGS[@]}" --cells "${CELLS// /,}" --figure "$FIG"
  say "完成：$run/fliprate.md"
}

do_report() {
  local run
  run=${LAB16_RUN_DIR:-$(ls -dt "runs/"*"-$TAG" 2>/dev/null | head -1)}
  [[ -n $run && -d $run ]] || die "找不到 run 目录（标签 $TAG；可用 LAB16_RUN_DIR 指定）"
  judge_reps "$run"
  rep_args "$run"
  "$PY" scripts/lab16_report.py --dir "$run" "${REP_ARGS[@]}" --cells "${CELLS// /,}" --figure "$FIG"
}

do_show() {
  local run
  run=${LAB16_RUN_DIR:-$(ls -dt "runs/"*"-$TAG" 2>/dev/null | head -1)}
  [[ -n $run && -f "$run/fliprate.md" ]] || die "还没有统计（先跑 run 或 report）"
  sed -n "1,70p" "$run/fliprate.md"
}

case ${1:-} in
  run) do_run ;;
  report) do_report ;;
  show) do_show ;;
  *) sed -n "2,16p" "$0"; exit 1 ;;
esac
