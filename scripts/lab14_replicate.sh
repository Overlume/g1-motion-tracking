#!/usr/bin/env bash
# LAB14 重复实验：同一配置重跑 N 遍，测"同种子、同参数"的噪声底。
#
# 用法：
#   bash scripts/lab14_replicate.sh run     # 2 格 x 5 种子 x 3 遍 = 30 次评估，约 15 min
#   bash scripts/lab14_replicate.sh report  # 只重跑每遍判分 + 顶层汇总
#   bash scripts/lab14_replicate.sh show    # 打印顶层汇总
#
# 格：ny_j60（300 N 持续 0.2 s = 60 N*s）与 ny_j40（200 N 持续 0.2 s = 40 N*s）
# 完全复用 LAB12 的评估与判分（同 ckpt / 动作 / 推力窗口 / 判据），
# 每一遍写进 <run>/rep<N>/，顶层再写 replication.md/json。
#
# 覆盖变量：LAB14_RUN_DIR / LAB14_TAG / LAB14_CELLS / LAB14_SEEDS / LAB14_REPS
#   LAB14_CELLS / LAB14_SEEDS 只用于卡住时试一格，正式 run 不要带。
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PY=environment/.venv/bin/python
TAG=${LAB14_TAG:-lab14-replicate}
CELLS=${LAB14_CELLS:-ny_j60 ny_j40}
SEEDS=${LAB14_SEEDS:-101 102 103 104 105}
REPS=${LAB14_REPS:-3}

say() { printf '\n== %s\n' "$*"; }

new_run_dir() {
  local stamp n
  stamp=$(date +%Y%m%d)
  n=1
  while [[ -d runs/$(printf '%s-%03d-%s' "$stamp" "$n" "$TAG") ]]; do n=$((n+1)); done
  printf 'runs/%s-%03d-%s' "$stamp" "$n" "$TAG"
}

do_run() {
  local run rep
  run=${LAB14_RUN_DIR:-$(new_run_dir)}
  mkdir -p "$run"
  printf 'tag: %s
cells: %s
seeds: %s
reps: %s
grid: <run>/rep<N>/<cell>/seed<S>
note: 同一组种子重复；每遍是新进程
' \
    "$TAG" "$CELLS" "$SEEDS" "$REPS" > "$run/params.txt"
  say "run 目录：$run"
  for ((rep=1; rep<=REPS; rep++)); do
    say "重复 $rep/$REPS → $run/rep$rep"
    LAB12_TAG="$TAG" LAB12_CELLS="$CELLS" LAB12_SEEDS="$SEEDS" \
      LAB12_RUN_DIR="$run/rep$rep" bash scripts/lab12_recovery.sh run
  done
  say "汇总"
  "$PY" scripts/lab14_compare.py --dir "$run"
  say "完成：$run/replication.md"
}

do_report() {
  local run d
  run=${LAB14_RUN_DIR:-$(ls -dt "runs/"*"-"${TAG} 2>/dev/null | head -1)}
  [[ -n $run && -d $run ]] || { echo "找不到 run 目录（标签 ${TAG}；可用 LAB14_RUN_DIR 指定）" >&2; exit 1; }
  for d in "$run"/rep*; do
    [[ -d $d ]] && "$PY" scripts/lab12_recovery_report.py --dir "$d"
  done
  "$PY" scripts/lab14_compare.py --dir "$run"
}

do_show() {
  local run
  run=${LAB14_RUN_DIR:-$(ls -dt "runs/"*"-"${TAG} 2>/dev/null | head -1)}
  [[ -n $run && -f $run/replication.md ]] || { echo "还没有汇总（先跑 run）" >&2; exit 1; }
  sed -n '1,60p' "$run/replication.md"
}

case ${1:-} in
  run) do_run ;;
  report) do_report ;;
  show) do_show ;;
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac
