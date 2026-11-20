#!/usr/bin/env bash
# LAB15 边界两格加到 20 种子：−x/−y 的 60 与 80 N*s（力 300/400 N）各跑种子 101-120。
#
# 用法：
#   bash scripts/lab15_boundary20.sh run     # 4 格 x 20 种子 = 80 次评估，约 40-50 min
#   bash scripts/lab15_boundary20.sh report  # 只重算判分 + 统计表 + 图（不重跑评估）
#   bash scripts/lab15_boundary20.sh show    # 打印统计表
#
# 完全复用 LAB12 的评估与判分（同 ckpt / 动作 / 推力窗口 / 判据），
# 只把种子从 5 个加到 20 个，并额外给每格倒地率的 95% 置信区间与两向差异检验。
#
# 覆盖变量：LAB15_RUN_DIR / LAB15_TAG / LAB15_CELLS / LAB15_SEEDS
#   LAB15_CELLS / LAB15_SEEDS 只用于卡住时试一格，正式 run 不要带。
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PY=environment/.venv/bin/python
TAG=${LAB15_TAG:-lab15-boundary20}
CELLS=${LAB15_CELLS:-nx_j60 nx_j80 ny_j60 ny_j80}
SEEDS=${LAB15_SEEDS:-101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120}
FIG=${LAB15_FIG:-outputs/figures/lab15_boundary20.png}

say() { printf "\n== %s\n" "$*"; }

new_run_dir() {
  local stamp n
  stamp=$(date +%Y%m%d)
  n=1
  while [[ -d runs/$(printf "%s-%03d-%s" "$stamp" "$n" "$TAG") ]]; do n=$((n+1)); done
  printf "runs/%s-%03d-%s" "$stamp" "$n" "$TAG"
}

do_run() {
  local run ncell nseed total
  read -r -a _c <<< "$CELLS"; ncell=${#_c[@]}
  read -r -a _s <<< "$SEEDS"; nseed=${#_s[@]}
  total=$(( ncell * nseed ))
  run=${LAB15_RUN_DIR:-$(new_run_dir)}
  say "run 目录：$run"
  say "规模：$ncell 格 x $nseed 种子 = $total 次评估"
  say "预计：约 $(( total * 35 / 60 )) min（单次 30-40 s）"
  say "注意：这期间不要让别的任务占 GPU；同一种子重跑判定应完全复现（D25），不复现先停下查环境"
  LAB12_TAG="$TAG" LAB12_CELLS="$CELLS" LAB12_SEEDS="$SEEDS" LAB12_RUN_DIR="$run" \
    bash scripts/lab12_recovery.sh run
  say "统计（倒地率 + 95% 置信区间 + 两向检验）"
  "$PY" scripts/lab15_report.py --dir "$run" --figure "$FIG"
  say "完成：$run/boundary20.md"
}

do_report() {
  local run
  run=${LAB15_RUN_DIR:-$(ls -dt "runs/"*"-$TAG" 2>/dev/null | head -1)}
  [[ -n $run && -d $run ]] || { echo "找不到 run 目录（标签 $TAG；可用 LAB15_RUN_DIR 指定）" >&2; exit 1; }
  "$PY" scripts/lab15_report.py --dir "$run" --figure "$FIG"
}

do_show() {
  local run
  run=${LAB15_RUN_DIR:-$(ls -dt "runs/"*"-$TAG" 2>/dev/null | head -1)}
  [[ -n $run && -f "$run/boundary20.md" ]] || { echo "还没有统计（先跑 run 或 report）" >&2; exit 1; }
  sed -n "1,80p" "$run/boundary20.md"
}

case ${1:-} in
  run) do_run ;;
  report) do_report ;;
  show) do_show ;;
  *) sed -n "2,16p" "$0"; exit 1 ;;
esac
