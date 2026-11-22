#!/usr/bin/env bash
# LAB20b：把 LAB12 的扰动协议搬到 LAB20（DR 加强）的 checkpoint 上，量「抗推能力」有没有变好。
#
# 目的：LAB20 的 held-out 完成度归零（D34），DR 的预期收益若存在，应体现在受控外力下
#       的稳定性上。这里跑与 LAB15/LAB16 完全相同的协议（同动作、同推力窗口、同判据），
#       只换 ckpt，就能直接和 M4 的 5-6/20、18-19/20 对比。
#
# 用法：
#   bash scripts/lab20b_perturb.sh run     # 2 格 x 20 种子 x 3 遍 = 120 次评估，约 70-90 min
#   bash scripts/lab20b_perturb.sh report  # 只重算判分 + 翻转率 + 与 M4 的对比表（不重跑评估）
#   bash scripts/lab20b_perturb.sh show    # 打印翻转率表与对比表
#
# 覆盖变量：LAB20B_RUN_DIR / LAB20B_TAG / LAB20B_CKPT / LAB20B_CELLS / LAB20B_SEEDS / LAB20B_REPS / LAB20B_FIG
#   LAB20B_CELLS / LAB20B_SEEDS 只用于卡住时试一格，正式 run 不要带。
set -euo pipefail

ROOT=${BASH_SOURCE[0]}
ROOT=$(cd "$(dirname "$ROOT")/.." && pwd)
cd "$ROOT"

PY=environment/.venv/bin/python
TAG=${LAB20B_TAG:-lab20b-perturb}
CKPT=${LAB20B_CKPT:-runs/20261121-003-lab20-dr/logs/g1_tracking/2026-11-21_19-27-09_lab20-dr/model_52998.pt}
EXPECT_SHA=${LAB20B_SHA:-e3747012f3c7d35f9b1de4720a8420b6c1fb7f53c3c8e00c3ed2afcd2625b287}
CELLS=${LAB20B_CELLS:-nx_j60 ny_j60}
SEEDS=${LAB20B_SEEDS:-101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120}
REPS=${LAB20B_REPS:-3}
FIG=${LAB20B_FIG:-outputs/figures/lab20b_robust.png}
BASE=${LAB20B_BASELINE:-runs/20261120-001-lab15-boundary20}

say() { printf "\n== %s\n" "$*"; }
die() { printf "ERROR: %s\n" "$*" >&2; exit 1; }

new_run_dir() {
  local stamp n
  stamp=$(date +%Y%m%d)
  n=1
  while [[ -d runs/$(printf "%s-%03d-%s" "$stamp" "$n" "$TAG") ]]; do n=$((n+1)); done
  printf "runs/%s-%03d-%s" "$stamp" "$n" "$TAG"
}

judge_reps() {
  local run=$1 rep
  for ((rep=1; rep<=REPS; rep++)); do
    if [[ -d $run/rep$rep ]]; then "$PY" scripts/lab12_recovery_report.py --dir "$run/rep$rep"; fi
  done
}

rep_args() {
  local run=$1 rep
  REP_ARGS=()
  for ((rep=1; rep<=REPS; rep++)); do
    if [[ -d $run/rep$rep ]]; then REP_ARGS+=("--rep" "rep$rep=$run/rep$rep"); fi
  done
}

do_report() {
  local run=$1
  judge_reps "$run"
  rep_args "$run"
  [[ ${#REP_ARGS[@]} -gt 0 ]] || die "$run 下没有 rep<N> 目录（先跑 run）"
  "$PY" scripts/lab16_report.py --dir "$run" "${REP_ARGS[@]}" \
    --cells "${CELLS// /,}" --figure "$FIG" \
    --title "LAB20b：加强 DR 的 ckpt 在 60 N·s 两向的抗推能力（对照 M4）" \
    --figure-title "LAB20b (harder-DR ckpt, 60 N s, 20 seeds x $REPS reps)" \
    --intro "同一套 LAB12 协议（同动作、8.0 s 起推 0.2 s、关 termination、同一批种子 101-120）换到 LAB20 的 DR 加强 ckpt 上，看 60 N·s 两向的倒地数是否低于 M4 的 5-6/20 与 18-19/20（LAB15/LAB16），并用三遍量出翻转率（边界格 >=3 遍，D28/D30）。"
  "$PY" scripts/lab20b_compare.py --run "$run" --baseline "$BASE"
}

do_run() {
  local run rep total
  read -r -a _c <<< "$CELLS"; local ncell=${#_c[@]}
  read -r -a _s <<< "$SEEDS"; local nseed=${#_s[@]}
  total=$(( ncell * nseed * REPS ))

  [[ -f $CKPT ]] || die "找不到 checkpoint：$CKPT"
  local got; got=$(sha256sum "$CKPT" | cut -d" " -f1)
  [[ $got == "$EXPECT_SHA" ]] || die "ckpt sha 不符：$CKPT = $got，期望 $EXPECT_SHA（别拿 pretrained/ 里的热启动副本）"

  if [[ -n "${LAB20B_RUN_DIR:-}" ]]; then run="${LAB20B_RUN_DIR}"; else run=$(new_run_dir); fi
  case $run in
    runs|runs/|.|./) die "LAB20B_RUN_DIR=$run 会污染 runs 根目录：先建 runs/<YYYYMMDD>-NNN-<tag>/（如 runs/20261122-001-lab20b-perturb）" ;;
  esac
  if [[ -n "${LAB20B_RUN_DIR:-}" && ! -d $run ]]; then die "$run 不存在：先 mkdir -p $run（本脚本不替你建外层目录）"; fi

  mkdir -p "$run"
  printf 'tag: %s\nckpt: %s\nsha256: %s\nmotion: data/processed/fight1_s3_000s_20s.npz\ncells: %s\nseeds: %s\nreps: %s\npush: 8.0 s +0.2 s\ntermination: OFF\nbaseline: %s（M4, LAB15 第 1 遍）\ngrid: <run>/rep<N>/<cell>/seed<S>\nnote: 与 LAB15/LAB16 协议相同，只换 ckpt\n' \
    "$TAG" "$CKPT" "$got" "$CELLS" "$SEEDS" "$REPS" "$BASE" > "$run/params.txt"

  say "run 目录：$run"
  say "ckpt：$CKPT（sha ${got:0:8}…${got: -6}）"
  say "规模：$ncell 格 x $nseed 种子 x $REPS 遍 = $total 次评估，约 $(( total * 35 / 60 ))-$(( total * 45 / 60 )) min"
  say "对照：$BASE（M4 的 20 种子单遍）"

  for ((rep=1; rep<=REPS; rep++)); do
    say "第 $rep/$REPS 遍 -> $run/rep$rep"
    LAB12_CKPT="$CKPT" LAB12_TAG="$TAG" LAB12_CELLS="$CELLS" LAB12_SEEDS="$SEEDS" \
      LAB12_RUN_DIR="$run/rep$rep" bash scripts/lab12_recovery.sh run
  done

  say "统计"
  do_report "$run"
  say "完成：$run/fliprate.md 与 $run/vs_m4.md"
}

cmd=${1:-}
case $cmd in
  run) do_run ;;
  report)
    run=${LAB20B_RUN_DIR:-$(ls -dt "runs/"*"-$TAG" 2>/dev/null | head -1)}
    [[ -n $run && -d $run ]] || die "找不到 run 目录（标签 $TAG；可用 LAB20B_RUN_DIR 指定）"
    do_report "$run"
    ;;
  show)
    run=${LAB20B_RUN_DIR:-$(ls -dt "runs/"*"-$TAG" 2>/dev/null | head -1)}
    [[ -n $run && -f "$run/fliprate.md" ]] || die "还没有统计（先跑 run 或 report）"
    sed -n '1,80p' "$run/fliprate.md"
    echo
    cat "$run/vs_m4.md"
    ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
