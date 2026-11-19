#!/usr/bin/env bash
# LAB12 恢复实验：对 M5 里"站着被判罚"的 6 个格，关闭 termination 跑推后 3 s，
# 看机器人是稳住、崩了又恢复、还是崩了没起来。
#
# 用法：
#   bash scripts/lab12_recovery.sh run     # 30 次评估（6 格 x 5 种子），约 20 min
#   bash scripts/lab12_recovery.sh report  # 只重跑判分
#   bash scripts/lab12_recovery.sh show    # 打印已有报告的汇总表
#
# 覆盖变量：LAB12_RUN_DIR / LAB12_TAG / LAB12_CELLS / LAB12_SEEDS / LAB12_CKPT / LAB12_MOTION /
#           LAB12_PUSH_S / LAB12_PUSH_DUR
#   LAB12_TAG 决定 run 目录后缀（默认 lab12-recovery；LAB13 用 lab13-mid-impulse）
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PY=environment/.venv/bin/python
EVAL=scripts/eval_tracking.py
REPORT=scripts/lab12_recovery_report.py

CKPT=${LAB12_CKPT:-runs/20261117-001-m4-w15-finetune/logs/g1_tracking/2026-11-17_09-25-17_m4-w15/model_52998.pt}
MOTION=${LAB12_MOTION:-data/processed/fight1_s3_000s_20s.npz}
CELLS=${LAB12_CELLS:-px_j120 nx_j60 nx_j120 py_j120 ny_j60 ny_j120}
SEEDS=${LAB12_SEEDS:-101 102 103 104 105}
TAG=${LAB12_TAG:-lab12-recovery}
PUSH_S=${LAB12_PUSH_S:-8.0}
PUSH_DUR=${LAB12_PUSH_DUR:-0.2}

say() { printf '\n== %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

force_of() {
  local cell=$1 dir j F
  dir=${cell%%_*}
  j=${cell##*_j}
  [[ $j =~ ^[0-9]+$ ]] || die "无法解析冲量: $cell"
  F=$(awk -v n="$j" 'BEGIN{printf "%d", n/0.2}')
  case $dir in
    px) printf '%d,0,0' "$F" ;;
    nx) printf -- '-%d,0,0' "$F" ;;
    py) printf '0,%d,0' "$F" ;;
    ny) printf '0,-%d,0' "$F" ;;
    *) die "未知方向: $dir" ;;
  esac
}

new_run_dir() {
  local stamp n
  stamp=$(date +%Y%m%d)
  n=1
  while [[ -d runs/$(printf '%s-%03d-%s' "$stamp" "$n" "$TAG") ]]; do n=$((n+1)); done
  printf 'runs/%s-%03d-%s' "$stamp" "$n" "$TAG"
}

write_meta() {
  local run=$1
  mkdir -p "$run"
  printf 'tag: %s\nckpt: %s\nsha256: %s\nmotion: %s\ncells: %s\nseeds: %s\npush: %s s +%s s\ntermination: OFF\n' \
    "$TAG" "$CKPT" "$(sha256sum "$CKPT" | cut -d' ' -f1)" "$MOTION" "$CELLS" "$SEEDS" "$PUSH_S" "$PUSH_DUR" > "$run/params.txt"
  {
    printf 'python: '; $PY -c 'import sys; print(sys.version.split()[0])'
    $PY -c 'import torch, mujoco; print("torch", torch.__version__); print("mujoco", mujoco.__version__)'
    printf 'git HEAD: '; git rev-parse --short HEAD
    printf 'git dirty: '; if [[ -n $(git status --porcelain) ]]; then echo yes; else echo no; fi
  } > "$run/env_versions.txt" 2>&1
}

do_run() {
  local run i total t0 t1
  [[ -f $CKPT ]] || die "找不到 checkpoint: $CKPT"
  [[ -f $MOTION ]] || die "找不到动作: $MOTION"

  if [[ -n ${LAB12_RUN_DIR:-} ]]; then run=$LAB12_RUN_DIR; else run=$(new_run_dir); fi
  write_meta "$run"

  say "run 目录：$run"
  say "checkpoint：$CKPT"

  local cells=() seeds=()
  read -r -a cells <<< "$CELLS"
  read -r -a seeds <<< "$SEEDS"
  total=$(( ${#cells[@]} * ${#seeds[@]} ))
  i=0

  for cell in "${cells[@]}"; do
    local vec; vec=$(force_of "$cell")
    for seed in "${seeds[@]}"; do
      i=$((i+1))
      local out="$run/$cell/seed$seed"
      say "[$i/$total] $cell seed $seed  force=$vec"
      mkdir -p "$out"
      t0=$(cut -d' ' -f1 /proc/uptime)
      # 负值参数必须用等号形式，否则 argparse 会当成选项
      MUJOCO_GL=egl $PY "$EVAL" \
        --checkpoint-file "$CKPT" \
        --motion-file "$MOTION" \
        --seed "$seed" \
        --max-steps 0 \
        --ignore-terminations \
        --perturb-body torso_link \
        --perturb-force="$vec" \
        --perturb-start-s "$PUSH_S" \
        --perturb-duration-s "$PUSH_DUR" \
        --out "$out" 2>&1 | tee "$out/eval.log" | tail -3
      t1=$(cut -d' ' -f1 /proc/uptime)
      awk -v a="$t0" -v b="$t1" -v c="$cell" -v s="$seed" 'BEGIN{printf "%s %s %.0f\n", c, s, b-a}' >> "$run/wallclock_s.txt"
    done
  done

  say "判分"
  $PY "$REPORT" --dir "$run"
  say "完成：$run/recovery.md"
}

do_report() {
  local run
  run=${LAB12_RUN_DIR:-$(ls -dt "runs/"*"-${TAG}" 2>/dev/null | head -1)}
  [[ -n $run && -d $run ]] || die "找不到 run 目录（标签 ${TAG}；可用 LAB12_RUN_DIR 指定）"
  $PY "$REPORT" --dir "$run"
}

do_show() {
  local run
  run=${LAB12_RUN_DIR:-$(ls -dt "runs/"*"-${TAG}" 2>/dev/null | head -1)}
  [[ -n $run && -f $run/recovery.md ]] || die "还没有报告（先跑 run）"
  sed -n '1,40p' "$run/recovery.md"
}

case ${1:-} in
  run) do_run ;;
  report) do_report ;;
  show) do_show ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
