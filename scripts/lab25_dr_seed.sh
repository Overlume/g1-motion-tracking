#!/usr/bin/env bash
# LAB25 —— 摩擦上界的收敛实验（跨训练 seed + f30 的 −y 扩测）
#
# 背景（D38）：LAB24 只沿摩擦上界扫了 3 个点、每点 1 个训练 seed，结果非单调：
#   0.2–1.4 → 8%（f14）、0.2–2.0 → 65%（f20）、0.2–3.0 → 0%（f30）；
#   只有 3.0 档同时满足 D37 判据（k ≤10% 且非倒地 root 中位 ≤40 cm），但仍 21/60 漂 ≥30 cm。
# 两个未定问题：
#   (1) f20 的 65% 是「训练不稳」还是「摩擦上界非单调」？→ 各复训 1 个训练 seed（seed 1）。
#   (2) 3.0 档会不会也改 −y？→ 用现有 f30 ckpt 扩测 ny_j60 与 ny_j45（零训练）。
#
# 四个臂（默认顺序）：
#   f30ny60  eval-only：f30 ckpt × ny_j60 × 20 种子 × 3 遍
#   f30ny45  eval-only：f30 ckpt × ny_j45 × 20 种子 × 3 遍
#   f20s1    train：摩擦 0.2–2.0、训练 seed 1（其余同 f20 seed0）→ nx_j60 × 3 遍
#   f30s1    train：摩擦 0.2–3.0、训练 seed 1（其余同 f30 seed0）→ nx_j60 × 3 遍
#   f20s2 / f30s2  备用：同一设置、训练 seed 2（只在讲义 §6 分支预案里用）
#
# 训练口径同 M4/LAB20/LAB23/LAB24：热启动 model_49999、3000 iter、4096 envs、S3 0–20 s；
# 评估同 LAB12 协议：终止关、8.0 s 起推 0.2 s、推后看 3 s（j60 = 300 N、j45 = 225 N）。
#
# 用法（项目根目录）：
#   LAB25_RUN=runs/20261124-001-lab25-converge bash scripts/lab25_dr_seed.sh run
#   ... run / train / eval / report / show / status
#
# 暂停与续训：训练 Ctrl-C 后重跑同一条命令，自动从该臂最新 ckpt 续到 iter 52998；
# 评估幂等（已有 summary.json 的种子、已有 recovery.json 的遍都跳过）。
#
# 覆盖变量：LAB25_RUN / LAB25_ARMS / LAB25_SEEDS / LAB25_REPS / LAB25_TARGET_ITER /
#           LAB25_ITERS（= LAB25_TRAIN_ITERS，训练迭代数）/ LAB25_ENVS / LAB25_WARM /
#           LAB25_F30_DIR / LAB25_F30_SHA
set -eo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

PY=environment/.venv/bin/python
RUN=$LAB25_RUN
ARMS=$LAB25_ARMS
[ -n "$ARMS" ] || ARMS="f30ny60 f30ny45 f20s1 f30s1"
SEEDS=$LAB25_SEEDS
[ -n "$SEEDS" ] || SEEDS="101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120"
REPS=$LAB25_REPS
[ -n "$REPS" ] || REPS=3
TARGET=$LAB25_TARGET_ITER
[ -n "$TARGET" ] || TARGET=52998
ITERS=$LAB25_TRAIN_ITERS
[ -n "$ITERS" ] || ITERS=$LAB25_ITERS
[ -n "$ITERS" ] || ITERS=3000
ENVS=$LAB25_ENVS
[ -n "$ENVS" ] || ENVS=4096
WARM=$LAB25_WARM
[ -n "$WARM" ] || WARM=scratch/model_49999.pt
F30DIR=$LAB25_F30_DIR
[ -n "$F30DIR" ] || F30DIR=runs/20261123-001-lab24-fric-scan/arms/f30
F30SHA=$LAB25_F30_SHA
[ -n "$F30SHA" ] || F30SHA=1119f986fd1a33054eb584c9865dae7c499390f8f48a41229ef8c560b2455ba0
CMD=$1
[ -n "$CMD" ] || CMD=run

die() { echo "[ERR] $*" >&2; exit 1; }
say() { printf '\n[lab25] %s\n' "$*"; }

[ -n "$RUN" ] || die "必须指定 LAB25_RUN（例如 runs/20261124-001-lab25-converge）"
case "$RUN" in
  runs|runs/|.|./) die "LAB25_RUN 不能是 runs 本身（会把结果写进根目录）" ;;
esac
[ -f "$WARM" ] || die "热启动权重不存在：$WARM（见 HANDOFF §6.2）"

arm_mode() {
  case "$1" in
    f30ny60|f30ny45) printf 'eval' ;;
    f20s1|f30s1|f20s2|f30s2) printf 'train' ;;
    *) die "未知臂 $1（只能是 f30ny60 / f30ny45 / f20s1 / f30s1）" ;;
  esac
}

arm_cell() {
  case "$1" in
    f30ny60) printf 'ny_j60' ;;
    f30ny45) printf 'ny_j45' ;;
    f20s1|f30s1|f20s2|f30s2) printf 'nx_j60' ;;
    *) die "未知臂 $1" ;;
  esac
}

arm_fric() {
  case "$1" in
    f20s1|f20s2) printf '0.2 2.0' ;;
    f30s1|f30s2) printf '0.2 3.0' ;;
    *) die "臂 $1 不是训练臂" ;;
  esac
}

arm_seed() {
  case "$1" in
    f20s1|f30s1) printf '1' ;;
    f20s2|f30s2) printf '2' ;;
    *) die "臂 $1 不是训练臂" ;;
  esac
}

arm_root() { printf '%s' "$RUN/arms/$1"; }

iter_of() {
  local n
  n=$(basename "$1" .pt | sed 's/.*_//')
  case "$n" in
    ''|*[!0-9]*) printf '0' ;;
    *) printf '%s' "$n" ;;
  esac
}

find_ckpt() {
  local d ck
  d="$1/logs/g1_tracking"
  ck=$(ls "$d"/*/model_*.pt 2>/dev/null | grep -v "/pretrained/" | sort -V | tail -1 || true)
  [ -n "$ck" ] && printf '%s' "$ck"
  return 0
}

check_f30() {
  local ck sha
  ck=$(find_ckpt "$F30DIR" || true)
  [ -n "$ck" ] || die "找不到 f30 的 ckpt（$F30DIR/logs/g1_tracking/*/model_*.pt）；可用 LAB25_F30_DIR 指定"
  sha=$(sha256sum "$ck" | cut -d' ' -f1)
  if [ "$LAB25_SKIP_SHA_CHECK" = "1" ]; then
    say "跳过 f30 sha 校验（LAB25_SKIP_SHA_CHECK=1），当前 $sha"
    printf '%s' "$ck"
    return 0
  fi
  [ "$sha" = "$F30SHA" ] || die "f30 ckpt sha 不符：期望 $F30SHA，实际 $sha（ckpt=$ck）。确认没选到 pretrained 副本；真要换权重请设 LAB25_F30_SHA"
  printf '%s' "$ck"
}

resolve_ckpt() {
  local arm
  arm=$1
  if [ "$(arm_mode "$arm")" = "eval" ]; then
    check_f30
    return 0
  fi
  find_ckpt "$(arm_root "$arm")" || true
  return 0
}

train_arm() {
  local arm ck n remain args seed
  arm=$1
  [ "$(arm_mode "$arm")" = "train" ] || return 0
  mkdir -p "$(arm_root "$arm")"
  read -r lo hi <<< "$(arm_fric "$arm")"
  seed=$(arm_seed "$arm")
  args="--friction $lo $hi --com 0.025 0.05 0.05 --bias 0.01 --seed $seed"
  ck=$(resolve_ckpt "$arm" || true)
  if [ -n "$ck" ]; then
    n=$(iter_of "$ck")
    if [ "$n" -ge "$TARGET" ]; then
      say "arm $arm 已完成训练（ckpt iter $n >= $TARGET）：$ck"
      return 0
    fi
    remain=$((TARGET - n + 1))
    say "arm $arm 断点续训：从 iter $n 续 $remain iter（$ck）"
    "$PY" -u scripts/lab20_dr_train.py \
      --run-root "$(arm_root "$arm")" --mode full --run-name "lab25-$arm" \
      --warm-start "$ck" --iters "$remain" --envs "$ENVS" $args
  else
    say "arm $arm 全新训练：$ITERS iter / $ENVS envs / train seed $seed（$args）"
    "$PY" -u scripts/lab20_dr_train.py \
      --run-root "$(arm_root "$arm")" --mode full --run-name "lab25-$arm" \
      --warm-start "$WARM" --iters "$ITERS" --envs "$ENVS" $args
  fi
  ck=$(resolve_ckpt "$arm" || true)
  [ -n "$ck" ] || die "arm $arm 训练结束但找不到 ckpt（看 $(arm_root "$arm")/logs）"
  n=$(iter_of "$ck")
  [ "$n" -ge "$TARGET" ] || die "arm $arm 停在 iter $n（目标 $TARGET）；重跑同一条命令会继续续训"
  say "arm $arm 训练完成：$ck（iter $n）"
}

eval_arm() {
  local arm cell ck rep d
  arm=$1
  cell=$(arm_cell "$arm")
  ck=$(resolve_ckpt "$arm" || true)
  [ -n "$ck" ] || die "arm $arm 缺 ckpt"
  [ -f "$ck" ] || die "arm $arm 的 ckpt 不存在：$ck"
  rep=1
  while [ "$rep" -le "$REPS" ]; do
    d="$(arm_root "$arm")/perturb/rep$rep"
    if [ -f "$d/recovery.json" ]; then
      say "arm $arm rep$rep $cell 已有判分，跳过"
    else
      say "arm $arm rep$rep $cell 评估（ckpt $(basename "$ck")）"
      LAB12_CKPT="$ck" LAB12_RUN_DIR="$d" LAB12_TAG="lab25-$arm" \
        LAB12_CELLS="$cell" LAB12_SEEDS="$SEEDS" LAB12_SKIP_EXISTING=1 \
        bash scripts/lab12_recovery.sh run
    fi
    rep=$((rep + 1))
  done
}

do_status() {
  local arm ck n reps sha
  printf '%-8s %-6s %-8s %-11s %s\n' "arm" "mode" "cell" "ckpt_iter" "reps / ckpt"
  for arm in $ARMS; do
    ck=$(resolve_ckpt "$arm" || true)
    if [ -n "$ck" ]; then
      n=$(iter_of "$ck")
      sha=$(sha256sum "$ck" | cut -d' ' -f1 | cut -c1-8)
      ck="$sha… $(basename "$ck")"
    else
      n="—"
      ck="（未训练）"
    fi
    reps=$(ls -d "$(arm_root "$arm")"/perturb/rep*/recovery.json 2>/dev/null | wc -l | tr -d ' ' || true)
    printf '%-8s %-6s %-8s %-11s %s/%s  %s\n' "$arm" "$(arm_mode "$arm")" "$(arm_cell "$arm")" "$n" "$reps" "$REPS" "$ck"
  done
}

if [ "$CMD" = "run" ] || [ "$CMD" = "train" ]; then
  mkdir -p "$RUN"
  if [ -f "$RUN/params.txt" ]; then
    echo "resumed: $(date -Iseconds)（本次为续跑，原始参数见本文件开头）" >> "$RUN/params.txt"
  else
  {
    echo "tag: lab25-converge"
    echo "started: $(date -Iseconds)"
    echo "arms: $ARMS"
    echo "design: f30 复用 LAB24 的 ckpt 扩测 -y；f20/f30 各复训 1 个训练 seed（seed 1）"
    echo "friction: f20s1=0.2-2.0  f30s1=0.2-3.0  com=0.025 0.05 0.05 (default)  bias=0.01 (default)"
    echo "f30_ckpt_dir: $F30DIR  f30_sha256: $F30SHA"
    echo "target_iter: $TARGET  train_iters: $ITERS  envs: $ENVS  warm: $WARM"
    echo "seeds: $SEEDS  reps: $REPS"
    echo "cells: f30ny60=ny_j60  f30ny45=ny_j45  f20s1=nx_j60  f30s1=nx_j60；终止关、8.0 s 起推 0.2 s"
    echo "warm_sha256: $(sha256sum "$WARM" | cut -d' ' -f1)"
  } > "$RUN/params.txt"
  fi
  for arm in $ARMS; do
    train_arm "$arm"
  done
  [ "$CMD" = "run" ] || exit 0
fi

if [ "$CMD" = "run" ] || [ "$CMD" = "eval" ]; then
  for arm in $ARMS; do
    eval_arm "$arm"
  done
  [ "$CMD" = "run" ] || exit 0
fi

case "$CMD" in
  run|report)
    "$PY" scripts/lab25_report.py --run "$RUN" --figure
    say "完成：$RUN/lab25_converge.md"
    ;;
  show)
    cat "$RUN/lab25_converge.md"
    ;;
  status)
    do_status
    ;;
  *)
    die "未知子命令 $CMD（run / train / eval / report / show / status）"
    ;;
esac
