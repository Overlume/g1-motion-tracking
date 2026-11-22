#!/usr/bin/env bash
# LAB23 —— DR 单档消融：把 LAB20 的三档加强拆成三个单因素臂（各自只加强一项）
#
# 背景：LAB20 三档同时加强 → held-out 0.100 → 0.018（D34）；LAB20b 显示 -x 60 N·s 失败率
# 30% → 0% 但扰动后 root 误差 18.7 → 92.9 cm（D35）；A1 审计把这种「不倒地但跟丢」量化成退让型（D36）。
# 本 LAB 回答：三档里到底是哪一档造成 held-out 崩塌与「退让式不倒地」。
#
# 单因素臂（其余两项回 mjlab 默认）：
#   fric = 摩擦 0.2–1.4（默认 0.3–1.2）
#   com  = 质心 ±4/±6/±6 cm（默认 ±2.5/±5/±5 cm）
#   bias = 编码器偏置 ±0.02 rad（默认 ±0.01）
# 训练口径与 M4 / LAB20 完全一致：热启动 model_49999、3000 iter、4096 envs、seed 0、S3 0–20 s。
#
# 用法（项目根目录）：
#   LAB23_RUN=runs/20261122-003-lab23-dr-ablation bash scripts/lab23_ablate.sh run    # 全量 3 臂 ≈7.5 h
#   LAB23_RUN=... LAB23_ARMS="fric com" bash scripts/lab23_ablate.sh run              # 只跑两臂 ≈5 h
#   LAB23_RUN=runs/... bash scripts/lab23_ablate.sh report                            # 只重算表与图
#   LAB23_RUN=runs/... bash scripts/lab23_ablate.sh show                              # 打印报告
#
# 可覆盖：LAB23_RUN / LAB23_ARMS / LAB23_SEEDS / LAB23_NX_REPS / LAB23_NY_REPS /
#         LAB23_HELDOUT_SEEDS / LAB23_ITERS / LAB23_ENVS / LAB23_WARM
set -eo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

PY=environment/.venv/bin/python
RUN=$LAB23_RUN
ARMS=$LAB23_ARMS
[ -n "$ARMS" ] || ARMS="fric com bias"
SEEDS=$LAB23_SEEDS
[ -n "$SEEDS" ] || SEEDS="101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120"
NX_REPS=$LAB23_NX_REPS
[ -n "$NX_REPS" ] || NX_REPS=3
NY_REPS=$LAB23_NY_REPS
[ -n "$NY_REPS" ] || NY_REPS=1
HELDOUT_SEEDS=$LAB23_HELDOUT_SEEDS
[ -n "$HELDOUT_SEEDS" ] || HELDOUT_SEEDS="101 102 103 104 105 106 107 108 109 110"
ITERS=$LAB23_ITERS
[ -n "$ITERS" ] || ITERS=3000
ENVS=$LAB23_ENVS
[ -n "$ENVS" ] || ENVS=4096
WARM=$LAB23_WARM
[ -n "$WARM" ] || WARM=scratch/model_49999.pt
CMD=$1
[ -n "$CMD" ] || CMD=run

die() { echo "[ERR] $*" >&2; exit 1; }
say() { printf '\n[lab23] %s\n' "$*"; }

[ -n "$RUN" ] || die "必须指定 LAB23_RUN（例如 runs/20261122-003-lab23-dr-ablation）"
case "$RUN" in
  runs|runs/|.|./) die "LAB23_RUN 不能是 runs 本身（会把结果写进根目录）" ;;
esac
[ -f "$WARM" ] || die "热启动权重不存在：$WARM（见 HANDOFF §6）"

arm_args() {
  case "$1" in
    fric) printf '%s' "--friction 0.2 1.4 --com 0.025 0.05 0.05 --bias 0.01" ;;
    com)  printf '%s' "--friction 0.3 1.2 --com 0.04 0.06 0.06 --bias 0.01" ;;
    bias) printf '%s' "--friction 0.3 1.2 --com 0.025 0.05 0.05 --bias 0.02" ;;
    *) die "未知臂 $1（只能是 fric / com / bias）" ;;
  esac
}

arm_ckpt() {
  local d ck
  d="$RUN/arms/$1/logs/g1_tracking"
  ck=$(ls "$d"/*/model_*.pt 2>/dev/null | grep -v "/pretrained/" | sort -V | tail -1 || true)
  [ -n "$ck" ] && echo "$ck"
  return 0
}

train_arm() {
  local arm ck args
  arm=$1
  ck=$(arm_ckpt "$arm" || true)
  if [ -n "$ck" ] && [ -f "$ck" ]; then
    say "arm $arm 已有 ckpt，跳过训练：$ck"
    return 0
  fi
  args=$(arm_args "$arm")
  say "arm $arm 开始训练：iters=$ITERS envs=$ENVS（$args）"
  "$PY" -u scripts/lab20_dr_train.py \
    --run-root "$RUN/arms/$arm" --mode full --run-name "lab23-$arm" \
    --warm-start "$WARM" --iters "$ITERS" --envs "$ENVS" $args
  ck=$(arm_ckpt "$arm" || true)
  [ -n "$ck" ] || die "arm $arm 训练结束但找不到 ckpt（看 $RUN/arms/$arm/logs）"
  say "arm $arm 训练完成：$ck"
}

robustness_arm() {
  local arm ck rep d
  arm=$1
  ck=$2
  rep=1
  while [ "$rep" -le "$NX_REPS" ]; do
    d="$RUN/arms/$arm/perturb/rep$rep"
    if [ -f "$d/recovery.json" ]; then
      say "arm $arm rep$rep nx_j60 已有判分，跳过"
    else
      say "arm $arm rep$rep nx_j60 评估（20 种子）"
      LAB12_CKPT="$ck" LAB12_RUN_DIR="$d" LAB12_TAG="lab23-$arm-nx" \
        LAB12_CELLS="nx_j60" LAB12_SEEDS="$SEEDS" \
        bash scripts/lab12_recovery.sh run
    fi
    rep=$((rep + 1))
  done
  if [ "$NY_REPS" -gt 0 ]; then
    d="$RUN/arms/$arm/perturb_ny/rep1"
    if [ -f "$d/recovery.json" ]; then
      say "arm $arm ny_j60 已有判分，跳过"
    else
      say "arm $arm ny_j60 评估（20 种子）"
      LAB12_CKPT="$ck" LAB12_RUN_DIR="$d" LAB12_TAG="lab23-$arm-ny" \
        LAB12_CELLS="ny_j60" LAB12_SEEDS="$SEEDS" \
        bash scripts/lab12_recovery.sh run
    fi
  fi
}

heldout_arm() {
  local arm ck s out
  arm=$1
  ck=$2
  for s in $HELDOUT_SEEDS; do
    out="$RUN/arms/$arm/eval/heldout/seed$s"
    if [ -f "$out/summary.json" ]; then continue; fi
    mkdir -p "$out"
    say "arm $arm heldout seed $s"
    MUJOCO_GL=egl "$PY" scripts/eval_tracking.py \
      --checkpoint-file "$ck" \
      --motion-file data/processed/fight1_s3_120s_20s.npz \
      --seed "$s" --out "$out" > "$out/eval.log" 2>&1
  done
}

if [ "$CMD" = "run" ]; then
  mkdir -p "$RUN"
  {
    echo "tag: lab23-dr-ablation"
    echo "started: $(date -Iseconds)"
    echo "arms: $ARMS"
    echo "iters: $ITERS  envs: $ENVS  warm: $WARM"
    echo "seeds: $SEEDS"
    echo "nx_reps: $NX_REPS  ny_reps: $NY_REPS"
    echo "heldout_seeds: $HELDOUT_SEEDS"
    echo "heldout_motion: data/processed/fight1_s3_120s_20s.npz"
    echo "robustness: nx_j60 / ny_j60，S3 0-20 s，终止关，8.0 s 起推 0.2 s，推后看 3 s"
    echo "warm_sha256: $(sha256sum "$WARM" | cut -d' ' -f1)"
  } > "$RUN/params.txt"
  for arm in $ARMS; do
    train_arm "$arm"
  done
  for arm in $ARMS; do
    ck=$(arm_ckpt "$arm" || true)
    [ -n "$ck" ] || die "arm $arm 缺 ckpt"
    robustness_arm "$arm" "$ck"
    heldout_arm "$arm" "$ck"
  done
  "$PY" scripts/lab23_report.py --run "$RUN" --figure
  say "全部完成：$RUN"
elif [ "$CMD" = "report" ]; then
  "$PY" scripts/lab23_report.py --run "$RUN" --figure
elif [ "$CMD" = "show" ]; then
  cat "$RUN/lab23_ablation.md"
else
  die "未知子命令 $CMD（run / report / show）"
fi
