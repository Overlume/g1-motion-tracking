#!/usr/bin/env bash
# LAB24 —— 摩擦档强度扫描（可随时 Ctrl-C 暂停，重跑同一条命令自动续训）
#
# 背景：LAB23（D37）把 LAB20 的三档加强拆成单因素臂，只有摩擦档降 -x 失败率（28% -> 10%），
# 但代价是 54/54 非倒地 rollout 全被推走（非倒地 root 中位 81.2 cm、骨盆 0.418 m 未触地）= 「漂移型不倒地」；
# 质心/编码器档反而更差（72%/95%），三臂 held-out 全 0.018。本 LAB 只沿摩擦一档做强度扫描。
#
# 臂（friction 区间；com/bias 回 mjlab 默认 ±2.5/5/5 cm、±0.01 rad）：
#   f14 = 0.2–1.4  = LAB23 的 fric 臂，复用已有 ckpt（不重训）
#   f20 = 0.2–2.0  新训
#   f30 = 0.2–3.0  新训
# 训练口径同 M4/LAB20/LAB23：热启动 model_49999、3000 iter、4096 envs、seed 0、S3 0–20 s。
# 评估：nx_j60 × 20 种子 (101–120) × 3 遍，终止关、8.0 s 起推 0.2 s、推后看 3 s（LAB12 协议）。
#
# 用法（项目根目录）：
#   LAB24_RUN=runs/20261123-001-lab24-fric-scan bash scripts/lab24_fric_scan.sh run
#   ... run / train / eval / report / show / status
#
# 暂停与续训：
#   * 训练中 Ctrl-C 即暂停；mjlab 每 500 iter 存一次 ckpt（model_50000/50500/...），最多丢 500 iter（约 16 min）。
#   * 重跑同一条命令自动从该臂最新 ckpt 续到 iter 52998（只跑剩余差值）。
#   * 评估幂等：已有 summary.json 的种子、已有 recovery.json 的遍都跳过。
#
# 覆盖变量：LAB24_RUN / LAB24_ARMS / LAB24_SEEDS / LAB24_NX_REPS / LAB24_TARGET_ITER /
#           LAB24_TRAIN_ITERS / LAB24_ENVS / LAB24_WARM / LAB24_REUSE_F14 / LAB24_HELDOUT
set -eo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

PY=environment/.venv/bin/python
RUN=$LAB24_RUN
ARMS=$LAB24_ARMS
[ -n "$ARMS" ] || ARMS="f14 f20 f30"
SEEDS=$LAB24_SEEDS
[ -n "$SEEDS" ] || SEEDS="101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120"
NX_REPS=$LAB24_NX_REPS
[ -n "$NX_REPS" ] || NX_REPS=3
TARGET=$LAB24_TARGET_ITER
[ -n "$TARGET" ] || TARGET=52998
ITERS=$LAB24_TRAIN_ITERS
[ -n "$ITERS" ] || ITERS=3000
ENVS=$LAB24_ENVS
[ -n "$ENVS" ] || ENVS=4096
WARM=$LAB24_WARM
[ -n "$WARM" ] || WARM=scratch/model_49999.pt
REUSE=$LAB24_REUSE_F14
[ -n "$REUSE" ] || REUSE=runs/20261122-003-lab23-dr-ablation/arms/fric
HELDOUT=$LAB24_HELDOUT
[ -n "$HELDOUT" ] || HELDOUT=0
CMD=$1
[ -n "$CMD" ] || CMD=run

die() { echo "[ERR] $*" >&2; exit 1; }
say() { printf '\n[lab24] %s\n' "$*"; }

[ -n "$RUN" ] || die "必须指定 LAB24_RUN（例如 runs/20261123-001-lab24-fric-scan）"
case "$RUN" in
  runs|runs/|.|./) die "LAB24_RUN 不能是 runs 本身（会把结果写进根目录）" ;;
esac
[ -f "$WARM" ] || die "热启动权重不存在：$WARM（见 HANDOFF §6.2）"

fric_range() {
  case "$1" in
    f14) printf '0.2 1.4' ;;
    f20) printf '0.2 2.0' ;;
    f30) printf '0.2 3.0' ;;
    *) die "未知臂 $1（只能是 f14 / f20 / f30）" ;;
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

resolve_ckpt() {
  local arm ck
  arm=$1
  if [ "$arm" = "f14" ]; then
    ck=$(find_ckpt "$REUSE" || true)
    [ -n "$ck" ] || die "找不到 LAB23 fric 臂的 ckpt（$REUSE/logs/g1_tracking/*/model_*.pt），可用 LAB24_REUSE_F14 指定"
    printf '%s' "$ck"
    return 0
  fi
  ck=$(find_ckpt "$(arm_root "$arm")" || true)
  [ -n "$ck" ] && printf '%s' "$ck"
  return 0
}

train_arm() {
  local arm lo hi ck n remain args
  arm=$1
  if [ "$arm" = "f14" ]; then
    ck=$(resolve_ckpt f14)
    say "arm f14 复用 LAB23 fric 的 ckpt（不训练）：$ck"
    return 0
  fi
  mkdir -p "$(arm_root "$arm")"
  read -r lo hi <<< "$(fric_range "$arm")"
  args="--friction $lo $hi --com 0.025 0.05 0.05 --bias 0.01"
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
      --run-root "$(arm_root "$arm")" --mode full --run-name "lab24-$arm" \
      --warm-start "$ck" --iters "$remain" --envs "$ENVS" $args
  else
    say "arm $arm 全新训练：$ITERS iter / $ENVS envs（$args）"
    "$PY" -u scripts/lab20_dr_train.py \
      --run-root "$(arm_root "$arm")" --mode full --run-name "lab24-$arm" \
      --warm-start "$WARM" --iters "$ITERS" --envs "$ENVS" $args
  fi
  ck=$(resolve_ckpt "$arm" || true)
  [ -n "$ck" ] || die "arm $arm 训练结束但找不到 ckpt（看 $(arm_root "$arm")/logs）"
  n=$(iter_of "$ck")
  [ "$n" -ge "$TARGET" ] || die "arm $arm 训练停在 iter $n（目标 $TARGET）；重跑同一条命令会从该 ckpt 继续续训"
  say "arm $arm 训练完成：$ck（iter $n）"
}

eval_arm() {
  local arm ck rep d s out
  arm=$1
  ck=$(resolve_ckpt "$arm" || true)
  [ -n "$ck" ] || die "arm $arm 缺 ckpt"
  [ -f "$ck" ] || die "arm $arm 的 ckpt 不存在：$ck"
  rep=1
  while [ "$rep" -le "$NX_REPS" ]; do
    d="$(arm_root "$arm")/perturb/rep$rep"
    if [ -f "$d/recovery.json" ]; then
      say "arm $arm rep$rep nx_j60 已有判分，跳过"
    else
      say "arm $arm rep$rep nx_j60 评估（ckpt $(basename "$ck")）"
      LAB12_CKPT="$ck" LAB12_RUN_DIR="$d" LAB12_TAG="lab24-$arm-nx" \
        LAB12_CELLS="nx_j60" LAB12_SEEDS="$SEEDS" LAB12_SKIP_EXISTING=1 \
        bash scripts/lab12_recovery.sh run
    fi
    rep=$((rep + 1))
  done
  if [ "$HELDOUT" = "1" ]; then
    for s in 101 102 103 104 105 106 107 108 109 110; do
      out="$(arm_root "$arm")/eval/heldout/seed$s"
      [ -f "$out/summary.json" ] && continue
      mkdir -p "$out"
      say "arm $arm heldout seed $s"
      MUJOCO_GL=egl "$PY" scripts/eval_tracking.py \
        --checkpoint-file "$ck" \
        --motion-file data/processed/fight1_s3_120s_20s.npz \
        --seed "$s" --out "$out" > "$out/eval.log" 2>&1
    done
  fi
}

do_status() {
  local arm ck n reps line state
  printf '%-6s %-12s %-14s %s\n' "arm" "ckpt_iter" "reps_done" "ckpt"
  for arm in $ARMS; do
    ck=$(resolve_ckpt "$arm" || true)
    if [ -n "$ck" ]; then
      n=$(iter_of "$ck")
      line=$ck
    else
      n="—"
      line="（未训练）"
    fi
    reps=$(ls -d "$(arm_root "$arm")"/perturb/rep*/recovery.json 2>/dev/null | wc -l | tr -d ' ' || true)
    printf '%-6s %-12s %-14s %s\n' "$arm" "$n" "$reps/$NX_REPS" "$line"
  done
}

if [ "$CMD" = "run" ] || [ "$CMD" = "train" ]; then
  mkdir -p "$RUN"
  {
    echo "tag: lab24-fric-scan"
    echo "started: $(date -Iseconds)"
    echo "arms: $ARMS"
    echo "friction: f14=0.2-1.4(reuse LAB23 fric) f20=0.2-2.0 f30=0.2-3.0"
    echo "com: 0.025 0.05 0.05 (default)  bias: 0.01 (default)"
    echo "target_iter: $TARGET  train_iters: $ITERS  envs: $ENVS  warm: $WARM"
    echo "seeds: $SEEDS  nx_reps: $NX_REPS  heldout: $HELDOUT"
    echo "robustness: nx_j60，S3 0-20 s，终止关，8.0 s 起推 0.2 s，推后看 3 s"
    echo "warm_sha256: $(sha256sum "$WARM" | cut -d' ' -f1)"
  } > "$RUN/params.txt"
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
    "$PY" scripts/lab24_report.py --run "$RUN" --figure
    say "完成：$RUN/lab24_scan.md"
    ;;
  show)
    cat "$RUN/lab24_scan.md"
    ;;
  status)
    do_status
    ;;
  *)
    die "未知子命令 $CMD（run / train / eval / report / show / status）"
    ;;
esac
