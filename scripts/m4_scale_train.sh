#!/usr/bin/env bash
# M4 —— 规模训练：weight 1.5（D16 定），3000 iter，4096 env，seed 0，S3，从官方 model_49999 续训
#
# 用法（项目根目录，WSL 里执行）:
#   bash scripts/m4_scale_train.sh train      # 约 1.7 h（4096 env x 3000 iter）
#   bash scripts/m4_scale_train.sh eval       # 5 个种子评估最终 ckpt（约 5-10 min）
#   bash scripts/m4_scale_train.sh report     # 与 300 iter 臂并表：comparison.md / sweep.png
#
# 前提：config 必须已改成 weight=1.5、std 仍为 0.3（脚本会校验；A 态 sha 见下）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CFG="repos/mjlab_research/src/mjlab/tasks/tracking/tracking_env_cfg.py"
CFG_SHA_A="f756b4d0e76a752a928cf41a6317f1147f8316319d8774c4c3769eb1df8b9a82"
BASE_CKPT="scratch/model_49999.pt"
MOTION_DEFAULT="data/processed/fight1_s3_000s_20s.npz"
START_ITER=49999
ITERS="$(echo "${M4_ITERS:-3000}")"
SEEDS_DEFAULT="101 102 103 104 105"
RUNS_BASE="$(echo "${M4_RUNS_DIR:-runs}")"
LAB06_RUNS_BASE="$(echo "${LAB06_RUNS_DIR:-runs}")"
SAVE_INTERVAL="$(echo "${M4_SAVE_INTERVAL:-500}")"

die() { echo "[ERR] $*" >&2; exit 1; }
say() { echo "[m4] $*"; }
cfg_sha() { sha256sum "$CFG" | cut -d' ' -f1; }
uptime_s() { cut -d' ' -f1 /proc/uptime; }

next_run_dir() {
  local tag="$1" day idx
  day=$(date +%Y%m%d); idx=1
  while compgen -G "$RUNS_BASE/${day}-$(printf "%03d" "$idx")-*" >/dev/null; do idx=$((idx + 1)); done
  echo "$RUNS_BASE/${day}-$(printf "%03d" "$idx")-${tag}"
}
newest_run() { ls -d "$RUNS_BASE"/*-"${1}" 2>/dev/null | sort | tail -1; }
newest_lab06_run() { ls -d "$LAB06_RUNS_BASE"/*-"${1}" 2>/dev/null | sort | tail -1; }

snapshot_machine() {
  local run="$1" tag="$2"
  { echo "# when: $(date -Is)"; free -m; } > "$run/mem_${tag}.txt" 2>&1 || true
  { echo "# when: $(date -Is)"; nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm --format=csv; } > "$run/gpu_${tag}.txt" 2>&1 || true
}

check_cfg_m4() {
  local added removed diff=/tmp/m4_diff.txt base
  base="$(newest_lab06_run lab06-a-finetune)/cfg_snapshot.py"
  [ -f "$base" ] || die "找不到 LAB06-A 快照（$base），无法确认只有 weight 改动"
  diff -u "$base" "$CFG" > "$diff" || true
  added=$(awk 'substr($0,1,1)=="+" && substr($0,1,3)!="+++"' "$diff" | wc -l | tr -d ' ')
  removed=$(awk 'substr($0,1,1)=="-" && substr($0,1,3)!="---"' "$diff" | wc -l | tr -d ' ')
  if [ "$added" != "1" ] || [ "$removed" != "1" ]; then
    die "M4 要求相对 A 态恰好 +1/-1 行（只改 weight），实际 +${added}/-${removed}。diff 见 $diff"
  fi
  local block; block=$(sed -n '210,214p' "$CFG")
  echo "$block" | grep -q 'weight=1.5,' || die "第 210-214 行的 weight 必须是 1.5（D16 定的 M4 配置）"
  echo "$block" | grep -q '"std": 0.3' || die "std 必须仍是 0.3"
  say "config 校验通过：weight=1.5 / std=0.3（相对 A 态只改 1 行）"
}

train_stage() {
  local num_envs="$1" iters="$2" seed="$3" motion="$4"
  local run ckpt_sha motion_sha expected up0 up1 wall
  run="$(next_run_dir "m4-w15-finetune")"
  mkdir -p "$run/logs/g1_tracking/pretrained"

  [ -f "$BASE_CKPT" ] || die "找不到 $BASE_CKPT"
  [ -f "$motion" ] || die "找不到动作文件 $motion"
  cp "$BASE_CKPT" "$run/logs/g1_tracking/pretrained/model_49999.pt"
  cp "$CFG" "$run/cfg_snapshot.py"
  ckpt_sha=$(sha256sum "$BASE_CKPT" | cut -d' ' -f1)
  motion_sha=$(sha256sum "$motion" | cut -d' ' -f1)
  expected=$((START_ITER + iters - 1))

  cat > "$run/params.txt" <<EOT
stage=m4
num_envs=${num_envs}
iters=${iters}
seed=${seed}
motion=${motion}
motion_sha256=${motion_sha}
base_checkpoint=${BASE_CKPT}
base_checkpoint_sha256=${ckpt_sha}
start_iter=${START_ITER}
end_iter=${expected}
env_steps_per_iter=${num_envs}*24
save_interval=${SAVE_INTERVAL}
cfg_sha256=$(cfg_sha)
EOT

  local CMD
  CMD=(environment/.venv/bin/train Mjlab-Tracking-Flat-Unitree-G1
       --env.commands.motion.motion-file "$motion"
       --env.scene.num-envs "$num_envs"
       --agent.seed "$seed"
       --agent.max-iterations "$iters"
       --agent.num-steps-per-env 24
       --agent.save-interval "${SAVE_INTERVAL}"
       --agent.logger tensorboard
       --agent.resume True
       --agent.load-run pretrained
       --agent.load-checkpoint model_49999.pt
       --agent.run-name "m4-w15"
       --log-root "$run/logs")

  printf '%s\n' "${CMD[@]}" > "$run/command.txt"
  say "run dir: $run"
  say "规模: ${num_envs} env x ${iters} iter（${START_ITER} -> ${expected}），seed ${seed}，weight 1.5"
  snapshot_machine "$run" before

  printf 'timestamp,memory_used_mib,util_gpu_pct,temp_c,power_w\n' > "$run/gpu_samples.csv"
  up0=$(uptime_s)
  ( while :; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits >> "$run/gpu_samples.csv" 2>/dev/null || true; sleep 5; done ) &
  local vpid=$!
  set +e
  MUJOCO_GL=egl "${CMD[@]}" 2>&1 | tee "$run/train.log"
  local rc=${PIPESTATUS[0]}
  set -e
  kill "$vpid" 2>/dev/null || true
  wait "$vpid" 2>/dev/null || true
  snapshot_machine "$run" after
  up1=$(uptime_s)
  wall=$(awk -v a="$up0" -v b="$up1" 'BEGIN{printf "%.0f", b-a}')

  local last_it
  last_it=$(grep -ao 'Learning iteration [0-9]*/' "$run/train.log" | tail -1 | tr -dc '0-9')
  [ "${last_it}" = "${expected}" ] || say "警告：日志最后迭代 ${last_it:-空}，期望 ${expected}"

  {
    echo "# run $(basename "$run") — M4 规模训练（weight 1.5 / std 0.3）"
    echo ""
    echo "- 目的：在 LAB07 选定的 weight 1.5 上把预算从 300 iter 提到 3000 iter，看 root 误差是否随预算下降、姿态是否付钱。"
    echo "- 训练：${num_envs} env x ${iters} iter，seed ${seed}，num_steps_per_env 24，${START_ITER} -> ${expected}，save_interval ${SAVE_INTERVAL}"
    echo "- 环境步数：${iters} x ${num_envs} x 24 ≈ $(awk -v a="$iters" -v b="$num_envs" 'BEGIN{printf "%d", a*b*24}')"
    echo "- 动作：${motion} (sha256 ${motion_sha})；基础权重：${BASE_CKPT} (sha256 ${ckpt_sha})"
    echo "- 真实墙钟（/proc/uptime）：${wall} s；退出码 ${rc}"
    echo "- 命令：见 command.txt；配置：cfg_snapshot.py（sha $(cfg_sha)）"
    echo ""
    echo "## 待填（用户）"
    echo ""
    echo "- 我预测 3000 iter 后 root 误差会到多少 cm："
    echo "- 我预测 R-MPKPE 会变好还是变差，为什么："
  } > "$run/README.md"

  say "退出码 ${rc}，真实墙钟 ${wall}s（实时钟口径见 D14）"
  M4_RUN="$run"
}

case "${1:-}" in
  train)
    num_envs="${2:-4096}"
    [ -f "$BASE_CKPT" ] || die "找不到 $BASE_CKPT"
    check_cfg_m4
    motion="${MOTION_FILE:-$MOTION_DEFAULT}"
    M4_RUN=""
    train_stage "$num_envs" "$ITERS" "${3:-0}" "$motion"
    run="$M4_RUN"
    [ -n "$run" ] || die "train 未产生 run 目录"
    diff -u "$(newest_lab06_run lab06-a-finetune)/cfg_snapshot.py" "$CFG" > "$run/diff_A.txt" || true
    ;;
  eval)
    shift || true
    seeds="${*:-$SEEDS_DEFAULT}"
    run="$(newest_run m4-w15-finetune)"
    [ -n "$run" ] || die "找不到 M4 训练 run（$RUNS_BASE/*-m4-w15-finetune）"
    iters=$(grep '^iters=' "$run/params.txt" | cut -d= -f2)
    expected=$((START_ITER + iters - 1))
    ckpt=$(ls "$run"/logs/g1_tracking/*/model_${expected}.pt 2>/dev/null | head -1)
    [ -n "$ckpt" ] || die "找不到 model_${expected}.pt（$run）"
    EVAL_RUN="$(next_run_dir m4-eval)"; mkdir -p "$EVAL_RUN/final"
    echo "$ckpt" > "$EVAL_RUN/final/ckpt.txt"
    echo "$run" > "$EVAL_RUN/final/train_run.txt"
    for s in $seeds; do
      out="$EVAL_RUN/final/seed${s}"; mkdir -p "$out"
      say "eval seed ${s}"
      MUJOCO_GL=egl environment/.venv/bin/python scripts/eval_tracking.py \
        --checkpoint-file "$ckpt" \
        --motion-file "$(grep '^motion=' "$run/params.txt" | cut -d= -f2)" \
        --seed "$s" --out "$out" > "$EVAL_RUN/final/seed${s}.log" 2>&1 || echo "[ERR] eval seed ${s} 失败"
    done
    say "eval run dir: $EVAL_RUN"
    ;;
  report)
    EVAL_RUN="$(newest_run m4-eval)"
    [ -n "$EVAL_RUN" ] || die "找不到 eval run（$RUNS_BASE/*-m4-eval）"
    LAB06_EVAL="$(ls -d "$LAB06_RUNS_BASE"/*-lab06-eval 2>/dev/null | sort | tail -1 || true)"
    [ -n "$LAB06_EVAL" ] && [ -d "$LAB06_EVAL/B" ] || die "找不到 300 iter 参照（$LAB06_EVAL/B）"
    environment/.venv/bin/python scripts/lab07_sweep_report.py \
      --eval-dir "$EVAL_RUN" \
      --title "M4 规模训练：300 iter vs 3000 iter（weight 1.5）" \
      --xlabel "training iterations" \
      --arm "300=${LAB06_EVAL}/B" \
      --arm "3000=${EVAL_RUN}/final"
    ;;
  *)
    sed -n '2,12p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
