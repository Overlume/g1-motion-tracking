#!/usr/bin/env bash
# LAB07 —— 权重扫描：只动 motion_global_root_pos 的 weight，std 固定 0.3
#
# 用法（项目根目录，WSL 里执行）:
#   bash scripts/lab07_weight_sweep.sh train 1.0     # 把 config 改成 weight=1.0 后跑这一档
#   bash scripts/lab07_weight_sweep.sh train 2.5     # 再改成 2.5 跑第二档
#   bash scripts/lab07_weight_sweep.sh eval          # 训练过的档各 5 个种子（默认 101-105）
#   bash scripts/lab07_weight_sweep.sh report        # 4 个臂一起汇总：comparison.md / sweep.csv / sweep.png
#
# 已有的两档不需要重训：w0.5 = runs/*-lab06-a-finetune（评估在 runs/*-lab06-eval/A），
#                       w1.5 = runs/*-lab06-b-finetune（评估在 runs/*-lab06-eval/B）。
# 本脚本只补 w1.0 与 w2.5，然后把四个臂放进同一张表。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CFG="repos/mjlab_research/src/mjlab/tasks/tracking/tracking_env_cfg.py"
CFG_SHA_A="f756b4d0e76a752a928cf41a6317f1147f8316319d8774c4c3769eb1df8b9a82"
BASE_CKPT="scratch/model_49999.pt"
MOTION_DEFAULT="data/processed/fight1_s3_000s_20s.npz"
START_ITER=49999
SEEDS_DEFAULT="101 102 103 104 105"
RUNS_BASE="$(echo "${LAB07_RUNS_DIR:-runs}")"
LAB06_RUNS_BASE="$(echo "${LAB06_RUNS_DIR:-runs}")"

die() { echo "[ERR] $*" >&2; exit 1; }
say() { echo "[lab07] $*"; }

cfg_sha() { sha256sum "$CFG" | cut -d' ' -f1; }
weight_key() { awk -v w="$1" 'BEGIN{printf "%02d", w*10}'; }

next_run_dir() {
  local tag="$1" day idx
  day=$(date +%Y%m%d); idx=1
  while compgen -G "$RUNS_BASE/${day}-$(printf "%03d" "$idx")-*" >/dev/null; do idx=$((idx + 1)); done
  echo "$RUNS_BASE/${day}-$(printf "%03d" "$idx")-${tag}"
}

newest_lab06_run() {
  local tag="$1"
  ls -d "$LAB06_RUNS_BASE"/*-"${tag}" 2>/dev/null | sort | tail -1
}

newest_run() {
  local tag="$1"
  ls -d "$RUNS_BASE"/*-"${tag}" 2>/dev/null | sort | tail -1
}

snapshot_machine() {
  local run="$1" tag="$2"
  { echo "# when: $(date -Is)"; free -m; } > "$run/mem_${tag}.txt" 2>&1 || true
  { echo "# when: $(date -Is)"; nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm --format=csv; } > "$run/gpu_${tag}.txt" 2>&1 || true
}

# A 态基线：优先用 LAB06-A 的 cfg_snapshot.py，没有就要求当前 config 就是 A 态（自测场景）
baseline_cfg() {
  local a_run
  a_run="$(newest_lab06_run lab06-a-finetune)"
  if [ -n "$a_run" ] && [ -f "$a_run/cfg_snapshot.py" ]; then
    echo "$a_run/cfg_snapshot.py"
  else
    [ "$(cfg_sha)" = "$CFG_SHA_A" ] || die "找不到 LAB06-A 快照，且当前 config 不是 A 态（sha=$(cfg_sha)）。先还原到 weight=0.5 / std=0.3。"
    echo "$CFG"
  fi
}

check_single_weight_edit() {
  local w="$1" base="$2" diff=/tmp/lab07_diff.txt
  diff -u "$base" "$CFG" > "$diff" || true
  local added removed
  added=$(awk 'substr($0,1,1)=="+" && substr($0,1,3)!="+++"' "$diff" | wc -l | tr -d ' ')
  removed=$(awk 'substr($0,1,1)=="-" && substr($0,1,3)!="---"' "$diff" | wc -l | tr -d ' ')
  [ "$added" = "0" ] && die "config 与 A 态完全相同：你还没改 weight（或改了没保存）。"
  if [ "$added" != "1" ] || [ "$removed" != "1" ]; then
    die "期望只改 1 行（只有 weight），实际 +${added}/-${removed} 行。diff 见 $diff"
  fi
  awk 'substr($0,1,1)=="+" && substr($0,1,3)!="+++"' "$diff" | grep -q 'weight=' \
    || die "被改的那一行不是 weight 行。diff 见 $diff"
  local block
  block=$(sed -n '210,214p' "$CFG")
  echo "$block" | grep -q "weight=${w}," || die "第 210-214 行的 weight 应为 ${w}（你可能改错了行或改了 std）"
  echo "$block" | grep -q '"std": 0.3' || die "第 210-214 行的 std 必须还是 0.3（本实验不动 std，见 D15）"
  say "config 校验通过：只有 weight=${w} 这一处改动，std 仍为 0.3"
}

train_arm() {
  local w="$1" num_envs="$2" iters="$3" seed="$4" motion="$5"
  local kg run ckpt_sha motion_sha expected
  kg=$(weight_key "$w")
  run="$(next_run_dir "lab07-w${kg}-finetune")"
  mkdir -p "$run/logs/g1_tracking/pretrained"

  [ -f "$BASE_CKPT" ] || die "找不到 $BASE_CKPT"
  [ -f "$motion" ] || die "找不到动作文件 $motion"
  cp "$BASE_CKPT" "$run/logs/g1_tracking/pretrained/model_49999.pt"
  cp "$CFG" "$run/cfg_snapshot.py"
  ckpt_sha=$(sha256sum "$BASE_CKPT" | cut -d' ' -f1)
  motion_sha=$(sha256sum "$motion" | cut -d' ' -f1)
  expected=$((START_ITER + iters - 1))

  cat > "$run/params.txt" <<EOT
stage=w${w}
num_envs=${num_envs}
iters=${iters}
seed=${seed}
motion=${motion}
motion_sha256=${motion_sha}
base_checkpoint=${BASE_CKPT}
base_checkpoint_sha256=${ckpt_sha}
start_iter=${START_ITER}
end_iter=${expected}
cfg_sha256=$(cfg_sha)
EOT

  local CMD
  CMD=(environment/.venv/bin/train Mjlab-Tracking-Flat-Unitree-G1
       --env.commands.motion.motion-file "$motion"
       --env.scene.num-envs "$num_envs"
       --agent.seed "$seed"
       --agent.max-iterations "$iters"
       --agent.num-steps-per-env 24
       --agent.save-interval 100
       --agent.logger tensorboard
       --agent.resume True
       --agent.load-run pretrained
       --agent.load-checkpoint model_49999.pt
       --agent.run-name "lab07-w${kg}"
       --log-root "$run/logs")

  printf '%s\n' "${CMD[@]}" > "$run/command.txt"
  say "run dir: $run"
  say "规模: ${num_envs} env x ${iters} iter, seed ${seed}, weight ${w}"
  snapshot_machine "$run" before

  printf 'timestamp,memory_used_mib,util_gpu_pct,temp_c,power_w\n' > "$run/gpu_samples.csv"
  local started; started=$(date +%s)
  ( while :; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits >> "$run/gpu_samples.csv" 2>/dev/null || true; sleep 2; done ) &
  local vpid=$!
  set +e
  MUJOCO_GL=egl "${CMD[@]}" 2>&1 | tee "$run/train.log"
  local rc=${PIPESTATUS[0]}
  set -e
  kill "$vpid" 2>/dev/null || true
  wait "$vpid" 2>/dev/null || true
  snapshot_machine "$run" after

  local last_it
  last_it=$(grep -ao 'Learning iteration [0-9]*/' "$run/train.log" | tail -1 | tr -dc '0-9')
  if [ "${last_it}" != "${expected}" ]; then
    say "警告：日志最后一次迭代是 ${last_it:-空}，期望 ${expected}（检查 --agent.max-iterations 语义）"
  fi
  local wall=$(( $(date +%s) - started ))

  {
    echo "# run $(basename "$run") — LAB07 权重扫描的 w${w} 档"
    echo ""
    echo "- 唯一变量：motion_global_root_pos 的 weight = ${w}（std 固定 0.3）"
    echo "- 训练：${num_envs} env x ${iters} iter，seed ${seed}，num_steps_per_env 24，从 iter ${START_ITER} 续训到 ${expected}"
    echo "- 动作：${motion} (sha256 ${motion_sha})"
    echo "- 基础权重：${BASE_CKPT} (sha256 ${ckpt_sha})"
    echo "- config sha256：$(cfg_sha)"
    echo "- 命令（完整）：见 command.txt"
    echo "- 墙钟：${wall} s（实时钟口径，见 D14）；退出码：${rc}"
    echo "- 产物：train.log、gpu_samples.csv（每 2 s 一次显存/温度/功耗）、gpu_/mem_ before/after、cfg_snapshot.py、params.txt"
    echo ""
    echo "## 待填（用户）"
    echo ""
    echo "- 我对这一档 root 误差的预测与理由（相对上一档）："
  } > "$run/README.md"

  say "退出码 ${rc}，墙钟 ${wall}s"
  LAB07_RUN="$run"
}

case "${1:-}" in
  train)
    w="${2:-}"; num_envs="${3:-4096}"; iters="${4:-300}"; seed="${5:-0}"
    [ -n "$w" ] || die "用法: train <weight> [num_envs] [iters] [seed]"
    case "$w" in *[!0-9.]*) die "weight 必须是数字，例如 1.0 或 2.5";; esac
    motion="${MOTION_FILE:-$MOTION_DEFAULT}"
    base="$(baseline_cfg)"
    check_single_weight_edit "$w" "$base"
    LAB07_RUN=""
    train_arm "$w" "$num_envs" "$iters" "$seed" "$motion"
    run="$LAB07_RUN"
    [ -n "$run" ] || die "train 未产生 run 目录"
    diff -u "$base" "$CFG" > "$run/diff_A.txt" || true
    ;;

  eval)
    shift || true
    weights="${*:-1.0 2.5}"
    seeds="${LAB07_SEEDS:-$SEEDS_DEFAULT}"
    EVAL_RUN="$(next_run_dir lab07-eval)"; mkdir -p "$EVAL_RUN"
    for w in $weights; do
      kg=$(weight_key "$w")
      sub="w${kg}"
      run="$(newest_run "lab07-w${kg}-finetune")"
      [ -n "$run" ] || die "找不到 w${w} 的训练 run（$RUNS_BASE/*-lab07-w${kg}-finetune）"
      iters=$(grep '^iters=' "$run/params.txt" | cut -d= -f2)
      ckpt=$(ls "$run"/logs/g1_tracking/*/model_$((START_ITER + iters - 1)).pt 2>/dev/null | head -1)
      [ -n "$ckpt" ] || die "w${w} 找不到 model_$((START_ITER + iters - 1)).pt（$run）"
      for s in $seeds; do
        out="$EVAL_RUN/${sub}/seed${s}"; mkdir -p "$out"
        say "eval w${w} seed ${s}"
        MUJOCO_GL=egl environment/.venv/bin/python scripts/eval_tracking.py \
          --checkpoint-file "$ckpt" \
          --motion-file "$(grep '^motion=' "$run/params.txt" | cut -d= -f2)" \
          --seed "$s" --out "$out" > "$EVAL_RUN/${sub}/seed${s}.log" 2>&1 || echo "[ERR] eval w${w} seed ${s} 失败，见日志"
      done
      echo "$ckpt" > "$EVAL_RUN/${sub}/ckpt.txt"
      echo "$run" > "$EVAL_RUN/${sub}/train_run.txt"
    done
    say "eval run dir: $EVAL_RUN"
    ;;

  report)
    EVAL_RUN="$(newest_run lab07-eval)"
    [ -n "$EVAL_RUN" ] || die "找不到 eval run（$RUNS_BASE/*-lab07-eval）"
    LAB06_EVAL="$(ls -d "$LAB06_RUNS_BASE"/*-lab06-eval 2>/dev/null | sort | tail -1 || true)"
    args=(--eval-dir "$EVAL_RUN")
    if [ -n "$LAB06_EVAL" ] && [ -d "$LAB06_EVAL/A" ]; then args+=(--arm "w0.5=$LAB06_EVAL/A"); fi
    for d in "$EVAL_RUN"/w*; do
      [ -d "$d" ] || continue
      kg=$(basename "$d" | sed 's/^w//')
      w=$(awk -v k="$kg" 'BEGIN{printf "%.1f", k/10}')
      args+=(--arm "w${w}=$d")
    done
    if [ -n "$LAB06_EVAL" ] && [ -d "$LAB06_EVAL/B" ]; then args+=(--arm "w1.5=$LAB06_EVAL/B"); fi
    say "臂: ${args[*]:1:2}"
    environment/.venv/bin/python scripts/lab07_sweep_report.py "${args[@]}"
    ;;

  *)
    sed -n '2,16p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
