#!/usr/bin/env bash
# LAB06 / M3-b 修正对照 —— 只改 weight：motion_global_root_pos weight 0.5 -> 1.5，std 保持 0.3
#
# 用法（项目根目录，WSL 里执行）:
#   bash scripts/lab06_weight_only.sh train A     # 基线组：config 保持原样
#   （然后你亲手改 1 行：weight=0.5 -> 1.5，std 不要动；见 docs/learning/LAB06-weight-only-ab.md）
#   bash scripts/lab06_weight_only.sh train B     # 实验组：脚本先校验“只改了这 1 行”
#   bash scripts/lab06_weight_only.sh eval        # A/B 各 5 个种子（默认 101-105）
#   bash scripts/lab06_weight_only.sh report      # 出 comparison.md / .json
#
# 默认规模：4096 env x 300 iter（从官方 model_49999 续训），seed 0，动作 S3。
# 与 LAB05 的唯一区别：本实验只动 weight（LAB05 同时动了 weight 与 std，分不开责任，见 D13）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CFG="repos/mjlab_research/src/mjlab/tasks/tracking/tracking_env_cfg.py"
CFG_SHA_A="f756b4d0e76a752a928cf41a6317f1147f8316319d8774c4c3769eb1df8b9a82"
BASE_CKPT="scratch/model_49999.pt"
MOTION_DEFAULT="data/processed/fight1_s3_000s_20s.npz"
START_ITER=49999
RUNS_BASE=${LAB06_RUNS_DIR:-runs}

die() { echo "[ERR] $*" >&2; exit 1; }
say() { echo "[lab06] $*"; }

next_run_dir() {
  local tag="$1" day idx
  day=$(date +%Y%m%d); idx=1
  while compgen -G "$RUNS_BASE/${day}-$(printf "%03d" "$idx")-*" >/dev/null; do idx=$((idx + 1)); done
  echo "$RUNS_BASE/${day}-$(printf "%03d" "$idx")-${tag}"
}

newest_run() {
  local tag="$1"
  ls -d "$RUNS_BASE"/*-"${tag}" 2>/dev/null | sort | tail -1
}

cfg_sha() { sha256sum "$CFG" | cut -d' ' -f1; }

check_cfg_is_A() {
  local sha; sha=$(cfg_sha)
  [ "$sha" = "$CFG_SHA_A" ] || die "config 当前 sha256=$sha，不是 A 态（应为 $CFG_SHA_A）。先按 LAB06 第 5 节还原。"
}

snapshot_machine() {
  local run="$1" tag="$2"
  { echo "# when: $(date -Is)"; free -m; } > "$run/mem_${tag}.txt" 2>&1 || true
  { echo "# when: $(date -Is)"; nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm --format=csv; } > "$run/gpu_${tag}.txt" 2>&1 || true
}

train_stage() {
  local stage="$1" num_envs="$2" iters="$3" seed="$4" motion="$5"
  local run ckpt_sha motion_sha
  run="$(next_run_dir "lab06-$(echo "$stage" | tr 'AB' 'ab')-finetune")"
  mkdir -p "$run/logs/g1_tracking/pretrained"

  [ -f "$BASE_CKPT" ] || die "找不到 $BASE_CKPT"
  [ -f "$motion" ] || die "找不到动作文件 $motion"
  cp "$BASE_CKPT" "$run/logs/g1_tracking/pretrained/model_49999.pt"
  cp "$CFG" "$run/cfg_snapshot.py"
  ckpt_sha=$(sha256sum "$BASE_CKPT" | cut -d' ' -f1)
  motion_sha=$(sha256sum "$motion" | cut -d' ' -f1)

  cat > "$run/params.txt" <<EOT
stage=$stage
num_envs=$num_envs
iters=$iters
seed=$seed
motion=$motion
motion_sha256=$motion_sha
base_checkpoint=$BASE_CKPT
base_checkpoint_sha256=$ckpt_sha
start_iter=$START_ITER
end_iter=$((START_ITER + iters - 1))
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
       --agent.run-name "lab06-$(echo "$stage" | tr 'AB' 'ab')"
       --log-root "$run/logs")

  printf '%s\n' "${CMD[@]}" > "$run/command.txt"
  say "run dir: $run"
  say "规模: ${num_envs} env x ${iters} iter, seed ${seed}"
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

  local expected=$((START_ITER + iters - 1))
  local last_it
  last_it=$(grep -ao 'Learning iteration [0-9]*/' "$run/train.log" | tail -1 | tr -dc '0-9')
  if [ "$last_it" != "$expected" ]; then
    say "警告：日志最后一次迭代是 ${last_it:-空}，期望 $expected（检查 max-iterations 语义）"
  fi
  local wall=$(( $(date +%s) - started ))

  {
    echo "# run $(basename "$run") — M3-b 修正对照（LAB06）的 ${stage} 组"
    echo ""
    echo "- 目的：只改 motion_global_root_pos 的 weight（0.5 -> 1.5），std 保持 0.3，检验 LAB05 的“核宽塌缩”解释。"
    echo "- 训练：${num_envs} env x ${iters} iter，seed ${seed}，num_steps_per_env 24，从 iter ${START_ITER} 续训到 ${expected}"
    echo "- 动作：${motion} (sha256 ${motion_sha})"
    echo "- 基础权重：${BASE_CKPT} (sha256 ${ckpt_sha})"
    echo "- config sha256：$(cfg_sha)"
    echo "- 命令（完整）：见 command.txt"
    echo "- 墙钟：${wall} s；退出码：${rc}"
    echo "- 产物：train.log、gpu_samples.csv（每 2 s 一次显存/温度/功耗）、gpu_before.txt / mem_before.txt / gpu_after.txt / mem_after.txt、logs/g1_tracking/*/model_*.pt、params.txt、cfg_snapshot.py"
    echo ""
    echo "## 待填（用户）"
    echo ""
    echo "- 我预测 B 组相对 A 的变化方向、幅度与理由："
  } > "$run/README.md"

  say "退出码 $rc，墙钟 ${wall}s"
  LAB06_RUN="$run"
}

case "${1:-}" in
  train)
    stage="${2:-}"; num_envs="${3:-4096}"; iters="${4:-300}"; seed="${5:-0}"; motion="${MOTION_FILE:-$MOTION_DEFAULT}"
    case "$stage" in
      A) check_cfg_is_A ;;
      B)
        A_RUN="$(newest_run lab06-a-finetune)"
        [ -n "$A_RUN" ] && [ -f "$A_RUN/cfg_snapshot.py" ] || die "找不到 A 组 run（runs/*-lab06-a-finetune）。先跑 train A。"
        diff -u "$A_RUN/cfg_snapshot.py" "$CFG" > /tmp/lab06_diff.txt || true
        added=$(awk 'substr($0,1,1)=="+" && substr($0,1,3)!="+++"' /tmp/lab06_diff.txt | wc -l | tr -d ' ')
        removed=$(awk 'substr($0,1,1)=="-" && substr($0,1,3)!="---"' /tmp/lab06_diff.txt | wc -l | tr -d ' ')
        if [ "$added" = "0" ]; then die "config 与 A 组完全相同：你还没改，或改了没保存。"; fi
        if [ "$added" != "1" ] || [ "$removed" != "1" ]; then
          die "期望只改 1 行（只有 weight），实际 +$added/-$removed 行。diff 见 /tmp/lab06_diff.txt"
        fi
        block=$(sed -n '210,214p' "$CFG")
        echo "$block" | grep -q 'weight=1.5,' || die "第 210-214 行的 weight 应为 1.5（注意别改到 std）"
        echo "$block" | grep -q '"std": 0.3' || die "第 210-214 行的 std 必须还是 0.3（本实验不动 std）"
        say "config 校验通过：只改了 weight（1.5），std 仍为 0.3"
        ;;
      *) die "stage 只能是 A 或 B" ;;
    esac
    LAB06_RUN=""
    train_stage "$stage" "$num_envs" "$iters" "$seed" "$motion"
    run="$LAB06_RUN"
    [ -n "$run" ] || die "train stage 未产生 run 目录"
    if [ "$stage" = "B" ]; then diff -u "$A_RUN/cfg_snapshot.py" "$CFG" > "$run/diff_A.txt" || true; fi
    ;;
  eval)
    shift
    seeds="${*:-101 102 103 104 105}"
    EVAL_RUN="$(next_run_dir lab06-eval)"; mkdir -p "$EVAL_RUN"
    for policy in A B; do
      run="$(newest_run "lab06-$(echo "$policy" | tr 'AB' 'ab')-finetune")"
      [ -n "$run" ] || die "找不到 $policy 组 run"
      iters=$(grep '^iters=' "$run/params.txt" | cut -d= -f2)
      ckpt=$(ls "$run"/logs/g1_tracking/*/model_$((START_ITER + iters - 1)).pt 2>/dev/null | head -1)
      [ -n "$ckpt" ] || die "$policy 组找不到 model_$((START_ITER + iters - 1)).pt（$run）"
      for s in $seeds; do
        out="$EVAL_RUN/$policy/seed$s"; mkdir -p "$out"
        say "eval $policy seed $s"
        MUJOCO_GL=egl environment/.venv/bin/python scripts/eval_tracking.py \
          --checkpoint-file "$ckpt" \
          --motion-file "$(grep '^motion=' "$run/params.txt" | cut -d= -f2)" \
          --seed "$s" --out "$out" > "$EVAL_RUN/$policy/seed$s.log" 2>&1 || echo "[ERR] eval $policy seed $s 失败，见日志"
      done
      echo "$ckpt" > "$EVAL_RUN/$policy/ckpt.txt"
      echo "$run" > "$EVAL_RUN/$policy/train_run.txt"
    done
    say "eval run dir: $EVAL_RUN"
    ;;
  report)
    EVAL_RUN="$(newest_run lab06-eval)"
    [ -n "$EVAL_RUN" ] || die "找不到 eval run（runs/*-lab06-eval）"
    environment/.venv/bin/python scripts/lab05_ab_report.py --eval-dir "$EVAL_RUN" \
      --title "M3-b 修正对照：只改 weight（LAB06）" \
      --labels "A = 原奖励（weight 0.5 / std 0.3）；B = 只改权重（weight 1.5 / std 0.3）。唯一变量：weight。"
    ;;
  *)
    sed -n '2,15p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac