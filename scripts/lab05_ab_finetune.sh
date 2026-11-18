#!/usr/bin/env bash
# LAB05 / M3-b —— 单参数 A/B 微调：motion_global_root_pos 权重 0.5->1.0、std 0.3->0.15
#
# 用法（在项目根目录，WSL 里执行）:
#   bash scripts/lab05_ab_finetune.sh train A          # 基线组：config 保持原样
#   （然后你亲手改 tracking_env_cfg.py 两行，见 LAB05）
#   bash scripts/lab05_ab_finetune.sh train B          # 实验组：脚本会先校验"只改了这两行"
#   bash scripts/lab05_ab_finetune.sh eval             # A/B 各 5 个种子评估（默认 101-105）
#   bash scripts/lab05_ab_finetune.sh report           # 生成对比表 comparison.md
#
# 默认规模：4096 env x 300 iter（从官方 model_49999 续训），seed 0，动作 S3。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CFG="repos/mjlab_research/src/mjlab/tasks/tracking/tracking_env_cfg.py"
CFG_SHA_A="f756b4d0e76a752a928cf41a6317f1147f8316319d8774c4c3769eb1df8b9a82"
BASE_CKPT="scratch/model_49999.pt"
MOTION_DEFAULT="data/processed/fight1_s3_000s_20s.npz"
START_ITER=49999
RUNS_BASE=${LAB05_RUNS_DIR:-runs}

die() { echo "[ERR] $*" >&2; exit 1; }
say() { echo "[lab05] $*"; }

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
  [ "$sha" = "$CFG_SHA_A" ] || die "config 当前 sha256=$sha，不是 A 态（应为 $CFG_SHA_A）。请先恢复原始文件（见 LAB05 的还原步骤）。"
}

train_stage() {
  local stage="$1" num_envs="$2" iters="$3" seed="$4" motion="$5"
  local run ckpt_sha motion_sha
  run="$(next_run_dir "m3b-$(echo "$stage" | tr 'AB' 'ab')-finetune")"
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
       --agent.run-name "m3b-$(echo "$stage" | tr 'AB' 'ab')"
       --log-root "$run/logs")

  printf '%s\n' "${CMD[@]}" > "$run/command.txt"
  say "run dir: $run"
  say "规模: ${num_envs} env x ${iters} iter, seed ${seed}"

  local started; started=$(date +%s)
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$run/vram_mib.txt" 2>/dev/null || true; sleep 0.5; done ) &
  local vpid=$!
  set +e
  MUJOCO_GL=egl "${CMD[@]}" 2>&1 | tee "$run/train.log"
  local rc=${PIPESTATUS[0]}
  set -e
  kill "$vpid" 2>/dev/null || true
  wait "$vpid" 2>/dev/null || true

  local expected=$((START_ITER + iters - 1))
  local last_it
  last_it=$(grep -ao 'Learning iteration [0-9]*/' "$run/train.log" | tail -1 | tr -dc '0-9')
  if [ "$last_it" != "$expected" ]; then
    say "警告：日志最后一次迭代是 ${last_it:-空}，期望 $expected（检查 max-iterations 语义）"
  fi
  local wall=$(( $(date +%s) - started ))

  cat > "$run/README.md" <<EOT
# run $(basename "$run") — M3-b ${stage} 组微调（从官方 model_49999 续训）

- 目的：M3-b 单参数对照的 ${stage} 组。**仅** ${stage} 组配置不同（见 cfg_snapshot.py 与 diff_A.txt）。
- 训练：${num_envs} env × ${iters} iter，seed ${seed}，num_steps_per_env 24，从 iter ${START_ITER} 续训到 ${START_ITER} + ${iters} - 1
- 动作：${motion} (sha256 ${motion_sha})
- 基础权重：${BASE_CKPT} (sha256 ${ckpt_sha})
- config sha256：$(cfg_sha)
- 命令（完整）：见 command.txt
- 墙钟：${wall} s；退出码：${rc}
- 产物：train.log、vram_mib.txt、logs/g1_tracking/*/model_*.pt、params.txt、cfg_snapshot.py

## 待填（用户）

- 我预测 B 组相对 A 的变化方向与理由：
EOT
  say "退出码 $rc，墙钟 ${wall}s"
  LAB05_RUN="$run"
}

case "${1:-}" in
  train)
    stage="${2:-}"; num_envs="${3:-4096}"; iters="${4:-300}"; seed="${5:-0}"; motion="${MOTION_FILE:-$MOTION_DEFAULT}"
    case "$stage" in
      A) check_cfg_is_A ;;
      B)
        A_RUN="$(newest_run m3b-a-finetune)"
        [ -n "$A_RUN" ] && [ -f "$A_RUN/cfg_snapshot.py" ] || die "找不到 A 组 run（runs/*-m3b-a-finetune）。先跑 train A。"
        diff -u "$A_RUN/cfg_snapshot.py" "$CFG" > /tmp/lab05_diff.txt || true
        added=$(awk 'substr($0,1,1)=="+" && substr($0,1,3)!="+++"' /tmp/lab05_diff.txt | wc -l | tr -d ' ')
        removed=$(awk 'substr($0,1,1)=="-" && substr($0,1,3)!="---"' /tmp/lab05_diff.txt | wc -l | tr -d ' ')
        if [ "$added" = "0" ]; then die "config 与 A 组完全相同：你还没改，或改了没保存。"; fi
        if [ "$added" != "2" ] || [ "$removed" != "2" ]; then
          die "期望只改 2 行（weight 与 std），实际 +$added/-$removed 行。diff 见 /tmp/lab05_diff.txt"
        fi
        block=$(sed -n '210,214p' "$CFG")
        echo "$block" | grep -q 'weight=1.0,' || die "第 210-214 行的 weight 应为 1.0"
        echo "$block" | grep -q '"std": 0.15' || die "第 210-214 行的 std 应为 0.15"
        say "config 校验通过：只改了 weight 与 std 两行"
        ;;
      *) die "stage 只能是 A 或 B" ;;
    esac
    LAB05_RUN=""
    train_stage "$stage" "$num_envs" "$iters" "$seed" "$motion"
    run="$LAB05_RUN"
    [ -n "$run" ] || die "train stage 未产生 run 目录"
    if [ "$stage" = "B" ]; then diff -u "$A_RUN/cfg_snapshot.py" "$CFG" > "$run/diff_A.txt" || true; fi
    ;;
  eval)
    shift
    seeds="${*:-101 102 103 104 105}"
    EVAL_RUN="$(next_run_dir m3b-eval)"; mkdir -p "$EVAL_RUN"
    for policy in A B; do
      run="$(newest_run "m3b-$(echo "$policy" | tr 'AB' 'ab')-finetune")"
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
    EVAL_RUN="$(newest_run m3b-eval)"
    [ -n "$EVAL_RUN" ] || die "找不到 eval run（runs/*-m3b-eval）"
    environment/.venv/bin/python scripts/lab05_ab_report.py --eval-dir "$EVAL_RUN"
    ;;
  *)
    sed -n '2,14p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
