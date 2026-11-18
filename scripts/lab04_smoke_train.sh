#!/usr/bin/env bash
# LAB04 —— 用户亲手起一次冒烟训练（默认 512 env x 100 iter）。
#
# 用法:
#   bash scripts/lab04_smoke_train.sh                 # 512 env, 100 iter, seed 0
#   bash scripts/lab04_smoke_train.sh 256 100         # 改 env 数 / 迭代数
#   RUN_DIR=runs/xxx bash scripts/lab04_smoke_train.sh 512 100   # 指定输出目录
#
# 产物（全部在 RUN_DIR 内）:
#   README.md  env_versions.txt  train.log  vram_mib.txt  iter_times.csv  summary.json
#   logs/g1_tracking/<时间戳>_lab04/   （mjlab 写入的 params/ tensorboard/ ckpt/ onnx）
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || { echo "[ERR] 无法进入项目根目录" >&2; exit 1; }

NUM_ENVS="${1:-512}"
ITERS="${2:-100}"
SEED="${3:-0}"
MOTION="${MOTION_FILE:-data/processed/lafan_dance1_subject1.npz}"

if [ ! -x environment/.venv/bin/train ]; then
  echo "[ERR] 找不到 environment/.venv/bin/train —— 先跑 bash scripts/setup_env.sh" >&2
  exit 1
fi
if [ ! -f "$MOTION" ]; then
  echo "[ERR] 找不到动作文件 $MOTION" >&2
  exit 1
fi

day=$(date +%Y%m%d)
if [ -n "${RUN_DIR:-}" ]; then
  RUN="$RUN_DIR"
else
  idx=1
  while compgen -G "runs/$day-$(printf "%03d" "$idx")-*" >/dev/null; do
    idx=$((idx + 1))
  done
  RUN="runs/$day-$(printf "%03d" "$idx")-lab04-train-smoke"
fi
mkdir -p "$RUN/logs"

START_HUMAN=$(date "+%Y-%m-%d %H:%M:%S %z")
START_EPOCH=$(date +%s)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "no-commit")
GIT_DIRTY=$(git status --porcelain 2>/dev/null | wc -l | tr -d " ")

CMD=(environment/.venv/bin/train Mjlab-Tracking-Flat-Unitree-G1
     --env.commands.motion.motion-file "$MOTION"
     --env.scene.num-envs "$NUM_ENVS"
     --agent.seed "$SEED"
     --agent.max-iterations "$ITERS"
     --agent.num-steps-per-env 24
     --agent.save-interval 50
     --agent.logger tensorboard
     --agent.run-name lab04
     --log-root "$RUN/logs")

{
  echo "[env] $START_HUMAN"
  environment/.venv/bin/python - <<"PY"
import sys
import torch
import mujoco
import warp

print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
print("mujoco", mujoco.__version__)
print("warp", warp.__version__)
PY
  echo "git $GIT_COMMIT dirty_files=$GIT_DIRTY"
  echo "uname $(uname -srm)"
} > "$RUN/env_versions.txt"

{
  echo "# run $(basename "$RUN") — LAB04 冒烟训练（用户亲手执行）"
  echo
  echo "- 目的：验证 (a) 训练迭代能推进 (b) 参数确实更新 (c) 检查点能读回；为 M3-b 建立耗时/显存基线。**不是**学习效果实验。"
  echo "- 执行：$START_HUMAN；GPU：${GPU_NAME:-unknown}（driver ${GPU_DRIVER:-unknown}）"
  echo "- 规模：num_envs=$NUM_ENVS，iterations=$ITERS，seed=$SEED，num_steps_per_env=24 → $((NUM_ENVS * 24)) 条样本/迭代"
  echo "- 动作：$MOTION（LAFAN1 dance1，6574 帧 / 131.48 s / 50 fps）"
  echo "- git：$GIT_COMMIT，未提交改动 $GIT_DIRTY 个文件"
  echo
  echo "## 命令"
  echo
  printf '%s\n' '```bash'
  printf "MUJOCO_GL=egl"
  printf " %q" "${CMD[@]}"
  echo
  printf '%s\n' '```'
  echo
  echo "## 环境"
  echo
  printf '%s\n' '```'
  cat "$RUN/env_versions.txt"
  printf '%s\n' '```'
  echo
  echo "## 结果（脚本自动填）"
  echo
  echo "见 summary.json、train.log、vram_mib.txt、iter_times.csv。"
  echo
  echo "## 我的记录（用户手填）"
  echo
  echo "### 预测 vs 实测"
  echo
  echo "| 项目 | 我的预测 | 实测 | 偏差/解释 |"
  echo "| --- | --- | --- | --- |"
  echo "| 稳态迭代耗时 s（不含第 0 次） |  |  |  |"
  echo "| 第 0 次迭代耗时 s |  |  |  |"
  echo "| 显存峰值 MiB |  |  |  |"
  echo "| 100 次后 mean reward 走势 |  |  |  |"
  echo
  echo "### 四个问题"
  echo
  echo "1. 512 env 的迭代耗时与 1024 / 4096 env 对比，说明瓶颈在哪？"
  echo "2. 显存峰值离 6 GB 余量多少？训练时 free -h 的可用内存是多少？"
  echo "3. 100 次迭代后 mean reward 变化能不能说明学会了？为什么？"
  echo "4. model_0.pt 与 model_99.pt 的 actor 参数平均变化量是多少？为 0 意味着什么？"
  echo
  echo "### 结论（一句话）"
} > "$RUN/README.md"

echo "[lab04] 运行目录: $RUN"
echo "[lab04] 训练开始（日志 $RUN/train.log），预计 2-4 分钟 ..."
echo
: > "$RUN/vram_mib.txt"
( while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null >> "$RUN/vram_mib.txt"
    sleep 0.5
  done ) &
SAMPLER=$!
cleanup() {
  kill "$SAMPLER" 2>/dev/null
  wait "$SAMPLER" 2>/dev/null
}
trap cleanup EXIT INT TERM

MUJOCO_GL=egl "${CMD[@]}" 2>&1 | tee "$RUN/train.log"
RC=${PIPESTATUS[0]}
cleanup
trap - EXIT INT TERM
END_EPOCH=$(date +%s)
echo
echo "[lab04] 训练退出码 $RC；墙钟 $((END_EPOCH - START_EPOCH)) s"

environment/.venv/bin/python scripts/lab04_summarize.py \
  "$RUN" "$NUM_ENVS" "$ITERS" "$SEED" "$MOTION" "$RC"

if [ "$RC" -ne 0 ]; then
  echo
  echo "[lab04] 训练失败，train.log 末尾 40 行:"
  tail -40 "$RUN/train.log"
  exit "$RC"
fi

LAST_CKPT=$(ls -1 "$RUN"/logs/*/*/model_*.pt 2>/dev/null | sort -V | tail -1)
echo
echo "[lab04] 下一步:"
echo "  1) 读 $RUN/summary.json，把你预测和实测的差填进 README.md 的『我的记录』"
echo "  2) 自己比一次权重（LAB04 第 3.4 节），确认 mean |dw| > 0"
if [ -n "${LAST_CKPT:-}" ]; then
  echo "  3) 用训练出的检查点回放（画面难看是正常的）:"
  echo
  echo "     MUJOCO_GL=egl environment/.venv/bin/python scripts/replay_policy.py \\"
  echo "       --checkpoint-file $LAST_CKPT \\"
  echo "       --motion-file $MOTION \\"
  echo "       --num-envs 1 --video-length 200 --width 640 --height 360 --ignore-terminations --tag lab04_ckpt"
fi
echo
echo "[lab04] done"
