#!/usr/bin/env bash
# 训练模式冒烟：测迭代耗时与显存，带 nvidia-smi 采样。
# 用法: bash scripts/smoke_train.sh NUM_ENVS ITERS [TAG]
set -e
cd <workspace>
NUM=${1:-512}
ITERS=${2:-10}
TAG=${3:-smoke}
LOG=environment/logs/train_${TAG}_${NUM}env.log
OUT=runs/20261112-003-train-scale/${TAG}_${NUM}env
mkdir -p "$OUT"
( while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$OUT/vram_mib.txt" 2>/dev/null || true
    sleep 0.5
  done ) &
SAMPLER=$!
MUJOCO_GL=egl environment/.venv/bin/train Mjlab-Tracking-Flat-Unitree-G1   --env.commands.motion.motion-file data/processed/lafan_dance1_subject1.npz   --env.scene.num-envs "$NUM" --env.seed 0 --agent.seed 0   --agent.max-iterations "$ITERS" --agent.logger tensorboard   --log-root "$OUT/logs" > "$LOG" 2>&1 || true
kill $SAMPLER 2>/dev/null || true
wait $SAMPLER 2>/dev/null || true
grep "Iteration time" "$LOG" | tail -6 || true
echo "peak vram MiB:" $(sort -n "$OUT/vram_mib.txt" | tail -1)
echo "EXIT done -> $LOG"
