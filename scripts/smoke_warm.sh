#!/usr/bin/env bash
set -u
cd <workspace>
rm -f /tmp/gpu_mem.txt
for i in $(seq 1 180); do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> /tmp/gpu_mem.txt
  sleep 1
done &
sampler=$!
NUM_ENVS=8 STEPS=400 environment/.venv/bin/python scripts/smoke_env.py > environment/logs/smoke_8env_warm.log 2>&1
echo "EXIT=$?"
sleep 1
kill "$sampler" 2>/dev/null || true
wait "$sampler" 2>/dev/null || true
echo "MAX_GPU_MIB=$(sort -n /tmp/gpu_mem.txt | tail -1)"
