#!/usr/bin/env bash
# usage: smoke_mem.sh NUM_ENVS STEPS
set -u
N="${1:-64}"; S="${2:-200}"
cd <workspace>
rm -f /tmp/gpu_mem.txt
(
  start=$(date +%s)
  while [ $(( $(date +%s) - start )) -lt 600 ]; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> /tmp/gpu_mem.txt
    sleep 0.5
  done
) &
sampler=$!
env NUM_ENVS="$N" STEPS="$S" environment/.venv/bin/python scripts/smoke_env.py > environment/logs/smoke_${N}env.log 2>&1
code=$?
kill "$sampler" 2>/dev/null || true
wait "$sampler" 2>/dev/null || true
echo "EXIT=$code"
grep -E "^(create|chunk|env.step|steps/s)" environment/logs/smoke_${N}env.log
echo "PEAK_GPU_MIB=$(sort -n /tmp/gpu_mem.txt | tail -1)"
