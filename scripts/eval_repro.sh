#!/usr/bin/env bash
# 同种子重复评估，检验可复现性。
set -e
cd <workspace>
MOTION=data/processed/lafan_dance1_subject1.npz
rm -rf runs/20261112-001-eval/metrics.csv runs/20261112-001-eval/summary.json
MUJOCO_GL=egl environment/.venv/bin/python scripts/eval_tracking.py \
  --motion-file "$MOTION" --seed 0 --out runs/20261112-001-eval > environment/logs/eval_tracking.log 2>&1
echo "run1 EXIT=$?"
MUJOCO_GL=egl environment/.venv/bin/python scripts/eval_tracking.py \
  --motion-file "$MOTION" --seed 0 --out runs/20261112-001-eval/repro > environment/logs/eval_tracking_repro.log 2>&1
echo "run2 EXIT=$?"
python3 - <<'PY'
import json
a = json.load(open('<workspace>/runs/20261112-001-eval/summary.json'))
b = json.load(open('<workspace>/runs/20261112-001-eval/repro/summary.json'))
keys = ['mpkpe_mean_cm','r_mpkpe_mean_cm','joint_rmse_mean_rad','slip_left_total_cm','slip_right_total_cm']
for k in keys:
    print(k, a[k], b[k], 'diff', round(b[k]-a[k], 6))
PY
