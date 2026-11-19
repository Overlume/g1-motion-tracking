#!/usr/bin/env bash
# LAB13：定位"必倒"边界——在 300 与 600 N·s 之间补 400 / 450 N·s（−x 与 −y 各 5 种子）。
#
# 用法：
#   bash scripts/lab13_mid_impulse.sh run     # 20 次评估（4 格 x 5 种子），约 12 min
#   bash scripts/lab13_mid_impulse.sh report  # 只重跑判分
#   bash scripts/lab13_mid_impulse.sh show    # 打印汇总表
#
# 完全复用 LAB12 的评估与判分（同一 ckpt / 动作 / 推力窗口 / 判据），
# 只改 run 目录后缀与格列表：
#   j40 -> 400 N*s（200 N 持续 0.2 s）  j45 -> 450 N*s（225 N 持续 0.2 s）
#
# 覆盖变量：LAB13_TAG / LAB13_CELLS / LAB13_SEEDS（会转给 LAB12_*）
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

export LAB12_TAG="${LAB13_TAG:-lab13-mid-impulse}"
export LAB12_CELLS="${LAB13_CELLS:-nx_j40 nx_j45 ny_j40 ny_j45}"
export LAB12_SEEDS="${LAB13_SEEDS:-101 102 103 104 105}"

exec bash scripts/lab12_recovery.sh "$@"
