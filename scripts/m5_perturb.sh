#!/usr/bin/env bash
# M5 / LAB10 —— torso_link 受控外力扰动（世界系 ±x/±y × 3 档冲量 × 5 种子）
#
# 用法（项目根目录）:
#   bash scripts/m5_perturb.sh run       # 4 方向 × 3 档 × 5 种子 = 60 次评估，约 40 min
#   bash scripts/m5_perturb.sh report    # 出 m5.md / m5.json / m5.png（baseline 用 LAB09 m4_train）
#   M5_RUN_DIR=runs/YYYYMMDD-NNN-m5-perturb M5_COMBOS="nx_j30 nx_j60 nx_j120" bash scripts/m5_perturb.sh run   # 补跑缺失组合
#
# 物理口径：torso_link 上施加世界系恒力 0.2 s（tsim 8.0-8.2 s，片段中段、避开片头与片尾）。
# 冲量 J = F * 0.2；整机质量 33.34 kg（g1.xml 30 个 body 质量求和）→ 等效力速 dv = J / 33.34。
#   j30 -> 150 N -> dv 0.90 m/s ；j60 -> 300 N -> dv 1.80 m/s ；j120 -> 600 N -> dv 3.60 m/s
# 自测观察：60 N(0.2 s) 几乎无影响；300 N 扰动后 1 s 内恢复；600 N 丢失跟踪（完成度 0.59）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUNS_BASE="$(echo "${M5_RUNS_DIR:-runs}")"
SEEDS="$(echo "${M5_SEEDS:-101 102 103 104 105}")"
FILTER="$(echo "${M5_COMBOS:-}")"
TIERS="$(echo "${M5_TIERS:-j30 j60 j120}")"
DURATION="0.2"
START_S="8.0"
M4_RUN_GLOB="runs/*-m4-w15-finetune"

die() { echo "[ERR] $*" >&2; exit 1; }
say() { echo "[m5] $*"; }

m4_ckpt() {
  local run ck
  run="$(ls -d $M4_RUN_GLOB 2>/dev/null | sort | tail -1 || true)"
  [ -n "$run" ] || die "找不到 M4 训练 run（$M4_RUN_GLOB）"
  ck=$(ls "$run"/logs/g1_tracking/*/model_52998.pt 2>/dev/null | head -1 || true)
  [ -n "$ck" ] || ck=$(ls "$run"/logs/g1_tracking/*/model_*.pt | sort -V | tail -1)
  [ -n "$ck" ] || die "找不到 ckpt（$run）"
  echo "$ck"
}

force_vec() { # dir tier -> "fx,fy,fz"
  local dir="$1" imp="$2" n f
  n="${imp#j}"
  [[ "$n" =~ ^[0-9]+$ ]] || die "未知档位 $imp（格式 j<N>，N = 冲量 N·s，如 j30 / j15）"
  f=$(( n * 5 ))   # 窗口 0.2 s：F = J / dt
  case "$dir" in
    px) echo "$f,0,0" ;;
    nx) echo "-$f,0,0" ;;
    py) echo "0,$f,0" ;;
    ny) echo "0,-$f,0" ;;
    *) die "未知方向 $dir（可选 px / nx / py / ny）" ;;
  esac
}

next_run_dir() {
  local tag="$1" day idx
  day=$(date +%Y%m%d); idx=1
  while compgen -G "$RUNS_BASE/${day}-$(printf "%03d" "$idx")-*" >/dev/null; do idx=$((idx + 1)); done
  echo "$RUNS_BASE/${day}-$(printf "%03d" "$idx")-${tag}"
}
newest_run() { ls -d "$RUNS_BASE"/*-"${1}" 2>/dev/null | sort | tail -1; }

run_stage() {
  local ck; ck="$(m4_ckpt)"
  [ -f data/processed/fight1_s3_000s_20s.npz ] || die "缺训练片段 npz"
  if [ -n "${M5_RUN_DIR:-}" ]; then
    RUN="$M5_RUN_DIR"
    [ -d "$RUN" ] || die "M5_RUN_DIR 不存在: $RUN"
    echo "- 追加运行：$(date -Is)" >> "$RUN/params.txt"
  else
    RUN="$(next_run_dir m5-perturb)"
  fi
  mkdir -p "$RUN"
  {
    echo "# M5 受控外力扰动"
    echo ""
    echo "- 对象：特化策略 \`${ck}\`（sha $(sha256sum "$ck" | cut -d' ' -f1)）"
    echo "- 动作：data/processed/fight1_s3_000s_20s.npz"
    echo "- 施加点：torso_link，世界系恒力，窗口 ${START_S}–$(awk -v a="$START_S" -v b="$DURATION" 'BEGIN{printf "%.2f", a+b}') s"
    echo "- 档位：j<N> = N N·s 冲量（F = N/0.2 s，dv = N/33.34 m/s）；本次组合：${FILTER:-j30 j60 j120}"
    echo "- 整机质量：33.34 kg（g1.xml 30 body 求和），dv = J / m"
    echo "- 评估：scripts/eval_tracking.py，终止检测开、启动 DR 关、999 步；种子 ${SEEDS}"
    echo "- 起始：$(date -Is)"
  } > "$RUN/params.txt"

  local up0 up1
  up0=$(cut -d' ' -f1 /proc/uptime)
  for dir in px nx py ny; do
    for imp in ${M5_TIERS:-j30 j60 j120}; do
      local combo="${dir}_${imp}"
      if [ -n "$FILTER" ] && ! echo " $FILTER " | grep -q " $combo "; then continue; fi
      local vec; vec="$(force_vec "$dir" "$imp")"
      for s in $SEEDS; do
        local out="$RUN/${combo}/seed${s}"
        mkdir -p "$out"
        say "${combo} seed ${s} force=${vec}"
        MUJOCO_GL=egl environment/.venv/bin/python scripts/eval_tracking.py \
          --checkpoint-file "$ck" \
          --motion-file data/processed/fight1_s3_000s_20s.npz \
          --seed "$s" \
          --perturb-body torso_link --perturb-force="$vec" \
          --perturb-start-s "$START_S" --perturb-duration-s "$DURATION" \
          --out "$out" > "$RUN/${combo}/seed${s}.log" 2>&1 || echo "[ERR] ${combo} seed ${s} 失败"
      done
    done
  done
  up1=$(cut -d' ' -f1 /proc/uptime)
  echo "- 真实墙钟（/proc/uptime）：$(awk -v a="$up0" -v b="$up1" 'BEGIN{printf "%.0f", b-a}') s" >> "$RUN/params.txt"
  say "run dir: $RUN"
}

report_stage() {
  local run base
  run="$(newest_run m5-perturb)"
  [ -n "$run" ] || die "找不到 M5 run（$RUNS_BASE/*-m5-perturb）"
  base="$(ls -d "$(echo "${LAB09_RUNS_DIR:-runs}")"/*-lab09-generalization 2>/dev/null | sort | tail -1 || true)"
  [ -n "$base" ] || die "找不到 LAB09 run（用作 baseline）"
  environment/.venv/bin/python scripts/m5_report.py --run "$run" --baseline "${base}/m4_train"
  say "report: $run/m5.md"
}

case "${1:-}" in
  run) run_stage ;;
  report) report_stage ;;
  *) sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
