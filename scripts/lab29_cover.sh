#!/usr/bin/env bash
# LAB29 —— 全段覆盖训练：dance1 的六个 20 s 窗口 + fight1 S3 0-20 s，一共 7 段一起训
#
# 目的：LAB28 显示「记忆没有可测的时间半径、增益更像内容相似」。本 lab 做反向检验：
#       把 dance1 六段全覆盖进训练，看六个窗口是不是一起被拉起来。
#       是  -> 覆盖率（哪段都练过就哪段都会）是主因；
#       否  -> 只有训练窗口本身好，方法侧才是瓶颈。
#
# 用法（项目根目录）：
#   LAB29_RUN=runs/20261127-001-lab29-cover bash scripts/lab29_cover.sh build   # 6 个窗口过地面校正（已存在的跳过）
#   LAB29_RUN=runs/20261127-001-lab29-cover bash scripts/lab29_cover.sh train   # 7 段打包 + 训练（可中断，重跑即续训）
#   LAB29_RUN=runs/20261127-001-lab29-cover bash scripts/lab29_cover.sh eval    # 9 个窗口 x 10 种子 = 90 次
#   LAB29_RUN=runs/20261127-001-lab29-cover bash scripts/lab29_cover.sh report  # 三策略对比表 + 图
#   LAB29_RUN=runs/20261127-001-lab29-cover bash scripts/lab29_cover.sh show    # 直接看表
#
# 对照数字直接复用历史 run（同协议同种子）：
#   M4 / LAB27 的 dance1 六窗口  -> runs/20261126-001-lab28-window-curve
#   M4 / LAB27 的 train/heldout/unseen -> runs/20261125-004-lab27-multiclip
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN="$LAB29_RUN"
SEEDS="$(echo $LAB29_SEEDS)"
FORCE="$LAB29_FORCE"
B28="$LAB29_L28_RUN"
B27="$LAB29_L27_RUN"

NAME="lab29-cover"
PY="environment/.venv/bin/python"
TARGET_ITER=52998

[ -n "$RUN" ] || RUN="$(ls -d runs/*-$NAME 2>/dev/null | sort | tail -1 || true)"
[ -n "$RUN" ] || RUN="runs/20261127-001-lab29-cover"
[ -n "$SEEDS" ] || SEEDS="101 102 103 104 105 106 107 108 109 110"
[ -n "$FORCE" ] || FORCE=0
[ -n "$B28" ] || B28="runs/20261126-001-lab28-window-curve"
[ -n "$B27" ] || B27="runs/20261125-004-lab27-multiclip"

say() { echo "[lab29] $*"; }
die() { echo "[ERR] $*" >&2; exit 1; }

[ "$(basename "$RUN")" != "runs" ] || die "RUN 不能是 runs 本身（会污染 runs 根目录）"

SRC_DANCE="data/processed/lafan_dance1_subject1.npz"
SRC_FIGHT="data/processed/fight1_s3_000s_20s.npz"
STARTS="0 20 40 60 80 100"

clip_dance() {
  printf 'data/processed/dance1_%03ds_20s_gfix.npz' "$1"
}

clips_csv() {
  local s out
  out="$SRC_FIGHT"
  for s in $STARTS; do
    out="$out,$(clip_dance "$s")"
  done
  echo "$out"
}

motion_of() {
  case "$1" in
    train)   echo "$SRC_FIGHT" ;;
    heldout) echo "data/processed/fight1_s3_120s_20s.npz" ;;
    unseen)  echo "data/processed/fight1_s2_134s_20s.npz" ;;
    w000)    echo "data/processed/dance1_000s_20s_gfix.npz" ;;
    w020)    echo "data/processed/dance1_020s_20s_gfix.npz" ;;
    w040)    echo "data/processed/dance1_040s_20s_gfix.npz" ;;
    w060)    echo "data/processed/dance1_060s_20s_gfix.npz" ;;
    w080)    echo "data/processed/dance1_080s_20s_gfix.npz" ;;
    w100)    echo "data/processed/dance1_100s_20s_gfix.npz" ;;
    *) die "未知窗口 $1" ;;
  esac
}

ckpt_of() {
  local ck
  ck="$(ls "$RUN"/logs/g1_tracking/*/model_*.pt 2>/dev/null | grep -v "/pretrained/" | sort -V | tail -1 || true)"
  [ -n "$ck" ] || die "找不到 LAB29 ckpt（$RUN/logs/g1_tracking/*/model_*.pt）；先跑 train"
  echo "$ck"
}

iter_of() {
  basename "$1" .pt | cut -d_ -f2
}

build_stage() {
  [ -f "$SRC_FIGHT" ] || die "缺 $SRC_FIGHT"
  [ -f "$SRC_DANCE" ] || die "缺 $SRC_DANCE"
  local s out
  for s in $STARTS; do
    out="$(clip_dance "$s")"
    if [ -f "$out" ]; then
      say "已有 $out（sha $(sha256sum "$out" | cut -d' ' -f1 | cut -c1-8)），跳过"
      continue
    fi
    say "地面校正 dance1 $s-$((s+20)) s（D41 闸门：max/p99 均 <= 20 mm）"
    "$PY" scripts/lab27_ground_fix.py --src "$SRC_DANCE" --start-s "$s" --dur-s 20 --out "$out" \
      || die "地面校正失败或闸门 FAIL：$out（看上面的 [gate] 行）"
  done
  for s in $STARTS; do
    out="$(clip_dance "$s")"
    [ -f "$out" ] || die "仍缺 $out"
  done
  say "7 段齐：$SRC_FIGHT + 6 个 dance1 窗口"
}

train_stage() {
  build_stage
  mkdir -p "$RUN"
  say "打包并训练（热启动 scratch/model_49999.pt，目标 iter $TARGET_ITER，seed 0 / 4096 envs）"
  say "片段顺序：$SRC_FIGHT, dance1 0/20/40/60/80/100 s"
  "$PY" scripts/lab27_multiclip_train.py \
    --run-root "$RUN" --mode full --resume --run-name "$NAME" --clips "$(clips_csv)"
}

eval_stage() {
  local ck n up0 up1 w mo s out
  ck="$(ckpt_of)"
  n="$(iter_of "$ck")"
  [ "$n" -ge "$TARGET_ITER" ] || die "ckpt 只到 iter $n（< $TARGET_ITER），训练没跑完；先跑 train"
  say "LAB29 ckpt: $ck（iter $n，sha $(sha256sum "$ck" | cut -d' ' -f1)）"
  say "种子: $SEEDS；窗口: train heldout unseen w000 w020 w040 w060 w080 w100"
  up0=$(cut -d' ' -f1 /proc/uptime)
  for w in train heldout unseen w000 w020 w040 w060 w080 w100; do
    mo="$(motion_of "$w")"
    [ -f "$mo" ] || die "缺窗口数据 $mo"
    for s in $SEEDS; do
      out="$RUN/eval/lab29/$w/seed$s"
      if [ -f "$out/summary.json" ] && [ "$FORCE" != "1" ]; then
        say "skip lab29/$w/seed$s（已有 summary.json）"
        continue
      fi
      mkdir -p "$out"
      say "lab29/$w/seed$s"
      MUJOCO_GL=egl "$PY" scripts/eval_tracking.py \
        --checkpoint-file "$ck" --motion-file "$mo" --seed "$s" --out "$out" \
        > "$RUN/eval/lab29/$w/seed$s.log" 2>&1 || echo "[ERR] lab29/$w/seed$s 失败"
    done
  done
  up1=$(cut -d' ' -f1 /proc/uptime)
  say "评估墙钟（/proc/uptime）：$(awk -v a="$up0" -v b="$up1" 'BEGIN{printf "%.0f", b-a}') s"
}

report_stage() {
  "$PY" scripts/lab29_report.py --run "$RUN" --b28 "$B28" --b27 "$B27" --figure \
    --title "LAB29 full coverage: 7 clips (fight1 S3 0s + dance1 x6) vs M4 / LAB27"
}

show_stage() {
  [ -f "$RUN/lab29_cover.md" ] || die "还没有 $RUN/lab29_cover.md，先跑 report"
  cat "$RUN/lab29_cover.md"
}

cmd="$1"
case "$cmd" in
  build)  build_stage ;;
  train)  train_stage ;;
  eval)   eval_stage ;;
  report) report_stage ;;
  show)   show_stage ;;
  *) die "用法：bash scripts/lab29_cover.sh build|train|eval|report|show" ;;
esac
