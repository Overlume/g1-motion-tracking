# LAB19b — 补 +x 的 100 N·s（1 格 × 20 种子 × 3 遍）

- 目的：把 +x 的 down 边界从「80–120」缩到「80–100」或「100–120」；判断 +x 的过渡是陡转还是缓升。
- 假设：用户按讲义要求先写预测（本次未作答，参考回答见 `docs/learning/LAB19b-prediction.md`）。
- 配置：ckpt model_52998（sha 前 8 位 fc8f975b）、fight1_s3_000s_20s.npz、终止关、8.0 s 起推 0.2 s、种子 101–120、3 遍、格 px_j100（500 N × 0.2 s = 100 N·s）。
- 命令：见 `docs/learning/LAB19b-px100.md` §2；三遍脚本 /tmp/run_lab19b.sh。
- 耗时：rep1 15:49:21–16:02:11、rep2 16:02:11–16:14:28、rep3 16:14:28–16:26:44，**合计 37 min 23 s**（约 12–13 min/遍，37 s/次）。

## 结果

| 遍 | n | stable | transient | down | 倒地率与 CP 95% |
| --- | --- | --- | --- | --- | --- |
| rep1 | 20 | 8 | 12 | **0** | 0% |
| rep2 | 20 | 8 | 12 | **0** | 0% |
| rep3 | 20 | 7 | 13 | **0** | 0% |
| 合计 | 20 种子 | 8/8/7 | 12/12/13 | **k = 0 / 0 / 0** | 0%，[0%, 16.8%] |

- 二进制翻转 **0/20**（k 三遍一致）；三分类翻转 **6/20**：104、107、109、110、112、113，全部 stable ↔ transient（k 不受影响）。
- 101–105 子集：三遍 0/5 down（其余为 transient 或 stable）；无 LAB12 基线可比（表里「—」）。

## 判读

1. **+x 的 down 边界 >100 N·s，落在 100–120 之间**：100 = 0/20（上界 16.8%），120 N·s（LAB12，5 种子）= 3/5（CI 14.7–94.7%）→ 50% 交点只能写「约 110–120」，且「120 比 100 更脆」只在点估计上成立（两 CI 刚擦边）。
2. **stable 份额非单调**：60 N·s 20/20 → 80 N·s **1/20** → 100 N·s **7–8/20** → 120 N·s 2/5。力更大反而更多种子没跨过崩溃阈值；推测与「被推一下→迈步调整」有关，**未做逐帧验证**，只记观察。
3. **三分类翻转 6/20 是目前最多的一格**（LAB16 两格合计 5、LAB17 3、LAB18 4、LAB19 0）→ k=0 的格子二进制稳定，但「崩没崩」很脆，引用时三种判定一起给。
4. 四向表补齐：+x 在 60/80/100/120 四档为 0/20、0/20、0/20、3/5 → 边界顺序 +y < −y < −x < +x 不变（D32/D33）。

## 局限

- 单一 ckpt（model_52998）、单一时段（8.0 s）、单一波形（0.2 s 方波）、单一作用点（torso_link）。
- 120 N·s 那一点只有 5 种子，所以 100–120 的区间无法再收紧；要定 50% 交点需补 20 种子。
- 「stable 非单调」只有观察，没有逐帧证据（骨盆高度/倾角时间序列）。

## 下一步

- LAB20 训练期 DR（先定协议；会换 ckpt）。P7（观测加世界系根位置/速度）、P8（脚底接触校验）未开始。
- 若以后要补 px_j110（37 min）可把区间从 20 N·s 压到 10 N·s；当前判断不值得。

## 复现

```bash
cd ~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh
RUN=runs/YYYYMMDD-NNN-lab19b-px100
for rep in 1 2 3; do
  LAB12_TAG=lab19b-px100 LAB12_CELLS="px_j100"     LAB12_SEEDS="101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120"     LAB12_RUN_DIR="$RUN/rep$rep" bash scripts/lab12_recovery.sh run
done
environment/.venv/bin/python scripts/lab16_report.py --dir "$RUN"   --rep rep1="$RUN/rep1" --rep rep2="$RUN/rep2" --rep rep3="$RUN/rep3"   --cells px_j100 --title "LAB19b +x 100 N s"   --figure outputs/figures/lab19b_px100.png   --figure-title "LAB19b: px_j100 (20 seeds x 3 reps)"
```

协议核对（三遍 params.txt 一致）：`sha256: fc8f975b…8ceed9`、`cells: px_j100`、`seeds 101–120`、`push: 8.0 s +0.2 s`、`termination: OFF`。
