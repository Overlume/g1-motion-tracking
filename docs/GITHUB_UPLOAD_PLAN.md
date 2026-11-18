# GitHub 上传计划

版本：v1，2026-11-14。目标：项目结束后把`supermimic-dsh`上传到一个 **public** 仓库，
用来证明"我独立做过 G1 动作模仿 + 受控扰动"，并让别人能按 README 重跑。

---

## 1. 合规前提（决定"能传什么"）

| 依赖 | 许可 | 能不能进仓库 |
|---|---|---|
| mjlab_research（HybridRobotics） | Apache-2.0 | **不进仓库**（95 MB 第三方代码）。README 写安装方式 + 固定 commit |
| GMR（YanjieZe） | MIT | **不进仓库**（1.5 GB，含 SMPL-X 相关资产）。README 写安装方式 |
| SuperMimic（HybridRobotics） | Apache-2.0 | 不进仓库；只作对照引用 |
| LAFAN1 BVH（Ubisoft） | **CC BY-NC-ND 4.0** | 原始 BVH 不进仓库（471 MB，且 ND 禁止分发演绎作品 → `data/processed/*.npz` 重定向产物**也不传**） |
| 官方预训练 `model_49999.pt` | 随 mjlab 发布 | 不进仓库（6.8 MB 二进制）；README 写 URL + sha256 |
| BONES-SEED | 专有学术许可 | 未使用（D4） |

> 结论：仓库里**只放自己写的代码、文档、小体积结果（json/csv/png/gif）**。
> 数据和权重一律"给来源 + 给校验 + 给生成脚本"，不随仓库分发。
> 这是 CC BY-NC-ND 下的唯一合规做法：别人拿到的是"怎么生成"，不是生成物。

---

## 2. 上传清单

### 纳入版本控制

| 路径 | 内容 | 体积估计 |
|---|---|---|
| `README.md` | 一页结论 + 快速开始 + 结果表 | <20 KB |
| `AGENTS.md` | 长期规则（可选，展示工程规范） | <10 KB |
| `.gitignore` | 见第 3 节 | <2 KB |
| `docs/` | PROJECT_BRIEF、STATE、PLAN、DECISIONS、SOURCES、ARTIFACT_INDEX、REPORT、COMPLETION_CRITERIA、GITHUB_UPLOAD_PLAN、`learning/*.md`（`handoffs/` 仅本地，不入库） | <1 MB |
| `scripts/` | 全部 .py/.sh（回放、评估、重定向、取景、统计） | <200 KB |
| `configs/` | 扰动协议 JSON、训练配置快照 | <100 KB |
| `runs/*/README.md`、`runs/*/summary.json` | 每次运行的命令 + 指标（**不含** metrics.csv 全量、不含 ckpt） | <2 MB |
| `outputs/figures/` | 曲线图 png | <5 MB |
| `outputs/videos/*/preview.gif`、`frames/*.png` | 小体积证据（gif ≤3 MB/个，png ≤500 KB） | <20 MB |
| `environment/requirements.txt`（新建） | `uv pip freeze` 导出的固定版本 | <20 KB |

### 排除

| 路径 | 原因 |
|---|---|
| `environment/`（`.venv`, `uv-cache`, `python`, `bin/uv`） | 6.3 GB 虚拟环境，平台相关 |
| `repos/` | 第三方仓库（mjlab 95 MB、GMR 1.5 GB） |
| `data/raw/` | LAFAN1 原始 BVH 471 MB + 许可限制 |
| `data/processed/` | 重定向产物（LAFAN1 演绎作品，ND 禁止分发） |
| `scratch/` | 论文 PDF、临时脚本、ckpt |
| `runs/*/metrics.csv`（全量逐帧） | 单文件 ~200 KB × 多次运行，价值低于 summary.json |
| `*.pt` / `*.npz` / `*.mp4` / `*.tar.gz` | 权重、数据、大媒体 |
| `__pycache__/`、`.venv/`、`*.egg-info/` | 构建垃圾 |
| `.vscode/`、`.idea/`、`.DS_Store` | 编辑器 |

> `runs/` 目前整体被忽略。**要改成白名单**（只放行 README/summary），见第 3 节。

---

## 3. `.gitignore` 具体条目（可直接替换现有文件）

```gitignore
# ---- 大体积 / 第三方 / 环境 ----
environment/
repos/
scratch/
tools/
data/raw/
data/processed/
outputs/videos/**/*.mp4

# ---- 二进制与权重 ----
*.pt
*.pth
*.onnx
*.npz
*.npy
*.tar.gz
*.zip
*.pdf

# ---- runs：只放行文字结果 ----
runs/*
!runs/*/
runs/*/*
!runs/*/README.md
!runs/*/summary.json
!runs/*/episode_log.json
!runs/*/config.json
runs/**/metrics.csv

# ---- Python ----
__pycache__/
*.py[cod]
.venv/
*.egg-info/
.pytest_cache/

# ---- 编辑器 / 系统 ----
.vscode/
.idea/
.DS_Store
Thumbs.db

# ---- 密钥（永不提交） ----
.env
*.key
wandb/
```

安装后自检（在仓库根执行）：

```bash
git status --ignored --short | head -50     # 确认 venv/repos/data 都被忽略
git check-ignore -v runs/20261114-001-s3-seed101/summary.json   # 应该"不"被忽略
```

---

## 4. 上传后的目录结构

```
supermimic-dsh/
├── README.md
├── AGENTS.md
├── .gitignore
├── docs/
│   ├── PROJECT_BRIEF.md  STATE.md  PLAN.md  DECISIONS.md
│   ├── SOURCES.md  ARTIFACT_INDEX.md  REPORT.md
│   ├── COMPLETION_CRITERIA.md  GITHUB_UPLOAD_PLAN.md
│   ├── learning/   LAB01..LAB05
├── scripts/
│   ├── setup_env.sh
│   ├── replay_policy.py  eval_tracking.py  camera_fit.py  motion_stats.py
│   ├── gmr_bvh_to_csv.py  csv_to_npz_local.py  convert_demo_motion.py
│   └── smoke_env.py  smoke_mem.sh  extract_frames.sh
├── configs/
│   ├── perturb_protocol_v1.json
│   └── train_m4_snapshot.json
├── environment/
│   └── requirements.txt
├── runs/
│   ├── 20261114-001-s3-seed101/{README.md,summary.json}
│   └── ...
└── outputs/
    ├── figures/*.png
    └── videos/*/preview.gif + frames/*.png
```

---

## 5. 提交划分（建议 8 次提交，按"能独立讲清一件事"划分）

| # | 提交信息 | 内容 |
|---|---|---|
| 1 | `chore: bootstrap project skeleton and env setup script` | `.gitignore`、`AGENTS.md`、`docs/{PROJECT_BRIEF,STATE,PLAN,DECISIONS,SOURCES}`、`scripts/setup_env.sh`、`environment/requirements.txt` |
| 2 | `feat(eval): replay official G1 checkpoint and quantify tracking error` | `scripts/replay_policy.py`、`eval_tracking.py`、`camera_fit.py`、`motion_stats.py` + 对应 `runs/` 摘要 |
| 3 | `feat(retarget): LAFAN1 BVH -> GMR -> mjlab npz pipeline` | `gmr_bvh_to_csv.py`、`csv_to_npz_local.py`、`convert_demo_motion.py` + `docs` 说明 |
| 4 | `docs: explain episode reset vs video repetition (LAB03)` | `docs/learning/LAB01..LAB03`、`DECISIONS` D11 |
| 5 | `feat(train): smoke training + single-parameter A/B` | M3 脚本与 `runs/*-train-*` |
| 6 | `feat(m4): fine-tune on fight clip and compare with official policy` | M4 配置、`runs/*-m4-*`、对比图 |
| 7 | `feat(m5): controlled external-force perturbation protocol` | `configs/perturb_protocol_v1.json`、`runs/*-m5-*`、恢复曲线图 |
| 8 | `docs: final report, results tables and reproduction guide` | `docs/REPORT.md`、README 结果表、`ARTIFACT_INDEX`、最后交接 |

提交信息风格：`type(scope): 一行祈使句`；正文写"为什么"和"怎么验证"，例如：

```
feat(m5): add torso push protocol at 50/100/150 N

Ground fixed, force applied in world +x on torso_link for 0.2 s at t=5 s.
5 seeds per level, identical to baseline eval settings.
Verification: runs/20261116-00*/summary.json
```

---

## 6. 上传前检查清单

- [ ] `grep -rn "C:\\\\" scripts/ docs/` 无 Windows 绝对路径（文档示例除外）
- [ ] `grep -rn "/home/<user>" scripts/` 无写死路径；脚本用相对路径或 `--flag`
- [ ] `grep -rniE "api[_-]?key|secret|token|wandb.*(key|login)" . --exclude-dir={environment,repos,data}` 无命中
- [ ] `find . -size +5M -not -path "./environment/*" -not -path "./repos/*" -not -path "./data/*" -not -path "./scratch/*"` 无意外大文件
- [ ] `git status --ignored` 确认 venv/repos/data 均被忽略
- [ ] `*.pt`、`*.npz`、`*.mp4` 均未入库（`git ls-files | grep -E "\.(pt|npz|mp4)$"` 应为空）
- [ ] `runs/*/summary.json` 中的绝对路径已改为相对或删除
- [ ] `README.md` 包含：环境要求、安装命令、回放命令、权重 URL+sha256、数据来源与许可、结果表
- [ ] `environment/requirements.txt` 与 `environment/logs/setup_env.log` 的版本一致（torch 2.15.0+cu132 等）
- [ ] 第三方引用：mjlab commit、GMR 版本、LAFAN1 许可声明写在 README `Acknowledgements/License` 段
- [ ] 仓库根加 `LICENSE`（自己的代码建议 MIT）+ NOTICE 说明第三方
- [ ] 在临时目录 `git clone` 一次，按 README 走通回放（3 条命令内出视频）
- [ ] GitHub 仓库描述 + topics（`humanoid`、`mujoco`、`motion-tracking`、`unitree-g1`）
- [ ] 视频大文件走 GitHub Release 附件或外部链接，不进 git 历史

---

## 7. 风险与备选

| 风险 | 影响 | 备选 |
|---|---|---|
| 想放 960x540 mp4 展示 | 单文件 6 MB，多次提交会撑大历史 | 只放 `preview.gif`（≤3 MB）或 Release 附件 |
| 别人 clone 后无法复现 | 缺 mjlab/GMR 固定版本 | README 固定 commit + `requirements.txt` |
| 派生数据合规 | 传 npz 违反 ND | 只传生成脚本 + 原始 BVH 的 sha256 + 选择窗口的帧号 |
| 检查点不可得 | 官方 URL 失效 | 记录 sha256，允许用户自行下载；不镜像分发 |


## 8. 本仓库实测状态（2026-11-14 14:15）

- git 仓库已初始化（branch `master`，**尚无任何 commit**）。
- `environment/requirements.txt` 已生成：`uv pip freeze`，142 行，torch 2.15.0 / mujoco 3.16.0 / mujoco-warp 3.16.0 / warp-lang 1.19.0 / rsl-rl-lib 5.5.2。注意 `mjlab` 是 editable 安装（`-e repos/mjlab_research`），freeze 里不出现，README 要单独写。
- `.gitignore` 已换成第 3 节的 `runs` 白名单版本，并实测验证：
  - `git add -A --dry-run` 会纳入 **92 个文件 / 约 18.0 MB**（2026-11-14 15:15 复核）；
  - `runs/` 下只放行 `README.md`、`summary.json`、`episode_log.json`、`config.json`；
  - `*.pt`、`*.onnx`、`*.npz`、`*.mp4`、`environment/`、`repos/`、`data/` 全部被忽略。
- 已删除被取代的调试产物：`outputs/videos/model_49999/960x540/cam_fit_test`（v1 公式，相机过远）、`cam_fit_v2`（中间版本）；保留 `cam_fit_final` 作为最终证据。
- 当前体积热点：`environment/` 15 GB、`repos/` 1.6 GB、`data/` 475 MB、`outputs/` 44 MB、`runs/` 64 MB（含 ckpt/onnx，已被忽略）。
- 仍未做：`docs/REPORT.md`（M6）、`LICENSE`、`configs/` 目录（M5 才会产生）。
