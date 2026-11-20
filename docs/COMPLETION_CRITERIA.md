# 项目结束边界（Definition of Done）

版本：v1，2026-11-14。适用于 `supermimic-dsh` 项目。

判定原则：**每一条都必须能用"文件在不在 / 命令能不能重跑 / 数字落不落在给定区间"来验证**。
不写"做得差不多"、"效果良好"这类无法验收的措辞。

---

## 1. 项目目标（一句话）

在纯仿真中用 mjlab（MuJoCo Warp）让 Unitree G1 双足人形（29 DoF）模仿参考动作，
并完成一组**受控外力扰动**测试；所有结论基于本地实测（RTX 3060 6 GB / WSL2），
产物可追溯、命令可复现。

---

## 2. 必跑实验（缺一不可）

| 编号 | 实验 | 完成判据（可验收） | 证据位置 |
|---|---|---|---|
| M3-a | 冒烟训练 | 从零（或从官方 ckpt）连续训练 ≥ 100 iter，loss 为有限值，产出可被 `runner.load` 读回的 checkpoint；记录 iter 耗时、CPU 内存、显存峰值 | `runs/YYYYMMDD-NNN-train-smoke/` 的 `summary.json` + `train.log` |
| M3-b | 单参数对照 | 用户亲手改 **1 个**有明确物理含义的参数（reward 权重 / num_envs / 动作片段之一），**先写预测再跑**，A/B 使用同一动作、同一种子集 | `docs/learning/LAB*.md` 的"预测 → 实测 → 解释"三段 |
| M4 | 武术动作跟踪 | 在 `fight1` 重定向片段上得到**自己训练**的策略；与官方预训练在同一动作、N≥5 个种子上对比；报中位数 + 范围，**不得引用单次 fall_step**（D11） | `runs/*-m4-*/` 的 `summary.json` + 960x540 全片 mp4 |
| M5 | 受控扰动 | 固定地面；力施加在 `torso_link`，世界系 ±x/±y；≥3 档幅值 × N≥5 种子；指标见第 3 节；至少做 **1 次改进**（训练时加扰动 / 调 reward）并在**完全相同协议**下复测 | `configs/perturb_*.json` + `runs/*-m5-*/` + `outputs/figures/` 对比图 |
| M6 | 汇总交付 | `docs/REPORT.md`（数字表 + 图 + 失败案例）＋ 一页 README 结论 ＋ GitHub 上传完成 | 仓库根 `README.md`、`docs/REPORT.md` |

时间预算参考（实测外推，非承诺）：M3 ≈ 1–2 个工作日，M4 ≈ 3–5 天（含调参），M5 ≈ 3–5 天。

---

## 3. 必须有、且必须自洽的结果

1. **评估指标表**（每个策略 × 每个动作）：`track_completion`、`fall`、`mpkpe_*`、`r_mpkpe_*`、
   `joint_rmse_*`、`root_xy_err`、`drift_rate`、`slip_*`、`pelvis_z_*`、`tilt_*`。
   单位写清（cm / rad / deg / s），坐标对齐方式写清（根锚定 vs 世界系）。
2. **可复现性声明**：每个数字注明种子与是否开启动域随机化（`--seed` / `--keep-dr`）。
   平稳动作可报单次值并注明约 0.5% 抖动；fight 类只报分布（D11）。
3. **区分三种结果**：官方预训练 / 自己训练 / 修改后。不得混写在同一列。
4. **保留失败案例**：至少 1 个失败视频 + 失败原因分析（例如 S3 在 step 380–781 之间失稳）。
5. **每个实验一条可复制的完整命令**（含环境变量），写在对应 `runs/*/README.md`。

---

## 4. 必须关闭的问题清单

| 问题 | 关闭标准 | 状态 |
|---|---|---|
| 4 env 只显示 3 台机器人 | `--num-envs 4 --width 960 --height 540` 重跑，画面内 4 台完整入镜，`episode_log.json` 记录 `camera.max_extra_envs=3` | 2026-11-14 修复（`scripts/camera_fit.py`），待用户按原命令复验 |
| 录像"只有 3 秒有效、之后重复" | 原因写进文档（termination → reset → `sampling_mode=start`）；`episode_log.json` 逐条记录 reset 步号与终止项 | LAB03 已解释，脚本已记录 |
| 评估被 reset 污染（completion 假 1.0） | `eval_tracking.py` v2 按 episode 分段统计，报告 reset 次数 | 已关闭（runs/20261113-006 起） |
| 同种子结果不一致 | D11 写明抖动幅度与报告口径；脚本记录 seed / keep_startup_dr | 已关闭（D11） |
| 指标定义含糊 | `runs/*/README.md` 写清公式与阈值（跌倒 0.15 m×5 步、脚滑支撑相条件） | 已关闭 |
| 许可合规 | LAFAN1(BY-NC-ND) 派生数据不入库；BONES-SEED 不使用；第三方仓库不入库；仓库内写清来源 | 上传前按 `docs/GITHUB_UPLOAD_PLAN.md` 检查 |
| 硬编码路径 | 仓库内无 `C:\\`、无 `/home/<user>` 绝对路径（除文档示例），脚本用相对路径 | 上传前 grep 检查 |

---

## 5. 明确不在本项目范围内

| 不做的事 | 理由 |
|---|---|
| 真机部署 / sim2real | 无 G1 硬件；项目目标是纯仿真 |
| 复现 SuperMimic 论文规模训练（4096 env、H100 3 h、12.9 万动作） | 单卡 6 GB，且单动作用不上"跨动作摊销"卖点（D6） |
| 下载 / 使用 BONES-SEED（23.5 GB，专有学术许可） | 用户非学术身份未确认，授权范围不符（D4） |
| 租用云 GPU | 本地实测可跑，无兼容性/资源证据支持上云（成本不必要） |
| 多 GPU / 分布式训练 | 规模不需要 |
| 语言条件控制、VLA、世界模型自研 | 超出"动作模仿 + 扰动"目标 |
| 重新训练通用跟踪策略到论文级指标 | 官方预训练已覆盖；自训练只做片段级对照 |
| 角色绑定/动捕硬件、Unity/Isaac 可视化 | 用 mjlab + MuJoCo 渲染即可 |
| 公开分发 LAFAN1 派生动作数据 | CC BY-NC-ND 4.0 禁止分发演绎作品 |

新增需求若落入上表，先记入 `docs/PLAN.md` 的"候选扩展"，不进入本次验收。

---

## 6. 结束时的交付物清单（打勾即完成）

- [x] `README.md`：一页结论 + 快速开始（3 条命令内跑起回放）——已交付（2026-11-20 结果表更新到 LAB16）
- [x] `docs/REPORT.md`：目标、方法、指标表、图、失败案例、局限——已交付（含 §2.5.5–§2.5.7）
- [x] `docs/STATE.md` / `PLAN.md` / `DECISIONS.md` / `SOURCES.md` / `ARTIFACT_INDEX.md` 与最终状态一致——2026-11-20 同步（D29 起每次任务后按 HANDOFF §8 走）
- [x] `docs/learning/*.md`：至少 5 个 LAB 记录（含用户预测与实测对比）——现有 LAB01–LAB16
- [ ] `runs/`：M3–M5 每次运行都有 `README.md` + `summary.json`——**部分**：50 个 run 目录 / 38 份 README / 21 份 summary.json；缺 README 的 12 个是 2026-11-13/14 的 S3 分布评估（D11 的 N=5，数据在 summary.json）
- [ ] `outputs/videos/`：≥1 个 960x540 完整动作视频、≥1 个扰动对比视频——**部分**：动作/取景视频有（`outputs/videos/model_49999/960x540/`），**扰动对比视频缺**
- [x] `outputs/figures/`：≥2 张图（跟踪误差曲线、扰动恢复对比）——7 张（含 m5_perturbation、lab16_flipmatrix）
- [ ] `configs/`：扰动协议 JSON、训练配置快照——**部分**：只有 `perturb_protocol_v1.json`，**训练配置快照缺**
- [ ] GitHub 仓库 public，`git clone` 后按 README 能跑通回放——仓库已 public；**clone 复验未做**（§7 第 3 条，待用户）
- [x] 交接记录：`docs/handoffs/HANDOFF.md` 唯一一份（**仅本地保留，不入库**；2026-11-20 起原地持续更新，D29）

### 6.1 现状核对（2026-11-20）

- 上面 4 个未打勾项是**已知缺口**，不是文档遗漏；补法：给 12 个旧 run 补 README（或注明数据在 summary.json）、录一段扰动前后对比视频、存一份训练配置快照、由用户在新目录 `git clone` 复验回放。
- 另有一项**必跑实验的缺口**：§2 的 M5 要求"至少 1 次改进并同协议复测"，目前**未做**（REPORT §7 未来工作 1 = 训练期 DR，对应 HANDOFF §4 的 P6）；M5 只完成了评估侧。
- 结论：对外交付（报告 + README + 公开仓库）已完成；"完成定义"里仍缺 M5 改进实验与 4 个小交付物。

---

## 7. 验收方式

1. 用户用**原始命令**重跑两条：4 env 取景、S3 评估；结果与 `docs/REPORT.md` 表格一致。
2. 用户抽查 1 个 `runs/*/README.md`，能照着命令重现该次运行。
3. `git clone` 到新目录，按 README 装环境 + 跑回放，3 条命令内出视频。
4. 上述 3 条通过 = 项目结束；未通过项回到第 4 节清单继续关闭。
