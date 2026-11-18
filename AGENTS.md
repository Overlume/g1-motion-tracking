# AGENTS.md — 长期规则（给未来的智能体/自己）


## 工作根目录

唯一根目录：`<workspace>`，本机为 `~/embodiedAI/supermimic-replica/ai-ws/supermimic-dsh`
（Windows：`\\wsl.localhost\Ubuntu-22.04\home\<user>\embodiedAI\supermimic-replica\ai-ws\supermimic-dsh`）。
所有产物写在这里。目录外安装、共享环境、全局缓存（`~/.local`、`~/.cache/uv`、系统 Python）必须先说明理由并获用户同意；`environment/` 内已配置 `UV_CACHE_DIR`、`UV_PYTHON_INSTALL_DIR`，优先使用。

## 机器人事实（不要凭印象改）

- 目标机器人：Unitree G1 双足人形（29 DoF），不是 G1D 轮式。
- 关节名/顺序/限位/执行器以 mjlab 的 G1 资产与 MuJoCo 模型为准；改动机器人 XML 前先记录来源与版本。

## 数据与许可

- BONES-SEED（bones-studio/seed）：专有许可，仅限 Academic User（非营利学术、成果公开）或年收入 <100 万美元的 Qualifying Startup；个人非学术用途不在授权范围。使用前必须确认用户身份符合条款。
- 第三方仓库/数据/检查点不得重新分发到本仓库；大文件（权重、tar、视频原始素材）不入 Git。

## 实验可追溯

- 每次训练/评估建 `runs/YYYYMMDD-NNN-<name>/`，至少含：目的与假设、运行位置（本地/WSL）、git commit 与未提交 diff、环境版本、数据与动作标识、配置快照、命令、种子、日志/指标/检查点/视频路径、结果与下一步。
- 一次只改一个主要因素；对比实验用相同动作、种子、扰动条件。
- 区分"官方预训练 / 自己训练 / 修改后结果"，不得混写。

## 诚实性

- 不声称运行过未运行的命令；工具缺失、网络受限如实报告。
- 不编造指标；保留失败案例。
- 结论必须给证据路径（文件行号、日志、URL）。

## 交接

- 用户说"准备交接/换窗口/收尾"时：到安全停顿点 → 更新 `docs/STATE.md`、`docs/PLAN.md`、`docs/ARTIFACT_INDEX.md` → 生成 `docs/handoffs/YYYYMMDD-HHMM.md`。

- `docs/handoffs/` 只保留在本地，不提交；`.gitignore` 已排除该目录（2026-11-18 用户确认）。写完交接后不要 `git add` 它。
- 新窗口先读 AGENTS / README / PROJECT_BRIEF / STATE / PLAN + 最近交接，做最小只读核对；发现不一致先指出再动手。
