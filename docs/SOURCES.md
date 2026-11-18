# SOURCES

所有链接在 2026-11-11 核验。

## SuperMimic

- 论文：https://arxiv.org/abs/2610.28480 （v2，2026-11-09 online；CoRL 2026）
- 项目页：https://supermimic.github.io/ （含 FAQ、视频、demo）
- 仓库：https://github.com/HybridRobotics/supermimic （Apache-2.0；commit abc124c1bd5976341034668af3d12c087f2f1f55）
- Web demo：https://supermimic.github.io/demo/ （浏览器内世界模型想象，非可下载策略）

## 基线/替代

- mjlab：https://github.com/mujocolab/mjlab （v1.6.0，Apache-2.0；本地 `repos/mjlab_research`）
- mjlab 跟踪文档：`repos/mjlab_research/docs/source/training/motion_imitation.rst`
- BeyondMimic：https://github.com/HybridRobotics/whole_body_tracking （MIT，已归档，官方说明其跟踪实现已迁移到 mjlab；Isaac Lab 2.1 / Isaac Sim 4.5）
- SONIC / GR00T-WholeBodyControl：https://github.com/NVlabs/GR00T-WholeBodyControl （arXiv 2511.07820；训练用大规模 GPU；BONES-SEED 为其数据子集）
- GMR（重定向，MIT）：https://github.com/YanjieZe/GMR ，支持 Unitree G1 29DoF，输入 LAFAN1 BVH / SMPL-X / FBX

## 数据

- BONES-SEED：https://huggingface.co/datasets/bones-studio/seed （gated=auto；142,220 motions / 288h；`g1.tar.gz` 23.5GB）
- 许可：https://bones.studio/info/seed-license （AIDVISORY SP. Z O.O.；Academic User / Qualifying Startup）
- viewer：https://seed-viewer.bones.studio
- LAFAN1：https://github.com/ubisoft/ubisoft-laforge-animation-dataset （BVH）
  - 许可：CC BY-NC-ND 4.0（署名-非商业-禁止演绎）。个人非商业学习可用；重定向结果属于演绎作品，不得公开发布/再分发。
  - 本地副本：data/raw/lafan1/lafan1.zip（144,051,503 B，sha256 ea918082b500a5d158e9d3aa39039df04cd42e25f5c02fe8f7e88e8e9365a977），解压后 77 个 .bvh 位于 data/raw/lafan1/bvh/。
  - BVH 帧率：Frame Time 0.033333（30 fps）；类别含 dance1/2、fight1、fightAndSports1、push1、pushAndFall1、pushAndStumble1、fallAndGetUp1-3、jumps1 等。
- GMR（General Motion Retargeting）v0.2.0：https://github.com/YanjieZe/GMR （MIT，本地 repos/GMR）
  - 支持 bvh_lafan1 到 unitree_g1，机器人模型 assets/unitree_g1/g1_mocap_29dof.xml（29 关节，顺序与 mjlab G1 完全一致）。
  - 依赖 CPU torch + mink + qpsolvers(daqp)，装在项目内 environment/gmr-venv（torch 2.15.0+cpu、mujoco 4.0.1）。

## 本机环境事实

- WSL 版本、GPU、内存、磁盘：见 `docs/STATE.md` 与 `environment/logs/`。
