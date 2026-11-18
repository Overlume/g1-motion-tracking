# Third-party notices

本仓库（Overlume/g1-motion-tracking）自己的代码与文档采用 MIT，见 [LICENSE](LICENSE)。
下表列出被**引用**的第三方组件与数据；它们**不随本仓库分发**。

| 组件 | 用途 | 许可 | 是否随本仓库分发 |
| --- | --- | --- | --- |
| LAFAN1（Ubisoft La Forge） | 测试动作来源，原始 BVH | CC BY-NC-ND 4.0 | 否 |
| 由 LAFAN1 重定向得到的 npz | 实验输入动作 | 派生自 CC BY-NC-ND 4.0 | 否 |
| mjlab（mujocolab） | 训练 / 评估框架 | Apache-2.0 | 仅作为依赖安装 |
| MuJoCo、mujoco-warp | 物理仿真 | Apache-2.0 | 依赖 |
| GMR（YanjieZe） | BVH → SMPL 重定向 | MIT | 仅作为依赖安装 |
| Unitree G1 模型与官方预训练权重 | 机器人资产、基线策略 | 见 mjlab 与 Unitree 条款 | 否（scratch/model_49999.pt 未入 Git） |

## 为什么这些文件不在仓库里

- LAFAN1 是 CC BY-NC-ND 4.0：禁止分发衍生作品，重定向产物属于衍生作品。
- 预训练权重约 20 MB/份，归属上游项目，不适合再分发。
- 复现方法见 [docs/REPORT.md](docs/REPORT.md) 第 5 节：仓库给脚本与命令，数据与权重由使用者自行获取。

## 上游引用

若本项目的评估协议或扰动实验对你有用，请一并引用上游：mjlab、GMR、LAFAN1。
