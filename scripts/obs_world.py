"""LAB21：在 actor 观测里追加「世界系根位置 / 根速度」（P7 假设）。

动机：M4 的 actor 只看到与参考相关的相对量（motion_anchor_pos_b / ori_b）+ 本体感知 + IMU。
它没有任何与参考无关的绝对位姿信号，因此「自己漂走了」与「参考本来就在动」在输入上是同一种模式。
P7 假设：追加世界系 root 位置/速度后，策略能拿到一个独立的绝对位姿基准，
从而在未见窗口与受推后更好地判断「我在哪、我在往哪去」。

实现约定（训练与评估必须一致，否则 ckpt 与观测维度对不上）：
- 两个新项固定**追加在 actor.terms 末尾**（顺序 = 拼接顺序 = 输入维度顺序，新增 6 维在最后）；
- 位置 scale 0.5、速度 scale 0.5（把值域压到与 joint_pos / base_lin_vel 同量级）；
- 噪声 ±0.1（位置 m、速度 m/s），只在训练期生效（play 模式 enable_corruption=False）；
- 只改 actor，不动 critic（critic 本来就有 body_pos / body_ori，且不受策略输入影响）。
"""

import torch
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

POS_SCALE = 0.5
VEL_SCALE = 0.5
NOISE = 0.1
EXTRA_DIMS = 6  # 3 位置 + 3 速度
TERM_NAMES = ("root_pos_w", "root_vel_w")


def root_pos_w(env) -> torch.Tensor:
  """机器人根（pelvis）世界坐标，形状 (N, 3)。"""
  return env.scene["robot"].data.root_link_pos_w


def root_vel_w(env) -> torch.Tensor:
  """机器人根（pelvis）世界系线速度，形状 (N, 3)。"""
  return env.scene["robot"].data.root_link_lin_vel_w


def add_world_root_obs(cfg, pos_scale: float = POS_SCALE, vel_scale: float = VEL_SCALE,
                       noise: float = NOISE):
  """给 cfg.observations["actor"] 追加 root_pos_w / root_vel_w 两项（原地修改并返回 cfg）。"""
  terms = cfg.observations["actor"].terms
  dup = [n for n in TERM_NAMES if n in terms]
  if dup:
    raise RuntimeError("actor 观测里已存在 %s，不要重复追加" % dup)
  terms["root_pos_w"] = ObservationTermCfg(
    func=root_pos_w, scale=pos_scale, noise=Unoise(n_min=-noise, n_max=noise)
  )
  terms["root_vel_w"] = ObservationTermCfg(
    func=root_vel_w, scale=vel_scale, noise=Unoise(n_min=-noise, n_max=noise)
  )
  return cfg
