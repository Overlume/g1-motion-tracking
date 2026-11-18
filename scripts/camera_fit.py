"""相机取景工具：把整个 env 网格装进离屏渲染画面。

背景（实测 mjlab v1.6.0）：
- repos/mjlab_research/src/mjlab/terrains/terrain_entity.py:368-381
  俯视网格布局：num_rows=ceil(N/int(sqrt(N)))，num_cols=ceil(N/num_rows)，
  x = -(ii-(rows-1)/2)*env_spacing，y = (jj-(cols-1)/2)*env_spacing，默认 spacing=2.0。
- repos/mjlab_research/src/mjlab/viewer/offscreen_renderer.py:186-217
  _get_env_ids() 只渲染 cfg.env_idx + 最近 max_extra_envs 个邻居（默认 2）
  => 4 个 env 时只画 3 台机器人，第 4 台根本没被提交给 mjv_addGeoms。
- 同一文件 232-260：origin_type=AUTO 时相机是 TRACKING 模式，跟着 env_idx 的 pelvis 走，
  cfg.lookat 被忽略；env 之间会随着漂移散开，固定 distance 装不下。

取景公式（v2，2026-11-14 收紧）：
  水平方向只受 hfov 约束：d_x = half_x / tan(hfov/2)
  竖直方向受 vfov 约束，地面深度按仰角投影：half_v = half_y*sin(|elev|) + robot_h/2
                                   d_v = half_v / tan(vfov/2)
  distance = margin * max(d_x, d_v)（不小于 min_distance）
v1 用 min(vfov,hfov) 统一算，导致 16:9 时把水平余量浪费掉、相机拉太远（4 env 实测 6.37 m，
机器人只占画面高度 ~20%）。

本模块只做纯计算（无 mujoco 依赖），便于单测与复用。
"""

import math


def grid_origins(num_envs: int, env_spacing: float = 2.0):
    """复现 terrain_entity._compute_env_origins_grid 的布局，返回 [(x, y), ...]。"""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    num_rows = math.ceil(num_envs / int(math.sqrt(num_envs)))
    num_cols = math.ceil(num_envs / num_rows)
    out = []
    for i in range(num_rows):
        for j in range(num_cols):
            # 与 torch.meshgrid(indexing="ij") 展平顺序一致
            x = -(i - (num_rows - 1) / 2.0) * env_spacing
            y = (j - (num_cols - 1) / 2.0) * env_spacing
            out.append((x, y))
    return out[:num_envs]


def fit_camera(
    origins,
    width: int,
    height: int,
    fovy_deg: float = 45.0,
    margin: float = 1.25,
    robot_radius: float = 0.8,
    robot_height: float = 1.3,
    lookat_z: float = 0.8,
    elevation: float = -25.0,
    azimuth: float = 90.0,
    min_distance: float = 4.6,
):
    """给出能把所有 env 原点（含机器人尺寸）装进画面的自由相机参数。"""
    xs = [p[0] for p in origins]
    ys = [p[1] for p in origins]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    half_x = (max(xs) - min(xs)) / 2.0 + robot_radius
    half_y = (max(ys) - min(ys)) / 2.0 + robot_radius

    vfov = math.radians(fovy_deg)
    hfov = 2.0 * math.atan(math.tan(vfov / 2.0) * width / height)
    d_x = half_x / math.tan(hfov / 2.0)
    half_v = half_y * math.sin(math.radians(abs(elevation))) + robot_height / 2.0
    d_v = half_v / math.tan(vfov / 2.0)

    distance = max(min_distance, margin * max(d_x, d_v))
    return dict(
        lookat=(cx, cy, lookat_z),
        distance=round(distance, 3),
        elevation=elevation,
        azimuth=azimuth,
        half_x=round(half_x, 3),
        half_y=round(half_y, 3),
    )


if __name__ == "__main__":
    for n in (1, 2, 4, 8, 16):
        o = grid_origins(n)
        c = fit_camera(o, 960, 540)
        print(
            "num_envs=%2d origins=%s distance=%.2f lookat=(%.2f,%.2f)"
            % (n, len(o), c["distance"], c["lookat"][0], c["lookat"][1])
        )
