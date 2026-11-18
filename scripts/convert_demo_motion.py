"""把 mjlab 旧格式 demo npz 转换为 v1.6.0 MotionLoader 期望的格式。

旧格式字段: fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w
新格式字段: root_pos_w, root_quat_w(wxyz), root_lin_vel_w, root_ang_vel_w,
            joint_pos, joint_vel, joint_names

假设: body 维度第 0 个是 pelvis（G1 根链接），body 顺序与 MuJoCo 模型一致。
证据: 转换后由 MotionLoader 校验 joint_names 是否等于机器人关节名。
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

src = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch/demo_motion.npz")
dst = Path(sys.argv[2] if len(sys.argv) > 2 else "data/processed/lafan_dance1_subject1.npz")

z = dict(np.load(src))
model = mujoco.MjModel.from_xml_path(str(G1_XML))
body_names = [model.body(i).name for i in range(model.nbody)][1:]
joint_names = [model.joint(i).name for i in range(1, model.njnt)]
print("model bodies:", len(body_names), "joints:", len(joint_names))
print("pelvis index in model bodies:", body_names.index("pelvis"))
print("old npz bodies:", z["body_pos_w"].shape[1], "joints:", z["joint_pos"].shape[1])
print("first body names:", body_names[:5])
print("joint names:", joint_names)

assert z["joint_pos"].shape[1] == len(joint_names), "joint count mismatch"
assert z["body_pos_w"].shape[1] <= len(body_names)

root = body_names.index("pelvis")
out = {
    "root_pos_w": z["body_pos_w"][:, root].astype(np.float32),
    "root_quat_w": z["body_quat_w"][:, root].astype(np.float32),
    "root_lin_vel_w": z["body_lin_vel_w"][:, root].astype(np.float32),
    "root_ang_vel_w": z["body_ang_vel_w"][:, root].astype(np.float32),
    "joint_pos": z["joint_pos"].astype(np.float32),
    "joint_vel": z["joint_vel"].astype(np.float32),
    "joint_names": np.array(joint_names),
}
dst.parent.mkdir(parents=True, exist_ok=True)
np.savez(dst, **out)
print("wrote", dst, "frames:", out["joint_pos"].shape[0])
print("root_pos range:", out["root_pos_w"].min(0), out["root_pos_w"].max(0))
