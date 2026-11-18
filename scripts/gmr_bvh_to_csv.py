"""Retarget a LAFAN1 BVH clip to Unitree G1 with GMR, save .pkl and .csv.

Run with the GMR venv (environment/gmr-venv), e.g.:
  environment/gmr-venv/bin/python scripts/gmr_bvh_to_csv.py \
      --bvh data/raw/lafan1/bvh/fight1_subject2.bvh \
      --out-pkl data/processed/gmr/fight1_subject2.pkl \
      --out-csv data/processed/gmr/fight1_subject2.csv
"""
import argparse
import os
import pickle

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_bvh_file


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--bvh", required=True)
  ap.add_argument("--format", default="lafan1")
  ap.add_argument("--robot", default="unitree_g1")
  ap.add_argument("--fps", type=int, default=30)
  ap.add_argument("--out-pkl", required=True)
  ap.add_argument("--out-csv", default=None)
  ap.add_argument("--start-frame", type=int, default=0,
                  help="first source frame to retarget")
  ap.add_argument("--max-frames", type=int, default=0,
                  help="0 = to the end of the clip")
  ap.add_argument("--verbose", action="store_true")
  args = ap.parse_args()

  frames, human_height = load_bvh_file(args.bvh, format=args.format)
  print(f"[gmr] bvh={args.bvh} frames={len(frames)} human_height={human_height:.3f} m")

  retargeter = GMR(
    src_human=f"bvh_{args.format}",
    tgt_robot=args.robot,
    actual_human_height=human_height,
    verbose=args.verbose,
  )
  print("[gmr] robot dof order:", retargeter.robot_dof_names)

  start = max(0, args.start_frame)
  end = len(frames) if args.max_frames <= 0 else min(len(frames), start + args.max_frames)
  print(f"[gmr] retargeting source frames [{start}, {end}) "
        f"= {(end - start) / args.fps:.2f} s at {args.fps} fps")
  qpos_list = []
  for i in range(start, end):
    qpos_list.append(np.asarray(retargeter.retarget(frames[i]), dtype=np.float32))
    if (i - start) % 200 == 0:
      print(f"[gmr] retarget {i - start}/{end - start}", flush=True)
  qpos = np.stack(qpos_list)
  print("[gmr] qpos shape", qpos.shape)

  root_pos = qpos[:, :3]
  root_rot_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]]  # GMR gives wxyz, csv wants xyzw
  dof_pos = qpos[:, 7:]

  os.makedirs(os.path.dirname(os.path.abspath(args.out_pkl)), exist_ok=True)
  with open(args.out_pkl, "wb") as f:
    pickle.dump({
      "fps": args.fps,
      "root_pos": root_pos,
      "root_rot": root_rot_xyzw,
      "dof_pos": dof_pos,
      "local_body_pos": None,
      "link_body_list": None,
    }, f)
  print("[gmr] wrote", args.out_pkl)

  if args.out_csv:
    motion = np.zeros((qpos.shape[0], dof_pos.shape[1] + 7), dtype=np.float32)
    motion[:, :3] = root_pos
    motion[:, 3:7] = root_rot_xyzw
    motion[:, 7:] = dof_pos
    np.savetxt(args.out_csv, motion, delimiter=",")
    print("[gmr] wrote", args.out_csv, motion.shape)


if __name__ == "__main__":
  main()
