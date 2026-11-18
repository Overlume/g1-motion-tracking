"""CSV (pos3, quat_xyzw4, 29 dof) -> mjlab tracking npz, without W&B.

Mirrors mjlab/scripts/csv_to_npz.py but saves the npz locally.

  environment/.venv/bin/python scripts/csv_to_npz_local.py \
      --input-file data/processed/gmr/fight1_subject2.csv \
      --output-file data/processed/fight1_subject2.npz --input-fps 30 --output-fps 50
"""
import argparse

import numpy as np

from mjlab.scripts.csv_to_npz import MotionLoader


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--input-file", required=True)
  ap.add_argument("--output-file", required=True)
  ap.add_argument("--input-fps", type=float, default=30.0)
  ap.add_argument("--output-fps", type=float, default=50.0)
  ap.add_argument("--device", default="cpu")
  args = ap.parse_args()

  motion = MotionLoader(
    motion_file=args.input_file,
    input_fps=args.input_fps,
    output_fps=args.output_fps,
    device=args.device,
  )
  arrays = motion.arrays()
  for k, v in arrays.items():
    print(f"[npz] {k}: shape={np.asarray(v).shape}")
  np.savez(args.output_file, **arrays)
  print("[npz] wrote", args.output_file)


if __name__ == "__main__":
  main()
