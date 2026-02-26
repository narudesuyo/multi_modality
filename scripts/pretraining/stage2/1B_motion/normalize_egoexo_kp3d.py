#!/usr/bin/env python3
"""Normalize EgoExo4D kp3d in-place.

Steps:
  1) Z-up -> Y-up
  2) floor (min Y = 0)
  3) root center in XZ (first frame root)
  4) face Z+ (first frame across)
"""

import argparse
import os
import numpy as np


FACE_JOINTS = [2, 1, 17, 16]  # r_hip, l_hip, r_shoulder, l_shoulder
ROOT_IDX = 0


def zup_to_yup(kp):
    kp_yup = kp.copy()
    kp_yup[..., 1] = kp[..., 2]
    kp_yup[..., 2] = -kp[..., 1]
    return kp_yup


def _norm(v, eps=1e-8):
    n = np.linalg.norm(v)
    return v / (n + eps)


def rotation_matrix_from_a_to_b(a, b, eps=1e-8):
    a = _norm(a, eps)
    b = _norm(b, eps)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < eps:
        return np.eye(3, dtype=np.float32)
    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + vx + (vx @ vx) * ((1.0 - c) / (s ** 2 + eps))
    return R


def normalize_kp3d(kp):
    kp = zup_to_yup(kp)

    # floor
    floor_h = kp[..., 1].min()
    kp[..., 1] -= floor_h

    # root center XZ by first frame root
    root0 = kp[0, ROOT_IDX].copy()
    kp[..., 0] -= root0[0]
    kp[..., 2] -= root0[2]

    # face Z+ by first frame across
    r_hip, l_hip, sdr_r, sdr_l = FACE_JOINTS
    across = (kp[0, r_hip] - kp[0, l_hip]) + (kp[0, sdr_r] - kp[0, sdr_l])
    across = _norm(across)
    forward = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float32), across)
    forward = _norm(forward)
    target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    R = rotation_matrix_from_a_to_b(forward, target)
    kp = kp @ R.T
    return kp.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion-dir", required=True, help="Root motion_atomic directory")
    ap.add_argument("--dry-run", action="store_true", help="List files without writing")
    args = ap.parse_args()

    motion_dir = args.motion_dir
    npy_files = []
    for root, _dirs, files in os.walk(motion_dir):
        for fn in files:
            if fn.endswith("_kp3d.npy"):
                npy_files.append(os.path.join(root, fn))

    print(f"Found {len(npy_files)} kp3d files under {motion_dir}")
    for p in npy_files:
        if args.dry_run:
            print(p)
            continue
        kp = np.load(p).astype(np.float32)
        kp_n = normalize_kp3d(kp)
        np.save(p, kp_n)


if __name__ == "__main__":
    main()
