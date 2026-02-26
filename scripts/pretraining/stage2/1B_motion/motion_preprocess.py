"""Common motion preprocess for EgoExo4D + Assembly101.

- Unify to Y-up (auto-detect up axis)
- Face +Z forward (yaw rotation based on hips/shoulders)
"""

from __future__ import annotations

import numpy as np


FOOT_IDS = [7, 8, 10, 11]          # L_ankle, R_ankle, L_foot, R_foot (52-joint body ids)
FACE_JOINTS = [2, 1, 17, 16]       # r_hip, l_hip, r_shoulder, l_shoulder

_AXES = {
    "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
}


def _norm_np(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)


def axis_stats(kp3d: np.ndarray, joint_ids: list[int] = FOOT_IDS) -> dict:
    """Stats on foot min heights to guess up axis (fallback)."""
    p = kp3d[:, joint_ids, :]
    stats = {}
    for ax, c in zip(["x", "y", "z"], [0, 1, 2]):
        v = p[..., c]
        min_t = v.min(axis=1)
        stats[ax] = {
            "mean_min": float(min_t.mean()),
            "std_min": float(min_t.std()),
            "range_min": float(min_t.max() - min_t.min()),
        }
    return stats


def pick_up_axis(stats: dict) -> str:
    # Stability first (std, range), then mean_min
    return sorted(
        stats.keys(),
        key=lambda a: (stats[a]["std_min"], stats[a]["range_min"], stats[a]["mean_min"]),
    )[0]


def pick_up_axis_from_geom(
    kp3d: np.ndarray,
    face_joint_indx: list[int] = FACE_JOINTS,
    foot_ids: list[int] = FOOT_IDS,
    eps: float = 1e-8,
) -> tuple[str | None, dict, dict]:
    """Geometry-based up-axis detection using hips/shoulders + feet."""
    rhip, lhip, rsdr, lsdr = face_joint_indx
    lank, rank, lfoot, rfoot = foot_ids

    across1 = kp3d[:, rhip] - kp3d[:, lhip]
    across2 = kp3d[:, rsdr] - kp3d[:, lsdr]
    v_across = _norm_np(across1 + across2, eps)

    vL = kp3d[:, lfoot] - kp3d[:, lank]
    vR = kp3d[:, rfoot] - kp3d[:, rank]
    v_foot = _norm_np(vL, eps) + _norm_np(vR, eps)
    v_foot = _norm_np(v_foot, eps)

    n = np.cross(v_foot, v_across)
    w = np.linalg.norm(n, axis=-1)
    n = _norm_np(n, eps)

    good = w > (0.05 * np.median(w[w > eps]) if np.any(w > eps) else 0.0)
    if good.sum() < 5:
        good = w > eps
    if good.sum() < 5:
        return None, {k: 0.0 for k in _AXES}, {"good_frames": int(good.sum())}

    ng = n[good]
    wg = w[good]

    scores = {k: float(np.sum(wg * np.abs(ng @ a))) for k, a in _AXES.items()}
    up_axis = max(scores.keys(), key=lambda k: scores[k])
    dbg = {"good_frames": int(good.sum()), "w_mean": float(wg.mean()), "w_median": float(np.median(wg))}
    return up_axis, scores, dbg


def to_y_up(kp3d: np.ndarray, up_axis: str) -> np.ndarray:
    """Convert kp3d to Y-up, assuming kp3d is (T,J,3) numpy."""
    p = np.array(kp3d, copy=True)
    x = p[..., 0].copy()
    y = p[..., 1].copy()
    z = p[..., 2].copy()

    if up_axis == "y":
        return p
    if up_axis == "z":
        # Z-up -> Y-up: X'=X, Y'=Z, Z'=-Y
        p[..., 0] = x
        p[..., 1] = z
        p[..., 2] = -y
        return p
    if up_axis == "x":
        # X-up -> Y-up (keep right-handed)
        p[..., 0] = -y
        p[..., 1] = x
        p[..., 2] = z
        return p
    raise ValueError(up_axis)


def unify_clip_to_y_up(
    kp3d: np.ndarray,
    face_joint_indx: list[int] = FACE_JOINTS,
    foot_ids: list[int] = FOOT_IDS,
) -> tuple[np.ndarray, str, dict]:
    """Auto-detect up axis and convert to Y-up."""
    kp3d_np = np.asarray(kp3d)

    up_geom, scores, dbg = pick_up_axis_from_geom(
        kp3d_np, face_joint_indx=face_joint_indx, foot_ids=foot_ids
    )
    st = axis_stats(kp3d_np, joint_ids=foot_ids)
    up_stat = pick_up_axis(st)

    if up_geom is None:
        up = up_stat
        method = "stat_fallback"
    else:
        svals = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ratio = (svals[0][1] + 1e-8) / (svals[1][1] + 1e-8) if len(svals) > 1 else 999.0
        if ratio < 1.05:
            up = up_stat
            method = "stat_tie_fallback"
        else:
            up = up_geom
            method = "geom"

    kp_yup = to_y_up(kp3d_np, up)
    info = {"method": method, "up_geom": up_geom, "up_stat": up_stat, "scores_geom": scores, "dbg": dbg}
    return kp_yup, up, info


def face_z_forward(
    kp3d_yup: np.ndarray,
    face_joint_indx: list[int] = FACE_JOINTS,
    eps: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """Rotate around +Y so that forward faces +Z."""
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    root_pos_init = kp3d_yup[0]

    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    if np.linalg.norm(across) < eps:
        return kp3d_yup, 0.0
    across = across / (np.linalg.norm(across) + eps)

    forward = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float32), across)
    forward_xz = np.array([forward[0], 0.0, forward[2]], dtype=np.float32)
    if np.linalg.norm(forward_xz) < eps:
        return kp3d_yup, 0.0
    forward_xz = forward_xz / (np.linalg.norm(forward_xz) + eps)

    yaw = np.arctan2(forward_xz[0], forward_xz[2])  # angle from +Z toward +X
    c = np.cos(-yaw)
    s = np.sin(-yaw)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    kp_rot = kp3d_yup @ R.T
    return kp_rot, float(yaw)


def preprocess_kp3d(
    kp3d: np.ndarray,
    do_y_up: bool = True,
    do_z_forward: bool = True,
) -> tuple[np.ndarray, dict]:
    """Common preprocess: Y-up + face Z-forward."""
    info = {}
    out = np.asarray(kp3d).astype(np.float32)

    if do_y_up:
        out, up, up_info = unify_clip_to_y_up(out)
        info["up_axis"] = up
        info["up_info"] = up_info

    if do_z_forward:
        out, yaw = face_z_forward(out)
        info["yaw_to_z"] = yaw

    return out, info
