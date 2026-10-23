"""Retarget LAFAN1 BVH clips to the LimX Oli with GMR (github.com/YanjieZe/GMR).

Writes SuperMimic CSVs (30 fps rows: root pos (3), root quat xyzw (4), 31 joint angles in
supermimic.robot_oli.JOINT_NAMES order) to data/lafan1_oli/. Uses the same 40 clips as the G1
LAFAN1 retarget set (lvhaidong/LAFAN1_Retargeting_Dataset).

Each clip's root height is shifted once so that the foot soles touch the ground (median over the
frames' lowest foot-box corner, so single-frame outliers do not lift the whole clip).

Usage (GMR venv: mink, mujoco, daqp, scipy, rich):
  python tools/oli_retarget_lafan1.py --gmr ../GMR --bvh_dir ../lafan1_bvh -j 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
MJCF = os.path.join(ROOT, "data", "oli_description", "oli_31dof.xml")
IK_CFG = os.path.join(HERE, "oli_ik_config.json")
OUT = os.path.join(ROOT, "data", "lafan1_oli")
WARMUP_SOLVES = 100

CLIPS = """dance1_subject1 dance1_subject2 dance1_subject3 dance2_subject1 dance2_subject2 dance2_subject3
dance2_subject4 dance2_subject5 fallAndGetUp1_subject1 fallAndGetUp1_subject4 fallAndGetUp1_subject5
fallAndGetUp2_subject2 fallAndGetUp2_subject3 fallAndGetUp3_subject1 fight1_subject2 fight1_subject3
fight1_subject5 fightAndSports1_subject1 fightAndSports1_subject4 jumps1_subject1 jumps1_subject2
jumps1_subject5 run1_subject2 run1_subject5 run2_subject1 run2_subject4 sprint1_subject2 sprint1_subject4
walk1_subject1 walk1_subject2 walk1_subject5 walk2_subject1 walk2_subject3 walk2_subject4 walk3_subject1
walk3_subject2 walk3_subject3 walk3_subject4 walk3_subject5 walk4_subject1""".split()

JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_yaw_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
]  # fmt: skip  (kept in sync with supermimic/robot_oli.py; this script runs without Isaac Lab)


def _import_gmr(gmr_root: str, ik_cfg: str = IK_CFG):
    """Import GMR's retargeter and BVH loader without its heavy package __init__ (torch, viewers)."""
    pkg_dir = os.path.join(gmr_root, "general_motion_retargeting")
    pkg = types.ModuleType("general_motion_retargeting")
    pkg.__path__ = [pkg_dir]
    sys.modules["general_motion_retargeting"] = pkg
    from general_motion_retargeting import params

    params.ROBOT_XML_DICT["limx_oli"] = MJCF
    params.IK_CONFIG_DICT["bvh_lafan1"]["limx_oli"] = ik_cfg
    params.ROBOT_BASE_DICT["limx_oli"] = "base_link"
    from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    return GeneralMotionRetargeting, load_bvh_file


def _foot_corners(model):
    import mujoco

    out = []
    for name in ("left_foot", "right_foot"):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        sx, sy, sz = model.geom_size[g]
        c = np.array([[x, y, -sz] for x in (-sx, sx) for y in (-sy, sy)])
        out.append((g, c))
    return out


def retarget_clip(job):
    gmr_root, bvh_dir, name, out, ik_cfg = job
    import mujoco

    GMR, load_bvh_file = _import_gmr(gmr_root, ik_cfg)
    frames, human_height = load_bvh_file(os.path.join(bvh_dir, name + ".bvh"))
    rt = GMR(src_human="bvh_lafan1", tgt_robot="limx_oli", actual_human_height=human_height, verbose=False)
    model = rt.model
    jadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES]
    # GMR starts its IK from the zero configuration and runs only a few iterations per frame, so
    # the first frames converge from that pose (a ~0.3 s transient with 30-40 rad/s joint speeds).
    # Warm-start on the first frame until converged before recording.
    for _ in range(WARMUP_SOLVES):
        rt.retarget(frames[0])
    qpos = np.array([rt.retarget(f).copy() for f in frames])  # (T, 7 + 31), quat wxyz

    # ground the clip on the foot soles
    data = mujoco.MjData(model)
    feet = _foot_corners(model)
    low = np.empty(len(qpos))
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        low[t] = min((data.geom_xpos[g] + c @ data.geom_xmat[g].reshape(3, 3).T)[:, 2].min() for g, c in feet)
    dz = float(np.median(np.sort(low)[: max(1, len(low) // 2)]))  # median of the lowest half of frames
    qpos[:, 2] -= dz

    root_p = qpos[:, :3]
    quat_xyzw = np.concatenate((qpos[:, 4:7], qpos[:, 3:4]), 1)
    joints = qpos[:, jadr]
    lo = model.jnt_range[[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]]
    viol = float(np.maximum(lo[:, 0] - joints, joints - lo[:, 1]).max())
    np.savetxt(os.path.join(out, name + ".csv"), np.concatenate((root_p, quat_xyzw, joints), 1), delimiter=",", fmt="%.6f")
    return name, len(qpos), dz, float(np.percentile(low - dz, 5)), viol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gmr", default=os.path.join(ROOT, "..", "GMR"))
    ap.add_argument("--bvh_dir", default=os.path.join(ROOT, "..", "lafan1_bvh"))
    ap.add_argument("--clips", nargs="*", default=CLIPS)
    ap.add_argument("-j", "--jobs", type=int, default=8)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--ik_cfg", default=IK_CFG)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(os.path.abspath(a.gmr), os.path.abspath(a.bvh_dir), c, os.path.abspath(a.out), os.path.abspath(a.ik_cfg)) for c in a.clips]
    stats = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, T, dz, p5, viol in ex.map(retarget_clip, jobs):
            stats[name] = {"frames": T, "ground_shift": round(dz, 4), "foot_z_p5": round(p5, 4), "max_limit_violation": round(viol, 4)}
            print(f"{name}: {T} frames, ground shift {dz:+.3f} m, foot z p5 {p5:+.3f} m, limit viol {viol:.3f} rad", flush=True)
    json.dump(stats, open(os.path.join(a.out, "_retarget_stats.json"), "w"), indent=1)
    print("total frames", sum(s["frames"] for s in stats.values()))


if __name__ == "__main__":
    main()
