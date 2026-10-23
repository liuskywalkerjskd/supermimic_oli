"""Diagnose floor penetration in retargeted clips: human data vs GMR targets vs robot result.

For the deepest-penetration frame of each clip (in --motion_dir) this prints, per IK-matched body:
raw human joint height, GMR target height (after scaling/offsets, before the clip ground shift),
robot body height in the CSV, and the 3D target-to-robot distance. Also the lowest point of the
raw human skeleton, and the deepest robot geom.

Run in the GMR venv: python tools/oli_diag_penetration.py --clips fallAndGetUp1_subject1 walk3_subject4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oli_retarget_lafan1 as orl  # noqa: E402
import replay_motion as rm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gmr", default=os.path.join(HERE, "..", "..", "GMR"))
    ap.add_argument("--bvh_dir", default=os.path.join(HERE, "..", "..", "lafan1_bvh"))
    ap.add_argument("--motion_dir", default=os.path.join(HERE, "..", "data", "lafan1_oli_v2"))
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--frames", type=int, nargs="*", default=None, help="explicit frames (30 fps) instead of the worst one")
    a = ap.parse_args()

    GMR, load_bvh_file = orl._import_gmr(os.path.abspath(a.gmr))
    model = mujoco.MjModel.from_xml_string(rm.scene_xml())
    data = mujoco.MjData(model)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    stats = json.load(open(os.path.join(a.motion_dir, "_retarget_stats.json")))
    for clip in a.clips:
        x = np.loadtxt(os.path.join(a.motion_dir, clip + ".csv"), delimiter=",")
        dz = stats[clip]["ground_shift"]  # CSV root z = IK root z - dz
        if a.frames:
            frames = a.frames
        else:
            pen = np.zeros(len(x))
            for t in range(0, len(x), 2):
                rm.set_qpos(data, x[t])
                mujoco.mj_forward(model, data)
                pen[t] = max([-data.contact[k].dist for k in range(data.ncon) if floor in (data.contact[k].geom1, data.contact[k].geom2)], default=0)
            frames = [int(pen.argmax())]
        human, height = load_bvh_file(os.path.join(a.bvh_dir, clip + ".bvh"))
        rt = GMR(src_human="bvh_lafan1", tgt_robot="limx_oli", actual_human_height=height, verbose=False)
        for t in frames:
            raw = {k: (np.asarray(v[0]), v[1]) for k, v in human[t].items()}
            rt.update_targets({k: [v[0].copy(), np.asarray(v[1]).copy()] for k, v in human[t].items()})
            tgt = rt.scaled_human_data
            rm.set_qpos(data, x[t])
            mujoco.mj_forward(model, data)
            deep = max(((-data.contact[k].dist, model.body(model.geom_bodyid[data.contact[k].geom1 if data.contact[k].geom2 == floor else data.contact[k].geom2]).name)
                        for k in range(data.ncon) if floor in (data.contact[k].geom1, data.contact[k].geom2)), default=(0, "-"))  # fmt: skip
            low_bone = min(raw, key=lambda k: raw[k][0][2])
            print(f"\n=== {clip} frame {t} (t={t / 30:.2f}s)  deepest robot geom: {deep[1]} {100 * deep[0]:.1f} cm below floor")
            print(f"    lowest raw human bone: {low_bone} z={raw[low_bone][0][2]:+.3f} m   clip ground shift dz={dz:+.3f}")
            print(f"    {'robot body':24s} {'human bone':13s} {'human z':>8s} {'target z':>9s} {'robot z':>8s} {'|tgt-robot|':>11s}")
            for body, entry in rt.ik_match_table2.items():
                bone = entry[0]
                rb = data.body(body).xpos.copy()
                rb[2] += dz  # back to the IK frame
                tz = tgt[bone][0]
                print(f"    {body:24s} {bone:13s} {raw[bone][0][2]:+8.3f} {tz[2]:+9.3f} {rb[2]:+8.3f} {np.linalg.norm(tz - rb):11.3f}")
            for side in ("left", "right"):
                wr = data.body(f"{side}_wrist_roll_link")
                g = [i for i in range(model.ngeom) if model.geom_bodyid[i] == wr.id and model.geom_contype[i]]
                lowest = min(data.geom_xpos[i][2] for i in g) if g else float("nan")
                print(f"    {side} hand: wrist_roll origin z={wr.xpos[2]:+.3f}, its collision geoms' lowest center z={lowest:+.3f}")


if __name__ == "__main__":
    main()
