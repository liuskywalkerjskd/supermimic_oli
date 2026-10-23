"""Per-frame whole-body grounding of retargeted motion CSVs (v2 -> v3).

GMR's IK has no ground constraint and the retarget script only shifts each clip once by the foot
soles, so hands, knees and the torso go through the floor whenever they support the body
(crawling, kneeling, get-ups). For every frame this computes the deepest floor penetration of any
collision geom and lifts the root by that amount. The lift is a smoothed upper envelope: a max
filter over +-H frames followed by a Hann window of the same half-width, which keeps
lift_t >= penetration_t for every frame (each averaged value already covers frame t) while
adding no velocity spikes. Joint angles and the root's xy/orientation are unchanged.

Usage: python tools/oli_ground_motion.py --src data/lafan1_oli_v2 --dst data/lafan1_oli_v3
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def penetration(path: str) -> np.ndarray:
    import mujoco

    import replay_motion as rm

    model = mujoco.MjModel.from_xml_string(rm.scene_xml())
    data = mujoco.MjData(model)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    x = np.loadtxt(path, delimiter=",")
    pen = np.zeros(len(x))
    for t in range(len(x)):
        rm.set_qpos(data, x[t])
        mujoco.mj_forward(model, data)
        for k in range(data.ncon):
            c = data.contact[k]
            if floor in (c.geom1, c.geom2):
                pen[t] = max(pen[t], -c.dist)
    return x, pen


def smooth_envelope(p: np.ndarray, H: int) -> np.ndarray:
    pad = np.pad(p, H, mode="edge")
    mx = np.lib.stride_tricks.sliding_window_view(pad, 2 * H + 1).max(-1)
    w = np.hanning(2 * H + 3)[1:-1]
    w /= w.sum()
    return np.convolve(np.pad(mx, H, mode="edge"), w, mode="valid")


def process(job):
    src, dst, H = job
    x, pen = penetration(src)
    lift = smooth_envelope(pen, H)
    assert np.all(lift >= pen - 1e-9)
    y = x.copy()
    y[:, 2] += lift
    np.savetxt(dst, y, delimiter=",", fmt="%.6f")
    vz_add = np.abs(np.diff(lift)) * 30
    return os.path.basename(src)[:-4], {
        "pen_max_cm": round(100 * pen.max(), 1),
        "frames_lifted_%": round(100 * float((lift > 0.005).mean()), 1),
        "lift_max_cm": round(100 * lift.max(), 1),
        "added_vz_max": round(float(vz_add.max()), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--half_window", type=int, default=6, help="frames at 30 fps (0.2 s)")
    ap.add_argument("-j", "--jobs", type=int, default=40)
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.src, "*.csv")))
    jobs = [(f, os.path.join(a.dst, os.path.basename(f)), a.half_window) for f in files]
    stats = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, s in ex.map(process, jobs):
            stats[name] = s
            print(f"{name:26s} pen {s['pen_max_cm']:5.1f} cm -> lifted {s['frames_lifted_%']:5.1f}% of frames, "
                  f"max lift {s['lift_max_cm']:5.1f} cm, max added |vz| {s['added_vz_max']:.2f} m/s", flush=True)  # fmt: skip
    json.dump(stats, open(os.path.join(a.dst, "_ground_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
