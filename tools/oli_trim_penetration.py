"""Build a dataset without the physically infeasible segments (v5 -> v6).

A frame is bad when, in the ungrounded retarget (--ref, e.g. v4), some collision geom is more than
--max_pen below the floor: per-frame grounding turns such frames into floating poses held up by
one limb. Bad frames are dilated by --pad seconds, and each remaining run of good frames that is
at least --min_len seconds long is written from --src (e.g. v5, grounded) as its own clip
(<clip>.csv if the clip is untouched, else <clip>_partK.csv). Motion is never stitched across a cut.

  python tools/oli_trim_penetration.py --ref data/lafan1_oli_v4 --src data/lafan1_oli_v5 --dst data/lafan1_oli_v6
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
from oli_ground_motion import penetration  # noqa: E402

FPS = 30


def runs(mask: np.ndarray):
    """(start, end) of runs of True, end exclusive."""
    d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def process(job):
    ref, src, dst, max_pen, pad, min_len = job
    name = os.path.basename(src)[:-4]
    _, pen = penetration(ref)
    x = np.loadtxt(src, delimiter=",")
    assert len(x) == len(pen), (name, len(x), len(pen))
    bad = pen > max_pen
    p = int(round(pad * FPS))
    if p and bad.any():
        bad = np.convolve(bad.astype(float), np.ones(2 * p + 1), mode="same") > 0
    good = [(s, e) for s, e in runs(~bad) if e - s >= min_len * FPS]
    if len(good) == 1 and good[0] == (0, len(x)):
        np.savetxt(os.path.join(dst, name + ".csv"), x, delimiter=",", fmt="%.6f")
        parts = [name]
    else:
        parts = []
        for k, (s, e) in enumerate(good):
            part = f"{name}_part{k}"
            np.savetxt(os.path.join(dst, part + ".csv"), x[s:e], delimiter=",", fmt="%.6f")
            parts.append(part)
    kept = sum(e - s for s, e in good)
    return name, {"frames": len(x), "kept": int(kept), "kept_%": round(100 * kept / len(x), 1),
                  "parts": parts, "segments": [[int(s), int(e)] for s, e in good]}  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="ungrounded retarget used to measure penetration")
    ap.add_argument("--src", required=True, help="dataset the kept frames are taken from")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--max_pen", type=float, default=0.15, help="m")
    ap.add_argument("--pad", type=float, default=0.5, help="s, dilation around bad frames")
    ap.add_argument("--min_len", type=float, default=2.0, help="s, shortest kept segment")
    ap.add_argument("-j", "--jobs", type=int, default=40)
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    srcs = sorted(glob.glob(os.path.join(a.src, "*.csv")))
    jobs = [(os.path.join(a.ref, os.path.basename(s)), s, a.dst, a.max_pen, a.pad, a.min_len) for s in srcs]
    stats = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, s in ex.map(process, jobs):
            stats[name] = s
            if s["kept_%"] < 100:
                print(f"{name:26s} kept {s['kept_%']:5.1f}% in {len(s['parts'])} part(s)", flush=True)
    tot, kept = sum(s["frames"] for s in stats.values()), sum(s["kept"] for s in stats.values())
    print(f"total: kept {kept}/{tot} frames ({100 * kept / tot:.1f}%), {sum(len(s['parts']) for s in stats.values())} clips")
    json.dump({"args": vars(a), "clips": stats}, open(os.path.join(a.dst, "_trim_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
