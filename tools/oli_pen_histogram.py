"""Fraction of frames above floor-penetration thresholds, per motion category (ungrounded retarget)."""

import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oli_ground_motion import penetration  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else "data/lafan1_oli_v4"
files = sorted(glob.glob(os.path.join(src, "*.csv")))
with ProcessPoolExecutor(40) as ex:
    pens = {os.path.basename(f)[:-4]: p for f, (_, p) in zip(files, ex.map(penetration, files))}
cat = lambda k: k.split("_")[0].rstrip("0123456789")  # noqa: E731
groups = sorted({cat(k) for k in pens})
print(f"{'threshold':>9s} {'all':>6s} " + " ".join(f"{g:>14s}" for g in groups))
for th in (0.03, 0.05, 0.08, 0.10, 0.15):
    allp = np.concatenate(list(pens.values()))
    row = [100 * (np.concatenate([p for k, p in pens.items() if cat(k) == g]) > th).mean() for g in groups]
    print(f"{100 * th:7.0f}cm {100 * (allp > th).mean():6.2f} " + " ".join(f"{v:14.2f}" for v in row))
