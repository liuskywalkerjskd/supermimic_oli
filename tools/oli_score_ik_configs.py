"""Score Oli LAFAN1 IK configs with LimX GMR's limb-direction metric (scripts/_calibrate_ik_offsets.py).

Run from a limxdynamics/GMR checkout:
  PYTHONPATH=. python <this> --bvh ../lafan1_bvh/walk1_subject1.bvh \
      --configs limx_official=bvh_lafan1_to_oli_edu.json mine_v2=bvh_lafan1_to_oli_mine_v2.json
Lower is better: mean angle (deg) between each robot limb segment and the human segment it follows,
plus the mean orientation residual of the IK targets.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import _calibrate_ik_offsets as C  # noqa: E402
import general_motion_retargeting.params as P  # noqa: E402
from general_motion_retargeting.utils.lafan1 import load_bvh_file  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--bvh", required=True)
ap.add_argument("--configs", nargs="+", required=True, help="name=file.json (in ik_configs/)")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--upstream_gmr", default=None, help="score name=file pairs prefixed with up: on frames from the upstream YanjieZe/GMR loader")
a = ap.parse_args()

frames, h = load_bvh_file(a.bvh, format="lafan1")
up_frames = None
if a.upstream_gmr:
    import importlib.util
    import types

    # load the upstream loader under a separate package name (its axis convention differs by 90 deg yaw)
    pkg = types.ModuleType("gmr_upstream"); pkg.__path__ = [os.path.join(a.upstream_gmr, "general_motion_retargeting")]  # noqa: E702
    sys.modules["gmr_upstream"] = pkg
    for sub in ("utils", "utils.lafan_vendor"):
        m = types.ModuleType("gmr_upstream." + sub); m.__path__ = [os.path.join(pkg.__path__[0], *sub.split("."))]  # noqa: E702
        sys.modules["gmr_upstream." + sub] = m
    src = open(os.path.join(pkg.__path__[0], "utils", "lafan1.py")).read().replace("general_motion_retargeting.", "gmr_upstream.")
    mod = types.ModuleType("gmr_upstream.utils.lafan1")
    exec(compile(src, "upstream_lafan1.py", "exec"), mod.__dict__)
    up_frames, up_h = mod.load_bvh_file(a.bvh)
clip = os.path.splitext(os.path.basename(a.bvh))[0]
for spec in a.configs:
    name, f = spec.split("=")
    use_up = name.startswith("up:")
    P.IK_CONFIG_DICT["bvh_lafan1"]["limx_oli_edu"] = P.IK_CONFIG_ROOT / f
    e = C.segment_errors("limx_oli_edu", "bvh_lafan1", up_frames if use_up else frames, up_h if use_up else h, a.frames)
    seg = {f"{p}->{d}": v for (p, d), v in e.items()}
    arms = np.mean([v for k, v in seg.items() if "Arm" in k or "Hand" in k])
    legs = np.mean([v for k, v in seg.items() if "Leg" in k or "Foot" in k])
    res = seg.get("orientation->residual", float("nan"))
    detail = "  ".join(f"{k.replace('->', '>')}:{v:.0f}" for k, v in seg.items() if k != "orientation->residual")
    print(f"{clip:24s} {name:14s} legs {legs:5.1f}  arms {arms:5.1f}  orient_resid {res:5.1f}  | {detail}", flush=True)
