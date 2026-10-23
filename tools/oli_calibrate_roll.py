"""Run LimX GMR's roll calibration (scripts/_calibrate_ik_offsets.py --tune roll) on an Oli config
that is used with the upstream (YanjieZe/GMR) LAFAN1 loader.

LimX's tool loads BVH with its own axis convention, which is 90 deg of yaw away from the upstream
loader that our configs (and tools/oli_retarget_lafan1.py) use. This wrapper swaps in the upstream
loader and scores over several clips at once, then writes the calibrated config.

Run from a limxdynamics/GMR checkout (GMR venv):
  PYTHONPATH=. python <this> --upstream_gmr ../GMR --config tools/oli_ik_config_v4.json \
      --out tools/oli_ik_config_v5.json --bvh ../lafan1_bvh/walk1_subject1.bvh ../lafan1_bvh/dance2_subject1.bvh
"""

import argparse
import json
import os
import shutil
import sys
import types

ap = argparse.ArgumentParser()
ap.add_argument("--upstream_gmr", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--bvh", nargs="+", required=True)
ap.add_argument("--num_frames", type=int, default=20, help="per clip")
ap.add_argument("--skip", nargs="*", default=["Hips"], help="human bones whose roll is not searched")
a = ap.parse_args()

sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import _calibrate_ik_offsets as C  # noqa: E402
import general_motion_retargeting.params as P  # noqa: E402


def upstream_loader():
    root = os.path.join(a.upstream_gmr, "general_motion_retargeting")
    for name, path in (("gmr_upstream", root), ("gmr_upstream.utils", os.path.join(root, "utils")),
                       ("gmr_upstream.utils.lafan_vendor", os.path.join(root, "utils", "lafan_vendor"))):  # fmt: skip
        m = types.ModuleType(name)
        m.__path__ = [path]
        sys.modules[name] = m
    src = open(os.path.join(root, "utils", "lafan1.py")).read().replace("general_motion_retargeting.", "gmr_upstream.")
    mod = types.ModuleType("gmr_upstream.utils.lafan1")
    exec(compile(src, "upstream_lafan1.py", "exec"), mod.__dict__)
    return mod.load_bvh_file


load = upstream_loader()
frames, height = [], None
for path in a.bvh:
    f, height = load(path)
    step = max(1, len(f) // a.num_frames)
    frames += f[::step][: a.num_frames]
print(f"{len(frames)} frames from {len(a.bvh)} clips")
# segment_errors subsamples by len(frames)//num_frames; keep all pooled frames
C.load_frames = lambda src, motion_file: (frames, height)

shutil.copy(a.config, a.out)
P.IK_CONFIG_DICT["bvh_lafan1"]["limx_oli_edu"] = os.path.abspath(a.out)
if a.skip:
    # calibrate_roll iterates over limb chains starting at the root; drop bodies we do not want rolled
    C.CHAINS = {k: [[b for b in ch if b not in a.skip] for ch in v] for k, v in C.CHAINS.items()}
C.calibrate_roll("limx_oli_edu", "bvh_lafan1", "<pooled>", os.path.abspath(a.out), len(frames))
# keep our config's robot conventions: same bodies, only rot_offsets changed
cfg = json.load(open(a.out))
print("written", a.out, "bodies", list(cfg["ik_match_table1"]))
