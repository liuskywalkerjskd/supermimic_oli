"""Robot selection: ``SM_ROBOT=oli`` (default, LimX Oli HU_D04_01) or ``SM_ROBOT=g1`` (Unitree G1).

Every robot module exports the same names; all MDP/network dimensions derive from them.
"""

from __future__ import annotations

import os

ROBOT = os.environ.get("SM_ROBOT", "oli")
if ROBOT == "oli":
    from .robot_oli import *  # noqa: F401,F403
    from .robot_oli import JOINT_NAMES, TRACKED_BODIES
elif ROBOT == "g1":
    from .robot_g1 import *  # noqa: F401,F403
    from .robot_g1 import JOINT_NAMES, TRACKED_BODIES
else:
    raise ValueError(f"unknown SM_ROBOT={ROBOT}")

NUM_J = len(JOINT_NAMES)
NUM_B = len(TRACKED_BODIES)
