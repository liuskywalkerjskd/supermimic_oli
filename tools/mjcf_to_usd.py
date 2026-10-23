"""Convert the robot MJCF to USD with the standalone Isaac Sim MJCF importer (no Kit).

The robot is selected with SM_ROBOT (oli | g1), see supermimic/robot.py.
"""
import os
import sys

from pxr import Plug

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, ".."))
import isaacsim.asset.importer.utils.impl as _iu

Plug.Registry().RegisterPlugins(os.path.join(os.path.dirname(_iu.__file__), "schemas"))
from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

import importlib

robot = importlib.import_module("supermimic.robot_" + os.environ.get("SM_ROBOT", "oli"))
cfg = MJCFImporterConfig(mjcf_path=os.path.normpath(robot.MJCF_PATH), usd_path=os.path.normpath(robot.USD_DIR))
print(MJCFImporter(cfg).import_mjcf())
