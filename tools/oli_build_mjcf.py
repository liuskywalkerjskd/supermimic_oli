"""Build a serial (open-chain) MJCF of the LimX Oli (HU_D04_01) for SuperMimic.

LimX trains Oli in Isaac Lab on the serial model (ankle pitch/roll and waist roll/pitch driven
directly; the A/B push-rod linkages exist only in the MuJoCo sim2sim model and are mapped by
``kinematic_projection``). This script starts from the official MJCF
(humanoid-description/HU_D04_description/xml/HU_D04_01.xml), keeps its primitive collision
geoms, and removes the closed-loop linkage bodies, equality constraints, motors, sensors, the
floor and the zero-mass contact/manipulation frames. Output: data/oli_description/oli_31dof.xml.
"""

import argparse
import os
import shutil
import xml.etree.ElementTree as ET

import mujoco

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DROP_BODY_PREFIX = ("left_A_achilles", "left_B_achilles", "right_A_achilles", "right_B_achilles",
                    "waist_A", "waist_B", "contact_foot_", "left_hand_", "right_hand_")  # fmt: skip

ap = argparse.ArgumentParser()
ap.add_argument("--desc", default=os.path.join(ROOT, "..", "humanoid-description", "HU_D04_description"))
ap.add_argument("--out_dir", default=os.path.join(ROOT, "data", "oli_description"))
a = ap.parse_args()

src = os.path.join(a.desc, "xml", "HU_D04_01.xml")
os.makedirs(a.out_dir, exist_ok=True)
mesh_out = os.path.join(a.out_dir, "meshes")
if not os.path.isdir(mesh_out):
    shutil.copytree(os.path.join(a.desc, "meshes", "HU_D04_01"), mesh_out)

tree = ET.parse(src)
root = tree.getroot()
root.find("compiler").set("meshdir", "meshes/")
for tag in ("size", "equality", "actuator", "sensor", "visual"):
    for e in root.findall(tag):
        root.remove(e)
# MuJoCo-Warp rejects non-zero margins with multi-CCD; plain contacts suffice here
root.find("default").find("geom").set("margin", "0")
# the skybox/floor assets are only used by the dropped floor
for asset in root.findall("asset"):
    if asset.find("mesh") is None:
        root.remove(asset)


def prune(parent):
    for child in list(parent):
        if child.tag == "body" and child.get("name", "").startswith(DROP_BODY_PREFIX):
            parent.remove(child)
        elif child.tag in ("light", "camera") or (child.tag == "geom" and child.get("name") == "floor"):
            parent.remove(child)
        else:
            prune(child)


wb = root.find("worldbody")
prune(wb)
base = wb.find("body")
base.set("pos", "0 0 0.92")
for s in base.iter("site"):
    pass

out = os.path.join(a.out_dir, "oli_31dof.xml")
tree.write(out)
m = mujoco.MjModel.from_xml_path(out)
names = [m.joint(i).name for i in range(m.njnt)]
print("nq", m.nq, "nv", m.nv, "nbody", m.nbody, "ngeom", m.ngeom, "mass", round(float(m.body_mass.sum()), 3))
print("joints", names)
assert m.nq == 7 + 31 and m.neq == 0, (m.nq, m.neq)
