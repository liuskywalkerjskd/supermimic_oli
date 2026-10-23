"""LimX Oli (HU_D04_01, 31 DoF) articulation, following limxdynamics/humanoid-rl-isaaclab.

Actuator model from ``limx_rl_forge/assets/config/HU_D04_01.py``: implicit PD with a 5 Hz natural
frequency and damping ratio 2 on the reflected armature, the same effort/velocity limits, armature
and joint friction. The action scale is 0.25 * rated torque / Kp, which reproduces the per-group
scales of LimX's ``JointPositionActionCfg`` (0.2511 legs, 0.1121 ankle/waist, 0.1200 arms,
0.3141 head/wrist). The model is the serial one LimX trains on (no ankle/waist push-rods).
"""

from __future__ import annotations

import os
import re

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

NAME = "oli"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MJCF_PATH = os.path.join(DATA_DIR, "oli_description", "oli_31dof.xml")
MESH_DIR = os.path.join(DATA_DIR, "oli_description")
USD_DIR = os.path.join(DATA_DIR, "oli_usd")
USD_PATH = os.path.join(USD_DIR, "oli_31dof", "oli_31dof.usda")
MOTION_DIR = os.path.join(DATA_DIR, "lafan1_oli")
ROOT_BODY = "base_link"
TORSO_BODY = "waist_pitch_link"  # LimX randomizes the CoM of this link
VIS_HEIGHT = 0.95

ARMATURE_LEG = 0.14125
ARMATURE_ANKLE_WAIST = 0.1845504
ARMATURE_ANKLE_WAIST_BACK = 0.094889232  # LimX derives the ankle/waist gains from this value
ARMATURE_HEAD_WRIST = 0.0153218
ARMATURE_SHOULDER_ELBOW = 0.0886706

NATURAL_FREQ = 5.0 * 2.0 * 3.1415926535  # 5 Hz
DAMPING_RATIO = 2.0

# Order of the retargeted CSVs and of the simulator (MJCF depth-first order).
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
]  # fmt: skip

# Same 14 anatomical points as BeyondMimic's G1 set; the anchor/root is base_link (pelvis).
TRACKED_BODIES = [
    "base_link", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "waist_pitch_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_roll_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_roll_link",
]  # fmt: skip

# key -> (effort limit (sim), velocity limit, armature, gain armature, rated torque, friction)
_LEG = (140.0, 12.0, ARMATURE_LEG, ARMATURE_LEG, 140.0)
_AW = (80.0, 13.6, ARMATURE_ANKLE_WAIST, ARMATURE_ANKLE_WAIST_BACK, 42.0)
_ARM = (42.0, 19.6, ARMATURE_SHOULDER_ELBOW, ARMATURE_SHOULDER_ELBOW, 42.0)
_HW = (19.0, 13.0, ARMATURE_HEAD_WRIST, ARMATURE_HEAD_WRIST, 19.0)
JOINT_MOTORS = {
    "hip_pitch": (*_LEG, 2.0), "hip_roll": (*_LEG, 2.0), "hip_yaw": (*_LEG, 1.6), "knee": (*_LEG, 2.6),
    "ankle_pitch": (*_AW, 1.0), "ankle_roll": (*_AW, 1.0),
    "waist_yaw": (42.0, 13.6, ARMATURE_ANKLE_WAIST, ARMATURE_ANKLE_WAIST_BACK, 42.0, 0.48),
    "waist_roll": (*_AW, 1.1), "waist_pitch": (*_AW, 4.89),
    "head_yaw": (*_HW, 0.0), "head_pitch": (*_HW, 0.0),
    "shoulder_pitch": (*_ARM, 0.6), "shoulder_roll": (*_ARM, 0.6), "shoulder_yaw": (*_ARM, 0.45), "elbow": (*_ARM, 0.35),
    "wrist_yaw": (*_HW, 0.3), "wrist_pitch": (*_HW, 0.3), "wrist_roll": (*_HW, 0.3),
}  # fmt: skip


def motor_of(joint_name: str):
    return JOINT_MOTORS[re.sub(r"^(left|right)_|_joint$", "", joint_name)]


def _kp(arm):
    return arm * NATURAL_FREQ**2


def _kd(arm):
    return 2.0 * DAMPING_RATIO * arm * NATURAL_FREQ


def action_scale(joint_name: str) -> float:
    m = motor_of(joint_name)
    return 0.25 * m[4] / _kp(m[3])


# LimX's init_state joint_pos (standing pose)
DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.15, "left_hip_yaw_joint": -0.05, "left_knee_joint": 0.30, "left_ankle_pitch_joint": -0.16,
    "right_hip_pitch_joint": -0.15, "right_hip_yaw_joint": 0.05, "right_knee_joint": 0.30, "right_ankle_pitch_joint": -0.16,
    "left_shoulder_pitch_joint": 0.1, "left_shoulder_roll_joint": 0.1, "left_shoulder_yaw_joint": -0.2, "left_elbow_joint": -0.2,
    "right_shoulder_pitch_joint": 0.1, "right_shoulder_roll_joint": -0.1, "right_shoulder_yaw_joint": 0.2, "right_elbow_joint": -0.2,
}  # fmt: skip


def default_joint_pos_list() -> list[float]:
    return [DEFAULT_JOINT_POS.get(n, 0.0) for n in JOINT_NAMES]


def _actuators() -> dict[str, ImplicitActuatorCfg]:
    acts = {}
    for name in JOINT_NAMES:
        effort, vel, arm, gain_arm, _, fric = motor_of(name)
        acts[name.replace("_joint", "")] = ImplicitActuatorCfg(
            joint_names_expr=[name],
            effort_limit_sim=effort,
            velocity_limit_sim=vel,
            stiffness=_kp(gain_arm),
            damping=_kd(gain_arm),
            armature=arm,
            friction=fric,
        )
    return acts


ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        variants={"Physics": "mujoco"},
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0.0, angular_damping=0.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.92), joint_pos=DEFAULT_JOINT_POS, joint_vel={".*": 0.0}),
    soft_joint_pos_limit_factor=0.9,
    actuators=_actuators(),
)
