# SuperMimic for LimX Oli on Isaac Lab 3 (kit-less, MuJoCo-Warp)

**English** | [简体中文](README.zh-CN.md)

An **unofficial Oli adaptation** of [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed), Chong Zhang's reproduction of *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*. The original repository is our implementation reference and starting point. This repository adds the **LimX Oli HU_D04_01 (31 DoF)**, motion preparation, and complete rollout exports. G1 support is retained.

Oli adaptation: **Tianxing Liu**. Upstream implementation: **Chong Zhang (ETH Zurich)**. Code license: [MIT](LICENSE).

> **One Oli policy for 40 LAFAN1 motions.** At the selected checkpoint, iteration **18,000**, the original full-library evaluation recorded normalized return **0.6783**, with **36/40 clips completing without termination**. Training elapsed time at that checkpoint was approximately **2.20 h on one RTX 4090**.

[![Oli tracking a LAFAN1 walking motion](results/oli_walk1_subject1_preview.gif)](results/oli_walk1_subject1_full.mp4)

**[Watch or download the full Oli video — 4 min 19 s, MP4](results/oli_walk1_subject1_full.mp4)**

*Demo: `walk1_subject1`, the highest-scoring successful clip in the selected benchmark (0.8638). The GIF shows a 12-second excerpt; the MP4 contains the complete rollout after a 2.60-second initial T-pose segment is skipped. Solid robot: policy-controlled physical simulation. Blue ghost: reference motion, aligned to the simulated root in XY for pose comparison. This exported demo did not trigger the tracking termination threshold.*

Changes: [CHANGELOG.md](CHANGELOG.md). Recorded metrics: [benchmark JSON](results/ckpt_018000_eval.json) and [full-export summary](results/oli_v5_export_summary.json). The inherited [MILESTONES.md](MILESTONES.md) describes upstream G1 experiments.

## Layout

| Path | What |
|---|---|
| `IsaacLab/` | Untracked runtime, pinned by `tools/setup.sh` to `46d6cc3` |
| `docker/` | Upstream kit-less image and container launcher |
| `supermimic/` | World model, policy, training, evaluation, and robot configuration |
| `tools/oli_*.py`, `tools/oli_ik_config*.json` | Oli conversion, retargeting, calibration, grounding, and diagnostics |
| `tools/record_full_motion.py`, `tools/render_full_motion.py` | Batched continuous inference and individual full-length videos |
| `configs/` | Per-motion start frames for the selected v5 exports |
| `results/` | Oli demo and metrics; inherited G1 illustrations |
| `data/`, `logs/` | Untracked assets, motions, checkpoints, and training logs |

## Setup

**Requirements:** Linux, an NVIDIA GPU, an Isaac Lab kit-less runtime, and FFmpeg for videos. The Oli experiment used a 24 GB RTX 4090, Isaac Lab commit [`46d6cc3`](https://github.com/isaac-sim/IsaacLab/commit/46d6cc3d7b115906fad12efd372fbf69f4e3d3ed), PyTorch `2.11.0+cu126`, MuJoCo/MuJoCo-Warp `3.15.0`, and Warp `1.17.0`. Retargeting uses a separate [GMR](https://github.com/YanjieZe/GMR) environment.

Physics uses Newton with MuJoCo-Warp, **50 Hz control** and **four 5 ms physics substeps**. It does not require the Isaac Sim GUI.

### 1. Isaac Lab runtime

Run on the host with Docker and the NVIDIA Container Toolkit installed:

```bash
git clone https://github.com/liuskywalkerjskd/supermimic_oli.git
cd supermimic_oli
tools/setup.sh
docker/run_container.sh 0 128g
```

The repository is mounted at `/workspace/supermimic` in the `supermimic` container. `tools/setup.sh` pins Isaac Lab and builds the upstream kit-less image. An existing installation of the same pinned runtime can run the Python commands directly.

### 2. Oli robot assets

Download the official model on the host, then build the serial MJCF and USD using the runtime:

```bash
mkdir -p data/vendor
git clone --depth 1 https://github.com/limxdynamics/humanoid-description.git \
  data/vendor/humanoid-description
docker exec supermimic python tools/oli_build_mjcf.py \
  --desc data/vendor/humanoid-description/HU_D04_description
docker exec -e SM_ROBOT=oli supermimic python tools/mjcf_to_usd.py
```

Expected assets are `data/oli_description/oli_31dof.xml`, its `meshes/` directory, and `data/oli_usd/oli_31dof/oli_31dof.usda`. The serial model removes the closed-loop ankle/waist push-rods, following LimX's training representation. Asset sources retain their own licenses.

### 3. LAFAN1 → Oli motions

Obtain the LAFAN1 BVH files from [Ubisoft La Forge](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) under its terms and place the 40 named clips in `data/lafan1_bvh/`. Install GMR in a separate environment following [its instructions](https://github.com/YanjieZe/GMR). Our retargeting run used GMR commit `25f56db79ac411258b377b6d4ccba0c8ff3372ac`.

From the repository root, with that GMR environment active:

```bash
git clone https://github.com/YanjieZe/GMR.git data/vendor/GMR
git -C data/vendor/GMR checkout 25f56db79ac411258b377b6d4ccba0c8ff3372ac
python tools/oli_retarget_lafan1.py \
  --gmr data/vendor/GMR --bvh_dir data/lafan1_bvh \
  --ik_cfg tools/oli_ik_config_v5.json --out data/lafan1_oli_v5_raw -j 8
SM_ROBOT=oli python tools/oli_ground_motion.py \
  --src data/lafan1_oli_v5_raw --dst data/lafan1_oli_v5 -j 8
```

CSV rows contain root position, root quaternion in **xyzw** order, and **31 joint angles**, at 30 Hz. The motion library resamples them to 50 Hz. `tools/oli_retarget_lafan1.py` lists the 40 clip names. BVH files, retargeted CSVs, and robot meshes are not redistributed here.

## Usage

### Train and evaluate

Run these from the host after preparing the motions:

```bash
docker exec -e SM_ROBOT=oli supermimic python tools/test_env.py --motion_dir data/lafan1_oli_v5
docker exec -e SM_ROBOT=oli supermimic python supermimic/train.py \
  --motion_dir data/lafan1_oli_v5 --log_dir logs/oli_lafan1_v5 \
  --num_envs 4096 --iterations 320000 --amp bf16 --seed 0
```

In another terminal, evaluate new checkpoints as they appear:

```bash
docker exec -e SM_ROBOT=oli supermimic python supermimic/evaluate.py \
  --run_dir logs/oli_lafan1_v5 --motion_dir data/lafan1_oli_v5 --watch
```

Checkpoints are saved every 2,000 iterations. Training generates the checkpoint paths used below; pretrained weights are not bundled. Use `SM_ROBOT=g1` with G1 assets and motions for the retained upstream configuration. Optional CSV-to-SwanLab logging is available through `tools/swanlab_sync.py`, with the SwanLab SDK and your own login.

### Full-length inference without the initial T-pose

The supplied v5 selections specify start frames in the **50 Hz resampled reference**. The recorder simulates selected clips in parallel, initializes each at its configured reference pose and velocity, and records through the final source frame. It keeps simulating after a termination threshold and the renderer marks that event.

```bash
docker exec -e SM_ROBOT=oli supermimic python tools/record_full_motion.py \
  --checkpoint logs/oli_lafan1_v5/ckpt_018000.pt \
  --motion-dir data/lafan1_oli_v5 \
  --selection configs/oli_v5_export_best4.json --out-dir logs/oli_full

docker exec -e SM_ROBOT=oli -e MUJOCO_GL=egl supermimic \
  python tools/render_full_motion.py \
  --traj logs/oli_full/walk1_subject1.npz \
  --meta logs/oli_full/walk1_subject1.json \
  --out logs/oli_full/walk1_subject1.mp4
```

Use `configs/oli_v5_export_success36.json` to record all 36 clips that passed the selected original benchmark; render each resulting NPZ separately. These start frames are specific to the v5 dataset. Two clips retain early leg motion while the arms are still extended, avoiding removal of part of the action. Videos use H.264, 960×540, 25 fps and include the terminal state.

## Results

Results below refer to **v5, checkpoint 18,000**, recorded on 2026-10-23.

| Measurement | Result |
|---|---:|
| Original benchmark: normalized return over all 40 clips | 0.6783 |
| Original benchmark: clips without termination | 36 / 40 |
| Selected walking clip: original normalized return | 0.8638 |
| Separate full-length exports starting after T-pose | 36 clips |
| Exports without a termination threshold crossing | 35 / 36 |

The benchmark starts at each original first frame. Video exports start from later reference states, so the success counts describe different evaluation conditions. `walk3_subject4` crossed the termination threshold at **21.32 s** in the export run; its full trajectory was retained. All 36 local exports passed complete decoding, duration, and terminal-frame checks. Only the selected walking video is bundled here.

## Implementation

| Component | Files |
|---|---|
| Robot selection; Oli/G1 joints, actuators, tracked bodies | `supermimic/robot.py`, `robot_oli.py`, `robot_g1.py` |
| Motion resampling and differentiable kinematics | `supermimic/motion.py`, `kinematics.py` |
| Observations, tracking rewards, termination, simulation | `supermimic/mdp.py`, `env.py` |
| World model, policy, critic, replay, learning | `supermimic/networks.py`, `buffer.py`, `train.py` |
| Original full-library benchmark | `supermimic/evaluate.py` |
| Continuous recording and video rendering | `tools/record_full_motion.py`, `render_full_motion.py` |

## Deviations and limitations

- Oli uses 31 joints and a serial training model; upstream G1 uses 29 joints. Their scores are not a direct benchmark comparison.
- Self-collisions remain disabled. Grounding adjusts reference root height to reduce floor penetration; it does not make every retargeted motion dynamically feasible.
- The video ghost is aligned in root XY. It illustrates pose tracking and does not expose absolute position drift.
- This is simulation research code. Hardware deployment has not been validated in this project.

## Attribution and licenses

- **Primary code reference:** [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed), Chong Zhang / ETH Zurich, MIT. Base commit: `a1508d42c9d1699a52e8233ab16d08c488441ab9`. Upstream history and copyright are retained.
- **Method:** [SuperMimic](https://arxiv.org/abs/2610.28480). This unofficial adaptation is not affiliated with the paper authors.
- **Oli assets and actuator reference:** [LimX humanoid-description](https://github.com/limxdynamics/humanoid-description) and [humanoid-rl-isaaclab](https://github.com/limxdynamics/humanoid-rl-isaaclab), Apache-2.0.
- **Retargeting:** [GMR](https://github.com/YanjieZe/GMR); optional roll calibration uses [LimX's GMR tools](https://github.com/limxdynamics/GMR).
- **Simulation:** [Isaac Lab](https://github.com/isaac-sim/IsaacLab), [Newton](https://github.com/newton-physics/newton), and [MuJoCo-Warp](https://github.com/google-deepmind/mujoco_warp).
- **Motion data:** LAFAN1 by Ubisoft La Forge, CC BY-NC-ND 4.0; upstream G1 retargets from [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset). Data and assets retain their separate licenses. The illustrative demo media is not covered by the code's MIT license.

## License

Code is released under the [MIT License](LICENSE), retaining the upstream notice and identifying the Oli additions. See the attribution above for third-party data, assets, and media.
