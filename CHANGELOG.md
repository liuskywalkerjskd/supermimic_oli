# Changelog

## [0.1.0] - 2026-10-23

### Features
- Adapt the upstream implementation to 31-DoF LimX Oli HU_D04_01, retaining G1 through `SM_ROBOT`.
- Add Oli conversion, LAFAN1 retargeting, calibrated IK configurations, grounding, and diagnostics.
- Add motion-directory selection, tagged evaluations, and optional CSV-to-SwanLab logging.
- Export complete batched rollouts from per-clip start frames with persistent termination annotations.
- Publish English and Simplified Chinese documentation, recorded metrics, and an Oli demo.
- Preserve robot selection across existing tmux servers and accept successful evaluation rows with empty failure lists in SwanLab synchronization.

### Design Rationale
- Derive state and observation dimensions from the selected robot without duplicating the learning algorithm.
- Use Oli's serial training model and actuator parameters for simulation.
- Separate original benchmark evaluation from exports starting after T-pose because their initial states differ.
- Continue physics after a threshold crossing so full-length videos show the actual trajectory to the final reference frame.

### Notes & Caveats
- Based on `zita-ch/supermimic_vibed` at `a1508d42c9d1699a52e8233ab16d08c488441ab9`; upstream MIT notice and history are retained.
- Motion data, robot assets, generated USD files, and trained checkpoints are not bundled.
- The selected iteration-18,000 benchmark passed 36/40 clips. Separate post-T-pose exports passed 35/36 selected clips; `walk3_subject4` crossed the threshold at 21.32 s.
- Physical robot deployment has not been validated in this project.
