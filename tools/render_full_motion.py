#!/usr/bin/env python3
"""Render a continuous 50 Hz rollout, including states after tracking failure.

first_termination_seconds is relative to the beginning of this exported segment.
The NPZ contains per-state reference arrays, not a global motion library.
"""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=Path(__file__).resolve().parents[1], type=Path)
    parser.add_argument("--traj", required=True, type=Path)
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--preview", type=float, help="Render only the first N seconds")
    return parser.parse_args()


def frame_indices(count, duration, fps=25, sample_hz=50):
    """Sample at 25 Hz and retain the terminal state within one output frame."""
    indices = list(range(0, count, sample_hz // fps))
    if indices[-1] != count - 1:
        indices.append(count - 1)
    # Appending an off-grid final state can add 1.5 frames of container duration.
    while len(indices) > 1 and len(indices) / fps > duration + 1 / fps + 1e-8:
        indices.pop(-2)
    if abs(len(indices) / fps - duration) > 1 / fps + 1e-8:
        raise ValueError("Trajectory sample count and reference duration disagree")
    return indices


def main():
    args = arguments()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("SM_ROBOT", "oli")
    module_path = args.repo.resolve() / "tools" / "render.py"
    spec = importlib.util.spec_from_file_location("supermimic_video_scene", module_path)
    scene = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scene)
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    meta = json.loads(args.meta.read_text())
    for key in ("name", "checkpoint_iteration", "source_start_seconds",
                "first_termination_seconds", "reference_duration_seconds"):
        if key not in meta:
            raise ValueError(f"Missing metadata field: {key}")
    source_start = float(meta["source_start_seconds"])
    reference_duration = float(meta["reference_duration_seconds"])
    first_termination = meta["first_termination_seconds"]
    if first_termination is not None:
        first_termination = float(first_termination)
    if not math.isfinite(source_start) or not math.isfinite(reference_duration) or reference_duration < 0:
        raise ValueError("Source start and reference duration must be finite; duration must be nonnegative")
    if first_termination is not None and (not math.isfinite(first_termination) or first_termination < 0):
        raise ValueError("first_termination_seconds must be null or a finite nonnegative number")
    if args.preview is not None and (not math.isfinite(args.preview) or args.preview <= 0):
        raise ValueError("--preview must be positive and finite")

    with np.load(args.traj, allow_pickle=False) as archive:
        keys = ("root_p", "root_quat", "q", "ref_root_p", "ref_root_R", "ref_q", "f")
        data = {key: archive[key] for key in keys}
    count = len(data["root_p"])
    if count < 1 or data["q"].ndim != 2:
        raise ValueError("Expected a nonempty single-clip trajectory")
    joints = data["q"].shape[1]
    shapes = {"root_p": (count, 3), "root_quat": (count, 4), "q": (count, joints),
              "ref_root_p": (count, 3), "ref_root_R": (count, 3, 3),
              "ref_q": (count, joints), "f": (count,)}
    for key, expected in shapes.items():
        if data[key].shape != expected or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid {key}: expected finite values with shape {expected}, got {data[key].shape}")
    sample_duration = (count - 1) / 50
    if abs(reference_duration - sample_duration) > 0.020001:
        raise ValueError(f"Reference duration {reference_duration} disagrees with {count} states at 50 Hz")
    render_count = count
    duration = reference_duration
    if args.preview is not None:
        render_count = min(count, int(math.floor(args.preview * 50 + 1e-8)) + 1)
        duration = min(reference_duration, (render_count - 1) / 50)
    indices = frame_indices(render_count, duration)

    model = mujoco.MjModel.from_xml_string(scene.two_robot_xml())
    if model.nq != 2 * (7 + joints):
        raise ValueError(f"Scene has {model.nq} qpos values; trajectory requires {2 * (7 + joints)}")
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.55
    state = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 540, 960)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.6, 135.0, -12.0
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        try:
            font = ImageFont.load_default(size=18)
        except TypeError:
            font = ImageFont.load_default()

    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".partial.mp4")
    first_frame_path = output.with_suffix(".first.jpg")
    report_path = output.with_suffix(".render.json")
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "960x540", "-r", "25", "-i", "-", "-an", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-crf", "20", "-threads", "2", "-movflags", "+faststart", str(partial)]
    first_frame = None
    try:
        with tempfile.TemporaryFile() as errors:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
            try:
                render_started = time.monotonic()
                for output_index, index in enumerate(indices):
                    ghost = 7 + joints
                    state.qpos[:3] = data["root_p"][index]
                    state.qpos[3:7] = data["root_quat"][index]
                    state.qpos[7:ghost] = data["q"][index]
                    state.qpos[ghost:ghost + 3] = data["ref_root_p"][index]
                    state.qpos[ghost:ghost + 2] = data["root_p"][index, :2]
                    state.qpos[ghost + 3:ghost + 7] = scene._mat_to_quat(data["ref_root_R"][index])
                    state.qpos[ghost + 7:] = data["ref_q"][index]
                    mujoco.mj_kinematics(model, state)
                    camera.lookat[:] = [state.qpos[0], state.qpos[1], 0.95]
                    renderer.update_scene(state, camera=camera)
                    frame = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(frame)
                    elapsed = index / 50
                    exceeded = first_termination is not None and elapsed + 1e-8 >= first_termination
                    lines = [
                        ((16, 12), f"Oli | {meta['name']} | checkpoint {meta['checkpoint_iteration']}"),
                        ((16, 40), f"elapsed {elapsed:.2f}s / {reference_duration:.2f}s | source {source_start + elapsed:.2f}s"),
                        ((16, 474), "solid: simulated policy | blue ghost: root-aligned reference"),
                        ((16, 504), "Tracking threshold exceeded; rollout continues" if exceeded else "Continuous physics rollout"),
                    ]
                    for position, text in lines:
                        bounds = draw.textbbox(position, text, font=font)
                        draw.rectangle((bounds[0] - 5, bounds[1] - 3, bounds[2] + 5, bounds[3] + 3), fill="white")
                        draw.text(position, text, fill=(20, 20, 20), font=font)
                    if first_frame is None:
                        first_frame = frame.copy()
                    encoder.stdin.write(np.asarray(frame).tobytes())
                    if (output_index + 1) % 1000 == 0:
                        print(f"RENDER {output_index + 1}/{len(indices)} elapsed={time.monotonic() - render_started:.1f}s", flush=True)
                encoder.stdin.close()
                return_code = encoder.wait()
            except BaseException:
                encoder.kill()
                encoder.wait()
                errors.seek(0)
                diagnostic = errors.read().decode("utf-8", errors="replace").strip()
                if diagnostic:
                    print(diagnostic, flush=True)
                raise
            if return_code:
                errors.seek(0)
                raise RuntimeError(f"ffmpeg exited {return_code}: {errors.read().decode('utf-8', errors='replace')}")
        if not partial.exists() or not partial.stat().st_size:
            raise RuntimeError("ffmpeg produced no video")
        os.replace(partial, output)
        first_frame.save(first_frame_path, quality=92)
        report = dict(meta)
        report.update({"trajectory_path": str(args.traj.resolve()), "metadata_path": str(args.meta.resolve()),
                       "output_path": str(output), "source_sample_hz": 50, "video_fps": 25,
                       "source_state_count": count, "rendered_source_state_count": render_count,
                       "video_frame_count": len(indices), "video_duration_seconds": len(indices) / 25,
                       "render_reference_duration_seconds": duration,
                       "duration_difference_seconds": len(indices) / 25 - duration,
                       "first_source_index": indices[0], "last_source_index": indices[-1],
                       "first_reference_frame": int(data["f"][indices[0]]),
                       "last_reference_frame": int(data["f"][indices[-1]]),
                       "final_state_included": indices[-1] == render_count - 1,
                       "full_trajectory_included": render_count == count,
                       "preview_seconds": args.preview, "ghost_xy_alignment": "actual_root",
                       "continues_after_tracking_termination": True,
                       "first_frame_path": str(first_frame_path)})
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"video": str(output), "report": str(report_path), "frames": len(indices),
                          "duration_seconds": len(indices) / 25}), flush=True)
    finally:
        renderer.close()
        if partial.exists():
            partial.unlink()


if __name__ == "__main__":
    main()
