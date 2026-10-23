"""Side-by-side kinematic replay of two motion datasets (e.g. v2 vs v3 grounding).

Each clip is shown around its deepest floor penetration in dataset A (or from --start).
Usage:
  MUJOCO_GL=egl python tools/compare_motion.py --a data/lafan1_oli_v2 --b data/lafan1_oli_v3 \
      --out compare.mp4 --clips fallAndGetUp1_subject1 walk3_subject4
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_motion as rm  # noqa: E402

FPS = 30


def penetration_series(model, data, floor, x, stride=2):
    pen = np.zeros(len(x))
    for t in range(0, len(x), stride):
        rm.set_qpos(data, x[t])
        mujoco.mj_forward(model, data)
        for k in range(data.ncon):
            c = data.contact[k]
            if floor in (c.geom1, c.geom2):
                pen[t] = max(pen[t], -c.dist)
    return pen


def frame_pen(model, data, floor):
    mujoco.mj_forward(model, data)
    p = 0.0
    for k in range(data.ncon):
        c = data.contact[k]
        if floor in (c.geom1, c.geom2):
            p = max(p, -c.dist)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label_a", default="v2: clip-level foot grounding")
    ap.add_argument("--label_b", default="v3: per-frame whole-body grounding")
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--start", type=float, default=None, help="fixed start (s); default: centered on worst penetration")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    a = ap.parse_args()

    model = mujoco.MjModel.from_xml_string(rm.scene_xml())
    data = mujoco.MjData(model)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    renderer = mujoco.Renderer(model, a.height, a.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.4, 90.0, -8.0  # side view: ground contact is visible
    font = ImageFont.load_default(size=20)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    W = 2 * a.width
    ff = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{a.height}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out],
        stdin=subprocess.PIPE,
    )  # fmt: skip

    def panel(row, label, name, t):
        rm.set_qpos(data, row)
        pen = frame_pen(model, data, floor)
        cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.6]
        renderer.update_scene(data, camera=cam)
        img = Image.fromarray(renderer.render())
        d = ImageDraw.Draw(img)
        bad = pen > 0.02
        lines = (((12, 10), f"{label}"), ((12, 38), f"{name}  t={t / FPS:5.1f}s"),
                 ((12, a.height - 34), f"root z {row[2]:.2f} m   floor penetration {100 * pen:4.1f} cm"))  # fmt: skip
        for i, (xy, txt) in enumerate(lines):
            bb = d.textbbox(xy, txt, font=font)
            d.rectangle((bb[0] - 5, bb[1] - 3, bb[2] + 5, bb[3] + 3), fill=(255, 225, 225) if (bad and i == 2) else (255, 255, 255))
            d.text(xy, txt, fill=(170, 0, 0) if (bad and i == 2) else (20, 20, 20), font=font)
        return np.asarray(img)

    for name in a.clips:
        xa = np.loadtxt(os.path.join(a.a, name + ".csv"), delimiter=",")
        xb = np.loadtxt(os.path.join(a.b, name + ".csv"), delimiter=",")
        n = int(a.seconds * FPS)
        if a.start is None:
            worst = int(penetration_series(model, data, floor, xa).argmax())
            t0 = int(np.clip(worst - n // 2, 0, max(0, len(xa) - n)))
        else:
            t0 = int(a.start * FPS)
        for t in range(t0, min(len(xa), t0 + n)):
            frame = np.concatenate((panel(xa[t], a.label_a, name, t), panel(xb[t], a.label_b, name, t)), 1)
            frame[:, a.width - 1 : a.width + 1] = 40  # divider
            ff.stdin.write(frame.tobytes())
        print("rendered", name, f"from {t0 / FPS:.1f}s", flush=True)
    ff.stdin.close()
    ff.wait()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
