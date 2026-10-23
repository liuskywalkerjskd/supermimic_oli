"""Kinematic replay of retargeted motion CSVs on the Oli MJCF (no physics), plus per-clip checks.

Interactive (needs a display):
  python tools/replay_motion.py --motion_dir data/lafan1_oli_v2 --viewer
    space: pause/resume   right/left arrow: next/previous clip   up/down: faster/slower   r: restart clip
Video (headless, EGL):
  MUJOCO_GL=egl python tools/replay_motion.py --motion_dir data/lafan1_oli_v2 --out replay.mp4 --seconds 8
Report only:
  python tools/replay_motion.py --motion_dir data/lafan1_oli_v2 --report

The report lists, per clip: max ground penetration of any collision geom and the body it belongs
to, the fraction of frames penetrating more than 2 cm, max foot height while the other foot is up
(floating), and the 99.9th-percentile joint speed.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import time

import mujoco
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MJCF = os.path.join(ROOT, "data", "oli_description", "oli_31dof.xml")
FPS = 30.0


def scene_xml() -> str:
    xml = open(MJCF).read()
    meshdir = os.path.join(os.path.dirname(os.path.abspath(MJCF)), "meshes") + "/"
    xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{meshdir}"', xml)
    floor = (
        '<light pos="0 0 6" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8"/>'
        '<geom name="floor" type="plane" size="0 0 0.05" material="grid"/>'
    )
    xml = xml.replace("<worldbody>", "<worldbody>" + floor, 1)
    assets = (
        '<texture name="grid" type="2d" builtin="checker" rgb1="0.92 0.93 0.94" rgb2="0.80 0.82 0.85" width="512" height="512"/>'
        '<material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>'
        '<texture name="sky" type="skybox" builtin="gradient" rgb1="0.95 0.96 0.98" rgb2="0.75 0.8 0.88" width="256" height="256"/>'
    )
    xml = xml.replace("<asset>", "<asset>" + assets, 1)
    xml = xml.replace("<mujoco", "<mujoco", 1).replace(
        "<asset>", '<visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096"/></visual><asset>', 1
    )
    return xml


def load_clips(motion_dir: str, names: list[str] | None):
    files = sorted(glob.glob(os.path.join(motion_dir, "*.csv")))
    clips = {os.path.splitext(os.path.basename(f))[0]: f for f in files}
    if names:
        clips = {n: clips[n] for n in names}
    return clips


def set_qpos(data, row):
    data.qpos[0:3] = row[0:3]
    data.qpos[3:7] = (row[6], row[3], row[4], row[5])  # CSV quat is xyzw, MuJoCo wxyz
    data.qpos[7:] = row[7:]


def clip_report(model, data, x: np.ndarray, floor_id: int, stride: int = 2) -> dict:
    pen, pen_body, n_pen = 0.0, "-", 0
    lf, rf = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("left_foot", "right_foot"))
    foot_low = []
    frames = range(0, len(x), stride)
    for t in frames:
        set_qpos(data, x[t])
        mujoco.mj_forward(model, data)
        worst = 0.0
        for k in range(data.ncon):
            c = data.contact[k]
            if floor_id not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == floor_id else c.geom1
            if -c.dist > worst:
                worst = -c.dist
                if worst > pen:
                    pen, pen_body = worst, model.body(model.geom_bodyid[other]).name
        n_pen += worst > 0.02
        z = []
        for g in (lf, rf):
            R = data.geom_xmat[g].reshape(3, 3)
            sx, sy, sz = model.geom_size[g]
            corners = np.array([[a, b, -sz] for a in (-sx, sx) for b in (-sy, sy)])
            z.append((data.geom_xpos[g] + corners @ R.T)[:, 2].min())
        foot_low.append(min(z))
    foot_low = np.array(foot_low)
    qd = np.abs(np.diff(x[:, 7:], axis=0)) * FPS
    return {
        "frames": len(x),
        "max_pen_cm": 100 * pen,
        "pen_body": pen_body,
        "pen>2cm_%": 100 * n_pen / len(frames),
        # lowest foot above ground for a sustained stretch = whole robot floating (no flight phase lasts > 0.5 s)
        "float_cm": 100 * float(np.max(np.convolve(foot_low, np.ones(8) / 8, mode="valid"))),
        "qd_p999": float(np.percentile(qd.max(1), 99.9)),
        "qd_first10": float(qd[:10].max()),
    }


def run_viewer(model, clips: dict):
    import mujoco.viewer

    names = list(clips)
    st = {"i": 0, "t": 0.0, "pause": False, "speed": 1.0, "reload": True}

    def key(k):
        if k == 32:
            st["pause"] = not st["pause"]
        elif k == 262:  # right
            st["i"] = (st["i"] + 1) % len(names); st["reload"] = True  # noqa: E702
        elif k == 263:  # left
            st["i"] = (st["i"] - 1) % len(names); st["reload"] = True  # noqa: E702
        elif k == 265:  # up
            st["speed"] = min(st["speed"] * 2, 8.0)
        elif k == 264:  # down
            st["speed"] = max(st["speed"] / 2, 0.125)
        elif k == ord("R"):
            st["t"] = 0.0

    data = mujoco.MjData(model)
    x = None
    with mujoco.viewer.launch_passive(model, data, key_callback=key) as v:
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 3.5, -12, 135
        last = time.time()
        while v.is_running():
            if st["reload"]:
                x = np.loadtxt(clips[names[st["i"]]], delimiter=",")
                st["t"], st["reload"] = 0.0, False
                print(f"[{st['i'] + 1}/{len(names)}] {names[st['i']]}  ({len(x) / FPS:.1f} s)", flush=True)
            now = time.time()
            if not st["pause"]:
                st["t"] += (now - last) * st["speed"]
            last = now
            f = int(st["t"] * FPS) % len(x)
            set_qpos(data, x[f])
            mujoco.mj_kinematics(model, data)
            v.cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.9]
            v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                         f"{names[st['i']]}  [{st['i'] + 1}/{len(names)}]", f"t={f / FPS:6.2f}s  x{st['speed']:g}"))  # fmt: skip
            v.sync()
            time.sleep(1 / 60)


def run_video(model, clips: dict, out: str, seconds: float, start: float, width: int, height: int, fps: int, reports: dict):
    from PIL import Image, ImageDraw, ImageFont

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.6, 135.0, -12.0
    font = ImageFont.load_default(size=22)
    ff = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    for name, path in clips.items():
        x = np.loadtxt(path, delimiter=",")
        t0 = int(start * FPS)
        t1 = len(x) if seconds <= 0 else min(len(x), t0 + int(seconds * FPS))
        r = reports.get(name)
        info = f"pen {r['max_pen_cm']:.1f}cm ({r['pen_body']})  qd99.9 {r['qd_p999']:.0f} rad/s" if r else ""
        for k in np.arange(t0, t1, FPS / fps):
            f = int(k)
            set_qpos(data, x[f])
            mujoco.mj_kinematics(model, data)
            cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.9]
            renderer.update_scene(data, camera=cam)
            img = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)
            for xy, txt in (((16, 12), f"{name}  t={f / FPS:5.1f}s"), ((16, height - 38), f"retargeted LAFAN1 -> Oli (kinematic replay)  {info}")):
                bb = draw.textbbox(xy, txt, font=font)
                draw.rectangle((bb[0] - 6, bb[1] - 4, bb[2] + 6, bb[3] + 4), fill=(255, 255, 255))
                draw.text(xy, txt, fill=(20, 20, 20), font=font)
            ff.stdin.write(np.asarray(img).tobytes())
        print("rendered", name, flush=True)
    ff.stdin.close()
    ff.wait()
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_dir", default=os.path.join(ROOT, "data", "lafan1_oli_v2"))
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--out", default=None, help="mp4 path")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seconds", type=float, default=8.0, help="per clip in the video; <=0 for the full clip")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    a = ap.parse_args()

    model = mujoco.MjModel.from_xml_string(scene_xml())
    clips = load_clips(a.motion_dir, a.clips)
    reports = {}
    if a.report or a.out:
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        data = mujoco.MjData(model)
        print(f"{'clip':26s} {'frames':>6s} {'pen_cm':>7s} {'pen_body':>22s} {'pen>2cm%':>8s} {'float_cm':>8s} {'qd99.9':>7s} {'qd_0:10':>7s}")
        for n, p in clips.items():
            r = reports[n] = clip_report(model, data, np.loadtxt(p, delimiter=","), floor)
            print(f"{n:26s} {r['frames']:6d} {r['max_pen_cm']:7.1f} {r['pen_body']:>22s} {r['pen>2cm_%']:8.1f} {r['float_cm']:8.1f} {r['qd_p999']:7.1f} {r['qd_first10']:7.1f}", flush=True)  # fmt: skip
    if a.out:
        run_video(model, clips, a.out, a.seconds, a.start, a.width, a.height, a.fps, reports)
    if a.viewer:
        run_viewer(model, clips)


if __name__ == "__main__":
    main()
