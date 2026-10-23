"""Export complete batched rollouts from configured motion start frames."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--selection', required=True)
parser.add_argument('--motion-dir', required=True)
parser.add_argument('--out-dir', required=True)
parser.add_argument('--max-seconds', type=float, default=None, help='Smoke test only')
parser.add_argument('--seed', type=int, default=18000)
pre, _ = parser.parse_known_args()
sys.path.insert(0, str(Path(pre.repo).resolve()))
os.environ.setdefault('SM_ROBOT', 'oli')
from isaaclab.app import add_launcher_args, launch_simulation
add_launcher_args(parser)
args = parser.parse_args()

import numpy as np
import torch
import supermimic.networks as networks
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import RobotKinematics
from supermimic.mdp import MAX_STEP_REWARD, OBS_DIM, MDP
from supermimic.motion import CONTROL_DT, GOAL_FRAME_DIM, MotionLib
from supermimic.networks import Policy
from supermimic.robot import default_joint_pos_list
from supermimic.rotations import mat_to_quat


@torch.no_grad()
def record():
    selected = json.loads(Path(args.selection).read_text())
    assert selected and len({c['name'] for c in selected}) == len(selected)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    dev = 'cuda:0'
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    kin = RobotKinematics(dev)
    files = [str(Path(args.motion_dir) / (c['name'] + '.csv')) for c in selected]
    motion = MotionLib(files, kin, dev)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
    offsets = torch.tensor([c['start_frame_50hz'] for c in selected], device=dev)
    starts = motion.clip_start + offsets
    ends = motion.clip_end.clone()
    assert bool(((offsets >= 0) & (starts < ends)).all())
    if args.max_seconds is not None:
        if args.max_seconds < CONTROL_DT:
            raise ValueError('--max-seconds must include at least one control step (0.02 s)')
        ends = torch.minimum(ends, starts + int(args.max_seconds / CONTROL_DT))
    counts = ends - starts
    nsteps = int(counts.max())
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    conf = ck['args']
    K, H = conf['goal_frames'], conf['hist']
    networks.AMP_DTYPE = torch.bfloat16 if conf.get('amp', 'none') == 'bf16' else None
    pi = Policy(OBS_DIM, GOAL_FRAME_DIM * K, hist=H).to(dev)
    pi.load_state_dict(ck['pi'])
    pi.eval()
    sim_cfg = make_sim(dev, args)
    records = []
    with launch_simulation(sim_cfg, args):
        env = TrackingSim(sim_cfg, len(selected), motion, mdp, eval_mode=True)
        env.done_eval[:] = False
        env.failed_eval[:] = False
        env.reset(torch.arange(len(selected), device=dev), starts)
        env.refresh()
        # A compact GPU buffer avoids an expensive CPU transfer on every step.
        initial = env.state()
        assert torch.allclose(initial[0], motion.root_p[starts], atol=1e-4)
        expected_q = motion.q[starts].clamp(kin.q_lo, kin.q_hi)
        assert torch.allclose(initial[4], expected_q, atol=1e-4)
        buffers = {
            'root_p': torch.empty(nsteps + 1, len(selected), 3, device=dev),
            'root_quat': torch.empty(nsteps + 1, len(selected), 4, device=dev),
            'q': torch.empty(nsteps + 1, len(selected), initial[4].shape[-1], device=dev),
        }
        first_term = torch.full((len(selected),), -1, dtype=torch.long, device=dev)
        returns = torch.zeros(len(selected), device=dev)
        hist = None
        started = time.monotonic()
        for k in range(nsteps + 1):
            state = env.state()
            buffers['root_p'][k] = state[0]
            buffers['root_quat'][k] = mat_to_quat(state[1])
            buffers['q'][k] = state[4]
            if k % 500 == 0 or k == nsteps:
                active = k <= counts
                assert all(bool(torch.isfinite(x[active]).all()) for x in state), 'Non-finite state'
                print(f'RECORD {k}/{nsteps} elapsed={time.monotonic()-started:.1f}s', flush=True)
            if k == nsteps:
                break
            # Finished environments keep stepping, but no further states enter their exports.
            # Clamp their reference before the next step so f+1 never leaves the clip.
            env.f = torch.minimum(env.f, ends - 1)
            obs = mdp.obs(state, env.a_prev, env.xi, env.f)
            hist = obs.unsqueeze(1).repeat(1, H, 1) if hist is None else torch.cat((hist[:, 1:], obs.unsqueeze(1)), 1)
            action = pi(hist, motion.goal(env.f, K))
            out = env.step(action)
            active = k < counts
            newly_failed = active & out['term'] & (first_term < 0)
            first_term[newly_failed] = k + 1
            returns += torch.where(active, out['r'], 0.0)
        digest = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        for c, choice in enumerate(selected):
            length = int(counts[c]) + 1
            frames = torch.arange(starts[c], ends[c] + 1, device=dev)
            payload = {key: value[:length, c].cpu().numpy() for key, value in buffers.items()}
            payload.update(f=frames.cpu().numpy(), ref_root_p=motion.root_p[frames].cpu().numpy(),
                           ref_root_R=motion.root_R[frames].cpu().numpy(), ref_q=motion.q[frames].cpu().numpy())
            assert all(np.isfinite(value).all() for value in payload.values())
            assert len(payload['f']) == length and np.all(np.diff(payload['f']) == 1)
            meta = dict(choice, checkpoint_iteration=int(ck["iteration"]), checkpoint_sha256=digest,
                        source_start_seconds=int(offsets[c]) * CONTROL_DT,
                        reference_duration_seconds=(length-1) * CONTROL_DT,
                        source_full_duration_seconds=(int(motion.clip_len[c])-1) * CONTROL_DT,
                        first_termination_seconds=None if int(first_term[c]) < 0 else int(first_term[c]) * CONTROL_DT,
                        sample_count=length, control_fps=50, smoke_test=args.max_seconds is not None,
                        full_tail_recorded=int(ends[c]) == int(motion.clip_end[c]),
                        continued_rollout_mean_reward=float(returns[c] / (counts[c] * MAX_STEP_REWARD)),
                        note='Fresh initialization after T-pose; physics continues after any termination threshold. This is not the original benchmark evaluation.')
            stem = outdir / choice['name']
            np.savez_compressed(str(stem) + '.npz', **payload)
            Path(str(stem) + '.json').write_text(json.dumps(meta, indent=2) + '\n')
            records.append(meta)
            print('SAVED', json.dumps(meta), flush=True)
    (outdir / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    record()
