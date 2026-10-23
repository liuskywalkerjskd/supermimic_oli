"""Mirror a SuperMimic run's train_log.csv and eval.csv to SwanLab (runs beside training).

Polls the CSVs and logs new rows, using the training iteration as the step, so it can be started
or restarted at any time without touching the training process. Uses the existing SwanLab login
(~/.swanlab/.netrc). Exits after the final checkpoint has been evaluated when --until is given.

  python tools/swanlab_sync.py --run_dir logs/oli_lafan1_v5 --project supermimic-oli --name oli_lafan1_v5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

TRAIN_GROUPS = {
    "wm_loss": "world_model", "wm_err8": "world_model",
    "pi_loss": "policy", "imag_rew": "policy", "pi_gradnorm": "policy", "imag_survival": "policy",
    "v_loss": "critic",
    "sim_rew": "sim", "ep_len": "sim", "ep_ret_norm": "sim", "term_frac": "sim",
    "it_per_s": "perf", "lr": "perf", "wall_h": "perf", "sim_steps": "perf",
}  # fmt: skip


def read_rows(path: str, start: int):
    if not os.path.exists(path):
        return [], start
    with open(path) as f:
        lines = f.readlines()
    # Writers terminate complete records with a newline. A successful evaluation
    # legitimately has an empty "failed" field, which must still be uploaded.
    if lines and not lines[-1].endswith("\n"):
        lines.pop()
    rows = list(csv.DictReader(lines))
    if rows and (None in rows[-1] or any(v is None or (v == "" and k != "failed") for k, v in rows[-1].items())):
        rows = rows[:-1]
    return rows[start:], len(rows)


def main():
    import swanlab

    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--project", default="supermimic-oli")
    ap.add_argument("--name", default=None)
    ap.add_argument("--description", default="")
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--eval_tag", default="", help="e.g. _on40 to read eval_on40.csv")
    ap.add_argument("--until", type=int, default=None, help="stop after this iteration is evaluated")
    a = ap.parse_args()

    args_path = os.path.join(a.run_dir, "args.json")
    config = json.load(open(args_path)) if os.path.exists(args_path) else {}
    config["robot"] = os.environ.get("SM_ROBOT", "oli")
    run = swanlab.init(project=a.project, experiment_name=a.name or os.path.basename(a.run_dir.rstrip("/")),
                       description=a.description, config=config)  # fmt: skip
    n_train = n_eval = 0
    last_eval_it = -1
    while True:
        rows, n_train = read_rows(os.path.join(a.run_dir, "train_log.csv"), n_train)
        for r in rows:
            it = int(r["iteration"])
            swanlab.log({f"{TRAIN_GROUPS.get(k, 'misc')}/{k}": float(v) for k, v in r.items() if k != "iteration"}, step=it)
        rows, n_eval = read_rows(os.path.join(a.run_dir, f"eval{a.eval_tag}.csv"), n_eval)
        for r in rows:
            it = int(r["iteration"])
            swanlab.log({"eval/norm_return": float(r["norm_return"]), "eval/norm_return_framew": float(r["norm_return_framew"]),
                         "eval/n_failed": float(r["n_failed"]), "eval/train_wall_h": float(r["train_wall_h"])}, step=it)  # fmt: skip
            per_clip = os.path.join(a.run_dir, f"ckpt_{it:06d}_eval{a.eval_tag}.json")
            if os.path.exists(per_clip):
                swanlab.log({f"eval_clip/{k}": float(v) for k, v in json.load(open(per_clip))["per_clip"].items()}, step=it)
            last_eval_it = it
        if a.until is not None and last_eval_it >= a.until:
            break
        time.sleep(a.poll)
    run.finish()


if __name__ == "__main__":
    main()
