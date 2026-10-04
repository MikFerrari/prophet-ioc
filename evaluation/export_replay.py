#!/usr/bin/env python3
"""Exports a CARI v2 session to a plain .npz for the ROS 2 replay (human_motion_predictor replay_cari), with the goal
locations of the cell as a parameter file of the predictor.

The session is the concatenation of the instructions 0-8 of a subject at one velocity (cari_sessions.py: home,
object 1, home, object 2, home, object 3, home, robot, home). The CARI cache is a pandas pickle written with numpy
2, which the ROS 2 Jazzy container (numpy 1.26) cannot read: the replay reads this file instead. It holds the raw ZED
BODY_18 keypoints of the recording (human_kp0..17 of datasets/cari_v2/6_preprocessed, ZED input) and their IK angles
(joints input). The goals (7 locations with their hand) are measured on the sessions of the other velocities.
Run on the host:

    python export_replay.py --subject sub_4      # -> output/replay/sub_4_FAST.npz, output/replay/sub_4_FAST_goals.yaml
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import yaml

import cari_sessions as cs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="sub_4")
    ap.add_argument("--velocity", default="FAST")
    ap.add_argument("--csv", default=str(cs.CSV), help="preprocessed CARI v2 table with the raw ZED keypoints")
    ap.add_argument("--out-dir", default="output/replay")
    args = ap.parse_args()

    S = cs.load_session(args.subject, args.velocity, csv=None)
    cols = [f"human_kp{i}_{a}" for i in range(18) for a in "xyz"]
    df = pd.read_csv(args.csv, usecols=["Subject", "Velocity", "Instruction_id", "Time"] + cols)
    rows = df[(df.Subject == args.subject) & (df.Velocity == args.velocity)].sort_values(["Instruction_id", "Time"])
    if len(rows) != len(S.q28_raw):
        raise RuntimeError("the keypoint table and the CARI cache do not have the same frames")
    goals = cs.layout_goals(args.subject, [v for v in ("FAST", "MEDIUM", "SLOW") if v != args.velocity])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{args.subject}_{args.velocity}"
    np.savez(out / f"{name}.npz", q28=S.q28_raw.astype(np.float64), body_params=S.body_params.astype(np.float64),
             dt=float(S.dt), zed_kpts=rows[cols].to_numpy().reshape(-1, 18, 3), zed_body_format=0,
             movements=np.array([[m.segment, m.onset, m.offset] for m in S.movements]),
             movement_goals=np.array(["/".join(m.goals) for m in S.movements]), session=name)
    params = {"human_motion_predictor": {"ros__parameters": {
        "goal_names": list(goals),
        "goal_positions": [round(float(x), 4) for pos in goals.values() for x in pos],
        "goal_hands": ["both" if len(cs.GOAL_HANDS[g]) == 2 else cs.GOAL_HANDS[g][0] for g in goals]}}}
    (out / f"{name}_goals.yaml").write_text(
        f"# Goal locations of the CARI v2 cell of {args.subject} (wrist at the end of each instruction, measured on "
        f"the other velocities), frame of the replay\n" + yaml.safe_dump(params, sort_keys=False))
    print(f"{out / name}.npz: {len(S.q28_raw)} frames ({len(S.q28_raw) * S.dt:.1f} s) at {1 / S.dt:.0f} Hz, "
          f"{len(S.movements)} movements; goals -> {out / name}_goals.yaml")


if __name__ == "__main__":
    main()
