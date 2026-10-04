# Evaluation scripts

Activate the environment, then run each script from this folder:

```bash
source ../env/bin/activate
python <script>.py
```

Outputs go to `output/` at the repository root (git-ignored). The CARI v2 training and evaluation scripts,
`train.py` and `eval.py`, are in the repository root (see [`../README.md`](../README.md)); this folder holds their
helpers and the other experiments:

```bash
python example.py              # paper demo: simulated reaching, inverse gILQG vs MaxEnt (-r 50 -t 50: paper settings, --cpu)
python predict_motion.py       # moving-window prediction on a simulated 2-link arm (--benchmark: real-time rate)
python predict_motion_3d.py    # the same on a simulated 3D arm
python predict_motion_hkm.py   # the same with the 28-DOF human model on a simulated reach
python compare_spaces.py       # joint space vs Cartesian prediction of a simulated 3D reach
python save_results.py ../output/latest/obs30/results.json   # regenerate the tables of a saved eval.py run
python goal_inference.py       # online prediction on whole CARI sessions, goal inferred / known (config `online`;
                               #   online.reuse=<run>: tables and figures only)
python export_replay.py --subject sub_4   # CARI session -> .npz + goals .yaml for the ROS 2 replay (docker/README.md)
```

Synthetic reaches with Kimodo (NVIDIA kinematic motion diffusion; clone in `../../kimodo`, its own venv
`../../kimodo/.venv`: python 3.14, torch cu128, `pip install -e .`), to test whether they are representative of CARI:

```bash
python kimodo_reaches.py layout                                   # CARI cells -> output/kimodo/layouts.json
../../kimodo/.venv/bin/python kimodo_reaches.py generate          # 200 clips (10 subjects x 4 tasks x 5) -> clips.npz
../../kimodo/.venv/bin/python kimodo_reaches.py encode            # prompts -> text_features.pt (Llama access, ~16 GB RAM)
../../kimodo/.venv/bin/python kimodo_reaches.py generate --text --constraints loose --out clips_text.npz
python goal_inference.py online.source=kimodo online.clips=output/kimodo/clips_text.npz
python goal_inference.py online.source=kimodo online.clips=output/kimodo/clips_text.npz online.noise=true
python kimodo_reaches.py compare output/latest_goal_inference output/latest_goal_inference_kimodo \
    output/latest_goal_inference_kimodo_noise                     # -> output/kimodo/comparison.html
```

The clips are generated from wrist keyframes only (empty prompt: Kimodo's unconditional text branch, so the gated
Llama-3 text encoder is not needed) in each subject's cell, scaled to the subject's arm length; `goal_inference.py
--source kimodo` puts them through our IK and the same online replay, with the filter settings and nominal durations
of the real run (no re-tuning on the synthetic data).

- `cari_kinematic.py`: module shared by `train.py` and `eval.py` (trial frames, handover filter, environment,
  baselines, IOC fit);
- `save_results.py`: aggregate tables of `eval.py` (CSV, JSON, HTML with the best method per metric in bold);
- `kimodo_reaches.py`: Kimodo clips of the CARI tasks and the real-vs-synthetic comparison (see above);
- `cari_sessions.py`: CARI v2 sessions (instructions 0-8 of a subject concatenated: home, object 1, home, object 2,
  home, object 3, home, robot, home) and the goal locations of the cell, measured on the other velocities; used by
  `goal_inference.py` and `export_replay.py`.

`goal_inference.py` replays every session as the ROS 2 node sees it (raw IK angles, last 1 s observed, a prediction
every 1/15 s over a 1 s window) with the 8 hypotheses of the cell (7 goal locations with their hand, idle), and
compares the published prediction (most probable hypothesis) with the oracle goal, the known task schedule
(next goal, unknown onset), constant velocity and a frozen pose; the filter settings are selected
leave-one-subject-out on the balanced goal accuracy, and the cue prior is ablated. Output:
`output/goal_inference_<ts>/` (`summary.html`, accuracy-vs-progress and session-timeline figures, `sessions.pkl`
with the predictions of every tick, for `--reuse`).

`check_report.py` gathers the latest train / eval / goal-inference / Kimodo runs into `output/check_<ts>/`
(`overview.png`, `index.html` with links to every run's tables and figures).
