#!/usr/bin/env python3
"""Synthetic CARI-like reaches generated with Kimodo (NVIDIA kinematic motion diffusion model), to test whether
they are representative of the real CARI v2 data for the online prediction with goal inference.

For every CARI subject, clips of 6 s are generated in the subject's cell: hands at home, a reach of one hand to
object 1 (right), object 2 (left), object 3 (right) or of both hands to the robot, a hold, and the return home. The
goal locations are those of goal_inference.py (cari_sessions.layout_goals, wrist at the end of each instruction on
the other velocities). They are given to Kimodo as sparse wrist keyframes (end-effector constraints: wrist position
and rotation and hand end, rotation pointing the hand from the shoulder towards the target, palm down) with a random
schedule (leave home at 1.0-1.4 s, reach in 0.7-1.2 s, hold 0.6-1.0 s, return in 0.7-1.2 s), the other hand kept at
home, and the root kept in place (standing). By default the prompt is empty (unconditional text branch, zero text
features), so the text encoder of Kimodo (LLM2Vec on Llama-3-8B, gated) is not needed. With --text, each clip gets a
prompt describing its task (PROMPTS: random paraphrase, speed adverb of the CARI velocity), encoded once by `encode`
(on the CPU, ~16 GB of RAM) and cached in output/kimodo/text_features.pt. --constraints loose keeps fewer keyframes
(moving hand: leave home, arrive, leave the goal, back home; other hand: start and end only; no root path) and puts
the hips of the goal keyframes at the subject's measured pelvis displacement for that goal (the hand constraints of
Kimodo also fix the hips at their keyframes), so that trunk and passive hand can move as the text and the motion prior
suggest. The cell is mapped to Kimodo's canonical frame (Y up,
facing +Z, root at the origin) with the subject's pelvis and heading at home (the pelvis at Kimodo's standing hip
height; the CARI legs are not usable, the table hides them from the camera), and scaled by the ratio of the arm
lengths, so that the layout keeps its size relative to the body; the generated motion is mapped back.

The SOMA joints give the 13 keypoints of the human kinematic model (shoulder = Arm, elbow = ForeArm, wrist = Hand,
hip = Leg, knee = Shin, ankle = Foot, head = nose, estimated from the eyes and the head rotation), and the gaze
(nose - mid-ears, as from the ZED skeleton). goal_inference.py --source kimodo then runs them through our IK and the
same replay as the real sessions (clean, or with the noise of the real recordings: --noise).

    python kimodo_reaches.py layout                                   # prophet env -> output/kimodo/layouts.json
    ../../kimodo/.venv/bin/python kimodo_reaches.py generate          # Kimodo env (src/kimodo/.venv) -> output/kimodo/clips.npz
    ../../kimodo/.venv/bin/python kimodo_reaches.py encode            # text prompts -> output/kimodo/text_features.pt
    ../../kimodo/.venv/bin/python kimodo_reaches.py generate --text --constraints loose --out clips_text.npz
    python goal_inference.py --source kimodo --filter-from output/latest_goal_inference [--noise]
    python kimodo_reaches.py compare <real run> <kimodo run> [<kimodo noisy run>]   # -> output/kimodo/comparison.html
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "kimodo"
KEYPOINTS = ["head", "left_shoulder", "left_elbow", "left_wrist", "left_hip", "left_knee", "left_ankle",
             "right_shoulder", "right_elbow", "right_wrist", "right_hip", "right_knee", "right_ankle"]
SOMA_JOINT = {"left_shoulder": "LeftArm", "left_elbow": "LeftForeArm", "left_wrist": "LeftHand",
              "left_hip": "LeftLeg", "left_knee": "LeftShin", "left_ankle": "LeftFoot",
              "right_shoulder": "RightArm", "right_elbow": "RightForeArm", "right_wrist": "RightHand",
              "right_hip": "RightLeg", "right_knee": "RightShin", "right_ankle": "RightFoot"}
TASKS = [("object_1", ("right",)), ("object_2", ("left",)), ("object_3", ("right",)), ("robot", ("right", "left"))]
# Prompts (Kimodo: "A person ...", medium detail, at most two behaviours), from the CARI instructions ("Reach object 1
# with RIGHT HAND", ...): {speed} is the adverb of the CARI velocity
PROMPTS = {
    "object_1": [
        "A person standing at a table {speed}reaches forward and to the right with their right hand to grab an object, "
        "then puts the hand back on the table.",
        "A person at a workbench leans forward and {speed}picks up an object on their right with the right hand, then "
        "returns the hand to rest.",
        "A person standing in front of a table {speed}reaches out with the right hand to an object on the right side, "
        "then brings the hand back.",
    ],
    "object_2": [
        "A person standing at a table {speed}reaches forward and to the left with their left hand to grab an object, "
        "then puts the hand back on the table.",
        "A person at a workbench leans forward and {speed}picks up an object on their left with the left hand, then "
        "returns the hand to rest.",
        "A person standing in front of a table {speed}reaches out with the left hand to an object on the left side, "
        "then brings the hand back.",
    ],
    "object_3": [
        "A person standing at a table {speed}reaches forward with their right hand to grab an object in front of them, "
        "then puts the hand back on the table.",
        "A person at a workbench leans forward and {speed}picks up an object in front of them with the right hand, then "
        "returns the hand to rest.",
        "A person standing in front of a table {speed}reaches across the table with the right hand, then brings the hand "
        "back.",
    ],
    "robot": [
        "A person standing at a table {speed}raises both hands forward to take an object held at chest height in front "
        "of them, then lowers both hands back to the table.",
        "A person at a workbench {speed}reaches up and forward with both hands to grab something in front of their "
        "chest, then puts both hands back down.",
        "A person standing in front of a table {speed}lifts both hands to an object in front of them, then returns "
        "the hands to the table.",
    ],
}
SPEED = {"FAST": "quickly ", "MEDIUM": "", "SLOW": "slowly "}


def all_prompts(velocity: str):
    return [t.format(speed=SPEED[velocity]) for task in PROMPTS for t in PROMPTS[task]]


# =============================================================================
# layout (nioc environment)
# =============================================================================
def layout(args):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "evaluation"))
    os.chdir(ROOT)
    import yaml
    import cari_sessions as cs
    import human_kinematic_model_jax as hkm

    subjects = args.subjects or yaml.safe_load((ROOT / "config/data/cari.yaml").read_text())["subjects"]
    out = {}
    for s in subjects:
        S = cs.load_session(s, args.velocity, csv=None)
        rest = S.kp[: S.movements[0].onset]                       # standing at home before the first reach
        K = hkm.KP_INDEX
        lhip, rhip = rest[:, K["left_hip"]], rest[:, K["right_hip"]]
        pelvis = 0.5 * (lhip + rhip).mean(axis=0)
        left = (lhip - rhip).mean(axis=0)
        left[2] = 0.0
        left /= np.linalg.norm(left)
        up = np.array([0.0, 0.0, 1.0])
        forward = np.cross(left, up)
        goals = cs.layout_goals(s)
        lean = cs.pelvis_offsets(s)
        out[s] = {"pelvis": pelvis.tolist(), "left": left.tolist(), "forward": forward.tolist(),
                  "pelvis_offsets": {k: v.tolist() for k, v in lean.items()},
                  "arm": float(S.body_params[3] + S.body_params[4]),
                  "goals": {k: v.tolist() for k, v in goals.items()}}
        print(f"{s}: arm {out[s]['arm']:.3f} m, pelvis {np.round(pelvis, 3)}, forward {np.round(forward, 2)}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "layouts.json").write_text(json.dumps({"velocity": args.velocity, "subjects": out}, indent=2))
    print(f"-> {OUT / 'layouts.json'}")


# =============================================================================
# generate (Kimodo environment)
# =============================================================================
class _ZeroText:
    """Text encoder of the empty prompt: zero features (Kimodo's unconditional text branch)."""

    def to(self, device=None, dtype=None):
        return self

    def __call__(self, texts):
        import torch
        texts = [texts] if isinstance(texts, str) else texts
        return torch.zeros(len(texts), 1, 4096), [1] * len(texts)


class _CachedText:
    """Text encoder from the features cached by `encode` (LLM2Vec: one pooled 4096-d vector per prompt); the empty
    prompt gives zeros."""

    def __init__(self, features):
        self.features = features

    def to(self, device=None, dtype=None):
        return self

    def __call__(self, texts):
        import torch
        texts = [texts] if isinstance(texts, str) else texts
        missing = [t for t in texts if t.strip() and t not in self.features]
        if missing:
            raise KeyError(f"prompts not encoded (run kimodo_reaches.py encode): {missing[:2]}")
        feats = [self.features[t].float() if t.strip() else torch.zeros(4096) for t in texts]
        return torch.stack(feats)[:, None], [1] * len(texts)


def encode(args):
    """Encodes the prompts of PROMPTS with Kimodo's text encoder (LLM2Vec on Llama-3-8B-Instruct, on the CPU) and
    caches the features."""
    import torch
    from kimodo.model import LLM2VecEncoder
    from kimodo.model.load_model import TEXT_ENCODER_PRESETS

    velocity = json.loads((OUT / "layouts.json").read_text())["velocity"]
    prompts = all_prompts(velocity)
    kw = dict(TEXT_ENCODER_PRESETS["llm2vec"]["kwargs"], device=args.device)
    print(f"loading the text encoder ({kw['base_model_name_or_path']}, {kw['dtype']}) on {args.device} ...", flush=True)
    enc = LLM2VecEncoder(**kw)
    path = OUT / "text_features.pt"
    cache = torch.load(path) if path.exists() else {}
    for t in prompts:
        if t not in cache:
            feat, _ = enc([t])
            cache[t] = feat[0, 0].to(torch.float16).cpu()
            print(f"  {t[:90]}...  |f| = {float(cache[t].float().norm()):.2f}", flush=True)
    torch.save(cache, path)
    print(f"{len(cache)} prompts -> {path}")


class CellFrame:
    """CARI frame (z up) <-> Kimodo canonical frame (y up, +z forward, +x left, root at the origin), scaled by s
    about the subject's pelvis, which goes to the hip height root_y."""

    def __init__(self, lay, scale, root_y):
        self.p0 = np.array(lay["pelvis"])
        self.left, self.fwd = np.array(lay["left"]), np.array(lay["forward"])
        self.s, self.root_y = scale, root_y

    def to_kimodo(self, p):
        d = np.asarray(p) - self.p0
        return np.stack([self.s * (d @ self.left), self.root_y + self.s * d[..., 2], self.s * (d @ self.fwd)],
                        axis=-1)

    def from_kimodo(self, k):
        k = np.asarray(k)
        return (self.p0 + (k[..., :1] * self.left + k[..., 2:3] * self.fwd) / self.s
                + (k[..., 1:2] - self.root_y) / self.s * np.array([0.0, 0.0, 1.0]))


def _hand_rotation(rest_bone, d):
    """Global rotation taking the rest hand bone direction (T-pose: +-x, palm down) to d, keeping the palm down."""
    y = np.array([0.0, 1.0, 0.0])
    u = y - (y @ d) * d
    u /= np.linalg.norm(u)
    src = np.stack([rest_bone, y, np.cross(rest_bone, y)], axis=1)
    dst = np.stack([d, u, np.cross(d, u)], axis=1)
    return dst @ src.T


def generate(args):
    import torch
    from kimodo import load_model
    from kimodo.constraints import LeftHandConstraintSet, RightHandConstraintSet, Root2DConstraintSet

    lay_all = json.loads((OUT / "layouts.json").read_text())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.text:
        text_encoder = _CachedText(torch.load(OUT / "text_features.pt"))
    else:
        text_encoder = _ZeroText()
    model = load_model(args.model, device=device, text_encoder=text_encoder)
    sk, fps = model.skeleton, model.fps
    s77 = sk.somaskel77
    bi, bi77 = sk.bone_index, s77.bone_index
    neutral = sk.neutral_joints.cpu().numpy()
    root_y = args.root_height
    base = neutral + np.array([0.0, root_y, 0.0])
    kimodo_arm = float(np.linalg.norm(neutral[bi["LeftArm"]] - neutral[bi["LeftForeArm"]])
                       + np.linalg.norm(neutral[bi["LeftForeArm"]] - neutral[bi["LeftHand"]]))
    hand_len = float(np.linalg.norm(neutral[bi["LeftHandMiddleEnd"]] - neutral[bi["LeftHand"]]))
    n_frames = int(round(args.duration * fps))
    rng = np.random.default_rng(args.seed)

    def hand_constraint(side, frames, targets, hips=None):
        """Hand keyframes (wrist targets in the Kimodo frame); hips: (n, 3) hip positions of the keyframes (the
        constraint also fixes them), default standing at the origin."""
        hand, end = ("LeftHand", "LeftHandMiddleEnd") if side == "left" else ("RightHand", "RightHandMiddleEnd")
        rest_bone = np.array([1.0, 0.0, 0.0]) if side == "left" else np.array([-1.0, 0.0, 0.0])
        hips = np.repeat([[0.0, root_y, 0.0]], len(frames), axis=0) if hips is None else np.asarray(hips)
        pos = (neutral[None] + hips[:, None]).copy()
        rot = np.repeat(np.eye(3)[None, None], len(frames), axis=0).repeat(len(neutral), axis=1)
        for i, w in enumerate(targets):
            shoulder = pos[i, bi["LeftArm" if side == "left" else "RightArm"]]
            d = (w - shoulder) / np.linalg.norm(w - shoulder) + np.array([0.0, -0.3, 0.0])
            d /= np.linalg.norm(d)
            pos[i, bi[hand]] = w
            pos[i, bi[end]] = w + hand_len * d
            rot[i, bi[hand]] = _hand_rotation(rest_bone, d)
        cls = LeftHandConstraintSet if side == "left" else RightHandConstraintSet
        f = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
        # CPU tensors (Kimodo creates its index tensors there), then moved to the device
        return cls(sk, torch.tensor(frames), f(pos), f(rot), f(np.zeros((len(frames), 2)))).to(device)

    clips, meta, prompts = [], [], []
    velocity = lay_all["velocity"]
    random_prompt = lambda task: PROMPTS[task][rng.integers(len(PROMPTS[task]))].format(speed=SPEED[velocity])
    for subject, lay in lay_all["subjects"].items():
        frame = CellFrame(lay, kimodo_arm / lay["arm"], root_y)
        home = {h: frame.to_kimodo(lay["goals"][f"home_{h}"]) for h in ("right", "left")}
        for rep in range(args.per_task):
            for task, hands in TASKS:
                t_leave = rng.uniform(1.0, 1.4)
                t_arr = t_leave + rng.uniform(0.7, 1.2)
                t_back = t_arr + rng.uniform(0.6, 1.0)
                t_home = t_back + rng.uniform(0.7, 1.2)
                times = {"home": [0.0, 0.5 * t_leave, t_leave], "goal": [t_arr, 0.5 * (t_arr + t_back), t_back],
                         "back": [t_home, min(t_home + 0.4, args.duration - 0.1),
                                  min(t_home + 0.8, args.duration - 0.05)]}
                cons = []
                if args.constraints == "tight":
                    for h in ("right", "left"):
                        goal_name = (f"robot_{h}" if task == "robot" else task) if h in hands else f"home_{h}"
                        goal = frame.to_kimodo(lay["goals"][goal_name])
                        frames, targets = [], []
                        for phase in ("home", "goal", "back"):
                            for t in times[phase]:
                                frames.append(min(int(round(t * fps)), n_frames - 1))
                                targets.append(goal if phase == "goal" else home[h])
                        cons.append(hand_constraint(h, frames, targets))
                    rf = list(range(0, n_frames, 15))
                    cons.append(Root2DConstraintSet(sk, torch.tensor(rf), torch.zeros(len(rf), 2),
                                                    global_root_heading=torch.tensor([[1.0, 0.0]] * len(rf)))
                                .to(device))
                else:   # loose: moving hand at leave / arrive / leave goal / back home, other hand at start and end
                    offsets = lay["pelvis_offsets"]
                    lean = np.array(offsets.get(task, offsets.get(f"{task}_right", [0.0, 0.0, 0.0])))
                    hips_goal = frame.to_kimodo(np.array(lay["pelvis"]) + lean)
                    standing = np.array([0.0, root_y, 0.0])
                    f_ = lambda t: min(int(round(t * fps)), n_frames - 1)
                    for h in ("right", "left"):
                        if h in hands:
                            goal = frame.to_kimodo(lay["goals"][f"robot_{h}" if task == "robot" else task])
                            frames = [0, f_(t_leave), f_(t_arr), f_(t_back), f_(t_home), n_frames - 1]
                            targets = [home[h], home[h], goal, goal, home[h], home[h]]
                            hips = [standing, standing, hips_goal, hips_goal, standing, standing]
                        else:
                            frames, targets, hips = [0, n_frames - 1], [home[h], home[h]], [standing, standing]
                        cons.append(hand_constraint(h, frames, targets, hips))
                prompt = (random_prompt(task) if args.text else "")
                clips.append(cons)
                prompts.append(prompt)
                meta.append({"subject": subject, "task": task, "hands": list(hands), "rep": rep, "scale": frame.s,
                             "prompt": prompt, "constraints": args.constraints,
                             "t_leave": t_leave, "t_arrive": t_arr, "t_back": t_back, "t_home": t_home})
    print(f"{len(clips)} clips of {args.duration:g} s ({n_frames} frames at {fps} Hz), batches of {args.batch}")

    kp_all, head_all, gaze_all, hit_err = [], [], [], []
    torch.manual_seed(args.seed)
    for b0 in range(0, len(clips), args.batch):
        batch = clips[b0: b0 + args.batch]
        out = model(prompts[b0: b0 + args.batch], n_frames, num_denoising_steps=args.steps, constraint_lst=batch,
                    post_processing=args.postprocess, return_numpy=True, progress_bar=lambda x: x)
        pj, gr = out["posed_joints"], out["global_rot_mats"]
        for i, m in enumerate(meta[b0: b0 + len(batch)]):
            lay = lay_all["subjects"][m["subject"]]
            frame = CellFrame(lay, m["scale"], root_y)
            j = pj[i]
            R_head = gr[i, :, bi77["Head"]]
            eyes = 0.5 * (j[:, bi77["LeftEye"]] + j[:, bi77["RightEye"]])
            nose = eyes + np.einsum("tij,j->ti", R_head, [0.0, -0.035, 0.025])
            ears = j[:, bi77["Head"]] + np.einsum("tij,j->ti", R_head, [0.0, 0.04, 0.0])
            kp = np.stack([nose if k == "head" else j[:, bi77[SOMA_JOINT[k]]] for k in KEYPOINTS], axis=1)
            kp_all.append(frame.from_kimodo(kp))
            head_all.append(frame.from_kimodo(nose))
            g = frame.from_kimodo(nose) - frame.from_kimodo(ears)
            gaze_all.append(g / np.linalg.norm(g, axis=1, keepdims=True))
            for c in batch[i][:2]:   # wrist keyframe errors (Kimodo scale -> m in the cell)
                side = "LeftHand" if c.name == "left-hand" else "RightHand"
                fr = c.frame_indices.cpu().numpy()
                tgt = c.global_joints_positions[:, bi[side]].cpu().numpy()
                hit_err.append(np.linalg.norm(j[fr, bi77[side]] - tgt, axis=1) / m["scale"])
        print(f"  {min(b0 + args.batch, len(clips))}/{len(clips)}", flush=True)
    hit = np.concatenate(hit_err)
    np.savez(OUT / args.out, kp=np.array(kp_all), head=np.array(head_all), gaze=np.array(gaze_all), fps=fps,
             meta=json.dumps(meta), model=args.model, postprocess=args.postprocess, steps=args.steps)
    print(f"wrist keyframe error: median {100 * np.median(hit):.1f} cm, 95 % {100 * np.percentile(hit, 95):.1f} cm"
          f"\n-> {OUT / args.out}")


# =============================================================================
# compare (nioc environment)
# =============================================================================
def compare(args):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "evaluation"))
    import pickle
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import goal_inference as gi

    runs = [Path(r) if Path(r).is_absolute() else ROOT / r for r in args.runs]
    def label(run):
        a = json.loads((run / "results.json").read_text())["args"]
        d = np.load(ROOT / a["clips"], allow_pickle=True)
        m = json.loads(str(d["meta"]))[0]
        text = "text" if m.get("prompt") else "no text"
        return f"Kimodo, {text}, {m.get('constraints', 'tight')} constraints" + (", real noise" if a.get("noise") else "")
    labels = ["CARI v2 (real)"] + [label(r) for r in runs[1:]]
    rows, curves, stats = [], {}, {}
    for label, run in zip(labels, runs):
        res = json.loads((run / "results.json").read_text())
        durations, sessions = pickle.load(open(run / "sessions.pkl", "rb"))
        settings = res.get("applied_settings", res["loso_settings"])
        LLs = [gi.lagged_loglik(s) for s in sessions]
        cfgs = {s["subject"]: settings[s["subject"]] for s in sessions}
        agg = gi.aggregate(sessions, LLs, cfgs, res["args"]["rate"])
        t = agg["table"]["moving"]
        rows.append((label, len(sessions), sum(len(s["movements"]) for s in sessions), agg["accuracy_moving"],
                     agg["accuracy_rest"], agg["balanced_accuracy"], agg["decision_median_pct"],
                     t["inferred"][0], t["oracle"][0], t["frozen"][0], t["constant velocity"][0],
                     t["inferred"][1], t["oracle"][1], t["frozen"][1]))
        curves[label] = agg["accuracy_bins"]
        stats[label] = movement_stats(sessions)
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    x = np.arange(5, 100, 10)
    for label in labels:
        ax[0].plot(x, curves[label], "o-", label=label)
        ax[1].hist(stats[label]["duration"], bins=np.linspace(0, 2.5, 26), alpha=0.5, density=True, label=label)
        ax[2].plot(np.linspace(0, 1, 51), stats[label]["speed_profile"], label=label)
    ax[0].set(xlabel="movement progress (%)", ylabel="MAP goal correct (%)", ylim=(0, 102))
    ax[1].set(xlabel="movement duration (s, 12 % of peak speed)", ylabel="density")
    ax[2].set(xlabel="normalized time", ylabel="wrist speed / peak (mean)")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=150)
    plt.close(fig)

    head = ["data", "sessions", "movements", "goal acc. moving (%)", "rest (%)", "balanced (%)",
            "decision (% of mvt)", "MPJPE inferred", "MPJPE oracle", "MPJPE frozen", "MPJPE CV",
            "wrists inferred", "wrists oracle", "wrists frozen"]
    srow = [("duration (s)", "duration"), ("amplitude (cm)", "amplitude"), ("peak speed (m/s)", "peak"),
            ("time of peak speed (% of mvt)", "peak_time"), ("path / chord", "straightness"),
            ("chest displacement (cm)", "chest"), ("passive wrist displacement (cm)", "passive")]
    t1 = "".join(f"<th>{h}</th>" for h in head)
    b1 = "".join("<tr>" + "".join(f"<td>{v:.1f}</td>" if isinstance(v, float) else f"<td>{v}</td>" for v in r)
                 + "</tr>" for r in rows)
    t2 = "<th>reach statistic (median [IQR])</th>" + "".join(f"<th>{l}</th>" for l in labels)
    b2 = "".join(f"<tr><td>{name}</td>" + "".join(
        f"<td>{np.median(stats[l][k]):.2f} [{np.percentile(stats[l][k], 25):.2f}, "
        f"{np.percentile(stats[l][k], 75):.2f}]</td>" for l in labels) + "</tr>" for name, k in srow)
    style = ("<style>body{font-family:sans-serif;max-width:1200px;margin:24px auto;padding:0 16px}table{border-"
             "collapse:collapse;margin:8px 0 24px}td,th{border:1px solid #ccc;padding:4px 8px;text-align:right}"
             "td:first-child{text-align:left}img{max-width:100%}</style>")
    (OUT / "comparison.html").write_text(
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Kimodo vs CARI</title>{style}</head><body>"
        f"<h1>Are Kimodo reaches representative of CARI v2?</h1><p>Same online pipeline and filter settings (selected "
        f"on the real data); prediction errors in cm on movement ticks, 1 s window.</p><table><tr>{t1}</tr>{b1}"
        f"</table><table><tr>{t2}</tr>{b2}</table><img src='comparison.png'></body></html>")
    print("\n" + " | ".join(head))
    for r in rows:
        print(" | ".join(f"{v:.1f}" if isinstance(v, float) else str(v) for v in r))
    for name, k in srow:
        print(f"{name:34s}" + "".join(f"{l[:20]:>22s}: {np.median(stats[l][k]):6.2f}" for l in labels))
    print(f"-> {OUT / 'comparison.html'}")


def movement_stats(sessions):
    """Kinematics of the movements of the sessions (moving wrist of the hand that moves most): duration, amplitude,
    peak speed and its time, path length / chord, chest and passive-wrist displacement, mean normalized speed
    profile."""
    import cari_sessions as cs
    out = {k: [] for k in ("duration", "amplitude", "peak", "peak_time", "straightness", "chest", "passive")}
    profiles = []
    cache = {}
    for s in sessions:
        if "kp" not in s:   # runs saved before the keypoints were stored: the CARI session again
            if s["subject"] not in cache:
                S = cs.load_session(s["subject"], "FAST", csv=None)
                cache[s["subject"]] = (S.kp, S.q28_filt[:, 0:3])
            s["kp"], s["chest"] = cache[s["subject"]]
        kp, chest, dt = s["kp"], s["chest"], s["dt"]
        for seg, on, off, goals, hands in s["movements"]:
            w = kp[on: off + 1, cs.hkm.KP_INDEX[f"{hands[0]}_wrist"]]
            if len(w) < 5:
                continue
            v = np.linalg.norm(np.gradient(w, dt, axis=0), axis=1)
            other = "left" if hands[0] == "right" else "right"
            p = kp[on: off + 1, cs.hkm.KP_INDEX[f"{other}_wrist"]]
            chord = np.linalg.norm(w[-1] - w[0])
            out["duration"].append((off - on) * dt)
            out["amplitude"].append(100 * chord)
            out["peak"].append(float(np.percentile(v, 98)))
            out["peak_time"].append(100 * np.argmax(v) / max(len(v) - 1, 1))
            out["straightness"].append(np.sum(np.linalg.norm(np.diff(w, axis=0), axis=1)) / max(chord, 1e-3))
            out["chest"].append(100 * np.linalg.norm(chest[off] - chest[on]))
            out["passive"].append(100 * (np.linalg.norm(p[-1] - p[0]) if len(hands) == 1 else np.nan))
            profiles.append(np.interp(np.linspace(0, 1, 51), np.linspace(0, 1, len(v)), v / max(v.max(), 1e-6)))
    out = {k: np.array([x for x in v if np.isfinite(x)]) for k, v in out.items()}
    out["speed_profile"] = np.mean(profiles, axis=0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("layout")
    a.add_argument("--subjects", nargs="+", default=None)
    a.add_argument("--velocity", default="FAST")
    g = sub.add_parser("generate")
    g.add_argument("--model", default="Kimodo-SOMA-RP-v1.1")
    g.add_argument("--per-task", type=int, default=5, help="clips per subject and task (4 tasks)")
    g.add_argument("--duration", type=float, default=6.0)
    g.add_argument("--steps", type=int, default=50, help="denoising steps")
    g.add_argument("--batch", type=int, default=20)
    g.add_argument("--root-height", type=float, default=0.95, help="hip height of the keyframes (m, Kimodo scale)")
    g.add_argument("--postprocess", action="store_true",
                   help="Kimodo post-processing (off: it enforces the whole keyframe pose, of which only the hand is "
                        "meaningful here, and pulls the hand back towards the T-pose; without it the wrists hit the "
                        "keyframes to ~1 cm)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--text", action="store_true", help="prompt per clip (PROMPTS), from the features of `encode`")
    g.add_argument("--constraints", choices=["tight", "loose"], default="tight")
    g.add_argument("--out", default="clips.npz", help="output file in output/kimodo")
    e = sub.add_parser("encode")
    e.add_argument("--device", default="cpu", help="device of the text encoder (bf16: ~16 GB)")
    c = sub.add_parser("compare")
    c.add_argument("runs", nargs="+", help="goal_inference.py output folders: the real run first")
    args = ap.parse_args()
    {"layout": layout, "generate": generate, "encode": encode, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    main()
