#!/usr/bin/env python3
"""Online prediction replayed on whole CARI v2 sessions, with the goal inferred or known (config `online`).

Each session (home -> object 1 -> home -> object 2 -> home -> object 3 -> home -> robot -> home, ~30 s,
cari_sessions.py) is replayed as the ROS 2 node sees it: every 1/rate s, the last observation_time s of IK angles
(raw, as online) are resampled and every hypothesis of the cell (goal location x hand, plus idle) is predicted over
the prediction horizon with the IOC-fitted weights of a train.py run (prophet_ioc.human_prediction.predict_hypotheses:
receding horizon, temporary target when the goal is farther than the horizon). The goal filter (GoalFilter: evidence
of the lagged predictions, heading and gaze cue prior) gives the posterior over the hypotheses, and the published
prediction is the most probable one.

online.goal_mode selects the posterior the published prediction (and the node) uses:
- inferred: the filter runs over all the goals of the cell;
- known: the goal of the current instruction is given (task schedule): the filter only runs over that goal (its
  hands) and idle, i.e. it detects the hand and the onset.
Both are always evaluated, with
- goal and onset known (oracle): the true goal during each movement, idle outside them;
- constant velocity, frozen.
online.uncertainty selects the covariance of the published prediction:
- map: the covariance of the most probable hypothesis only;
- mixture: also the goal uncertainty, sum_g p_g (Sigma_g + (mu_g - mu) (mu_g - mu)^T) around the published mean mu
  (the law of total covariance over the hypotheses); both coverages are reported.

The filter settings are selected on the training subjects' sessions (data.subjects - data.test_subjects, the same
split as train.py) and the results are reported on the held-out test subjects and, for reference, on all subjects
(the training subjects' demonstrations were used by the IOC fit).

    python evaluation/goal_inference.py                                 # -> output/goal_inference_<ts>/
    python evaluation/goal_inference.py online.goal_mode=known online.uncertainty=map
    python evaluation/goal_inference.py online.reuse=output/latest_goal_inference   # tables / figures only
"""

import csv
import itertools
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")

import hydra
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import cari_kinematic as ck
import cari_sessions as cs
from prophet_ioc import human_prediction as hp
from prophet_ioc.infer import predict_constant_velocity

DEFAULT_FILTER = {"temperature": 0.25, "evidence_lag": 0.3, "switch_rate": 0.5, "kappa_heading": 2.0,
                  "kappa_gaze": 1.0}
GRID = {"temperature": [0.5, 1.0, 2.0, 4.0], "evidence_lag": [0.1, 0.2, 0.3], "switch_rate": [0.5, 1.0, 2.0],
        "kappa_heading": [0.0, 4.0, 8.0, 16.0], "kappa_gaze": [0.0, 4.0, 8.0, 16.0]}
AT_GOAL = 0.12   # m: at rest, a goal whose wrist is this close counts as correct (holding the hand there)
CUES = {"none": (0.0, 0.0), "heading": (None, 0.0), "gaze": (0.0, None), "heading + gaze": (None, None)}
METHODS = ["goal inferred", "goal known", "oracle", "constant velocity", "frozen"]
METHOD_NOTES = {"goal inferred": "Bayesian filter over the goals of the cell (+ idle)",
                "goal known": "goal of the current instruction given; the filter finds hand and onset",
                "oracle": "goal and onset known", "constant velocity": "every joint", "frozen": "last pose held"}
METRICS = [("mpjpe_cm", "MPJPE (cm)"), ("wrists_ade_cm", "Wrists ADE (cm)"), ("wrists_fde_cm", "Wrists FDE (cm)"),
           ("coverage_map_pct", "Wrists in 95 % region, hypothesis covariance (%)"),
           ("coverage_mixture_pct", "Wrists in 95 % region, goal-aware covariance (%)")]
OBS_NOISE = 0.01
V_REF = 0.3
WRIST = {"right": hp.JOINTS.index("right_wrist"), "left": hp.JOINTS.index("left_wrist")}
FIT, INIT, GT_C = "#2a78d6", "#eb6834", "#0b0b0b"
GRAYS = ["#5f5e5a", "#8a8984", "#a9a8a2", "#c4c3bd"]
TEXT2, GRID_C, SURFACE = "#52514e", "#e6e5e0", "#fcfcfb"
GOAL_COLORS = {"object_1": "#2a78d6", "object_2": "#eb6834", "object_3": "#1baf7a", "home_right": "#eda100",
               "home_left": "#e87ba4", "robot_right": "#008300", "robot_left": "#4a3aa7", "idle": "#a9a8a2"}


# =============================================================================
# Model parameters
# =============================================================================
def model_parameters(cfg):
    """(params, pred_noise, source): IOC-fitted weights and prediction noise of a train.py run (online.params; the
    median of its noise levels over the observed fractions), else the initial weights of config/model."""
    if not cfg.online.params:
        return ck.model_params(cfg.model), float(cfg.model.pred_noise), "initial weights (config/model)"
    path = ROOT / cfg.online.params
    path = path / "params.json" if path.is_dir() else path
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run train.py first, or use online.params=null")
    data = json.loads(path.read_text())
    noise = data.get("pred_noise", cfg.model.pred_noise)
    sigma = float(np.median(list(noise.values()))) if isinstance(noise, dict) else float(noise)
    return ck.load_params(path), sigma, f"IOC-fitted ({path.relative_to(ROOT)})"


# =============================================================================
# Phase 1: predictions of every hypothesis at every tick
# =============================================================================
def nominal_durations(subjects, velocity):
    """Nominal minimum-jerk duration per subject, leave-one-subject-out: median over the other subjects' movements
    of 1.875 A / v_peak (amplitude and peak speed of the wrist that moves most; a minimum-jerk motion of duration D
    peaks at 1.875 A / D). The onset-offset durations are longer (slow tails of the speed profile)."""
    from scipy.signal import medfilt
    dur = {}
    for s in subjects:
        S = cs.load_session(s, velocity, csv=None)
        dur[s] = []
        for m in S.movements:
            w = S.kp[:, cs.hkm.KP_INDEX[f"{m.hands[0]}_wrist"]]
            speed = medfilt(np.linalg.norm(np.gradient(w, S.dt, axis=0), axis=1), 11)[m.onset: m.offset + 1]
            dur[s].append(1.875 * np.linalg.norm(w[m.offset] - w[m.onset]) / np.percentile(speed, 98))
    return {s: float(np.median([d for o in subjects if o != s for d in dur[o]] or dur[s])) for s in subjects}


def replay_session(S, ocfg, params, H, max_iter, pred_noise, nominal_duration):
    """Per tick of the session S (cari_sessions.CariSession): the predictions of every hypothesis over the horizon
    (joints, wrist covariances), the ground truth, the baselines' errors, the cue terms and what the goal filter
    needs."""
    subject = S.subject
    goals = cs.layout_goals(subject)
    hyps = cs.layout_hypotheses(goals)
    names = [h.name for h in hyps]
    times = np.arange(0.0, ocfg.horizon + 1e-9, ocfg.prediction_dt)
    lag_steps = np.arange(0, int(np.ceil(max(GRID["evidence_lag"]) * ocfg.rate)) + 2)
    taus = lag_steps / ocfg.rate
    n_win = int(round(ocfg.observation_time / S.dt))
    dt_s = ocfg.observation_time / (ocfg.samples - 1)
    q_src = S.q28_raw if ocfg.angles == "raw" else S.q28_filt
    t_ticks = np.arange(ocfg.observation_time + 0.05, S.time[-1] - ocfg.horizon, 1.0 / ocfg.rate)
    gt_idx = lambda f: np.clip(f + np.round(times / S.dt).astype(int), 0, len(S.kp) - 1)
    pelvis = lambda kp: 0.5 * (kp[..., cs.hkm.KP_INDEX["left_hip"], :] + kp[..., cs.hkm.KP_INDEX["right_hip"], :])
    joint_of = lambda kp, chest, j: chest if j == "chest" else pelvis(kp) if j == "pelvis" else \
        kp[..., cs.hkm.KP_INDEX[j], :]

    K, T, n_t = len(hyps), len(t_ticks), len(times)
    out = {"subject": subject, "name": getattr(S, "name", subject), "names": names, "dt": S.dt,
           "hands": [h.hand for h in hyps], "kp": S.kp.astype(np.float32), "chest": S.q28_filt[:, 0:3],
           "goals": np.array([h.goal if h.goal is not None else (np.nan,) * 3 for h in hyps]), "t": t_ticks,
           "frame": np.round(t_ticks / S.dt).astype(int), "times": times,
           "pj": np.zeros((T, K, n_t, len(hp.JOINTS), 3), np.float32), "pc": np.zeros((T, K, n_t, 2, 3, 3), np.float32),
           "gt": np.zeros((T, n_t, len(hp.JOINTS), 3), np.float32), "base_err": np.zeros((T, 2, 3)),
           "heading": np.zeros((T, K)), "gaze": np.zeros((T, K)), "wobs": np.zeros((T, 2, 3)),
           "pmean": np.zeros((T, K, len(taus), 2, 3)), "pcov": np.zeros((T, K, len(taus), 2, 3, 3)),
           "latency": np.zeros(T), "taus": taus,
           "movements": [(m.segment, m.onset, m.offset, m.goals, m.hands) for m in S.movements]}
    for k, (t, f) in enumerate(zip(tqdm(t_ticks, desc=subject, leave=False), out["frame"])):
        stamps = S.time[f - n_win: f + 1]
        hist = hp.resample_history(stamps, q_src[f - n_win: f + 1], stamps[-1], ocfg.observation_time, ocfg.samples)
        t0 = time.perf_counter()
        preds, hs = hp.predict_hypotheses(hist, dt_s, S.body_params, hyps, params, H, max_iter,
                                          horizon=ocfg.horizon, nominal_duration=nominal_duration)
        out["latency"][k] = time.perf_counter() - t0
        out["heading"][k] = hp.goal_cue_logprior(hyps, hs, None, None, 1.0, 0.0, V_REF)
        if S.gaze is not None:
            out["gaze"][k] = hp.goal_cue_logprior(hyps, hs, S.head[f], S.gaze[f], 0.0, 1.0, V_REF)
        out["wobs"][k] = [preds[0].prediction.joints[f"{s}_wrist"][0] for s in ("right", "left")]
        gi = gt_idx(f)
        gt = {j: joint_of(S.kp[gi], S.q28_filt[gi, 0:3], j) for j in hp.JOINTS}
        out["gt"][k] = np.stack([gt[j] for j in hp.JOINTS], axis=1)
        for i, p in enumerate(preds):
            hand = p.hypothesis.hand
            other = "left" if hand == "right" else "right"
            joints, cov = hp.sample_prediction(p.prediction, times, pred_noise)
            out["pj"][k, i] = np.stack([joints[j] for j in hp.JOINTS], axis=1)
            for s, c in ((hand, "wrist"), (other, "passive_wrist")):
                out["pc"][k, i, :, 0 if s == "right" else 1] = cov[c]
            jl, cl = hp.sample_prediction(p.prediction, taus, pred_noise)
            for s, c in ((hand, "wrist"), (other, "passive_wrist")):
                out["pmean"][k, i, :, 0 if s == "right" else 1] = jl[f"{s}_wrist"]
                out["pcov"][k, i, :, 0 if s == "right" else 1] = cl[c]
        kp_hist = np.array(hp._fk_batch(jax.numpy.asarray(hist), jax.numpy.asarray(S.body_params)))
        obs = {j: joint_of(kp_hist, hist[:, 0:3], j) for j in hp.JOINTS}
        out["base_err"][k, 0] = _errors({j: predict_constant_velocity(obs[j], times, dt_s) for j in hp.JOINTS}, gt)
        out["base_err"][k, 1] = _errors({j: np.repeat(obs[j][-1:], len(times), axis=0) for j in hp.JOINTS}, gt)
    return out


def _errors(joints, gt):
    """MPJPE, mean wrist ADE and FDE (cm) over the horizon samples."""
    d = {j: np.linalg.norm(joints[j] - gt[j], axis=1) for j in hp.JOINTS}
    wr = np.mean([d["right_wrist"], d["left_wrist"]], axis=0)
    return 100.0 * np.array([np.mean([d[j] for j in hp.JOINTS]), wr.mean(), wr[-1]])


# =============================================================================
# Phase 2: goal filter replays and metrics
# =============================================================================
def lagged_loglik(sess):
    """LL[k, l, i]: log density of the observed wrists at tick k under the prediction of hypothesis i made l ticks
    earlier (as GoalFilter.loglikelihood), for every lag l of the grid."""
    T, K, L = sess["pmean"].shape[:3]
    LL = np.full((T, L, K), np.nan)
    S = sess["pcov"] + OBS_NOISE ** 2 * np.eye(3)
    for l in range(1, L):
        r = sess["wobs"][l:, None, :, :] - sess["pmean"][:-l, :, l]                     # (T-l, K, 2, 3)
        LL[l:, l] = hp.wrist_logdensity(r, S[:-l, :, l]).sum(axis=-1)
    return LL


def filter_posteriors(sess, LL, temperature, evidence_lag, switch_rate, kappa_heading, kappa_gaze, rate, mask=None):
    """Posterior at every tick of the GoalFilter recursion (uniform ticks: evidence from the prediction made
    ceil(lag * rate) ticks earlier, or the oldest one at the start). mask (T, K): the hypotheses allowed at each
    tick (known goal: the instruction's goal and idle), the others have zero probability."""
    T, _, K = LL.shape
    allowed = np.ones((T, K), bool) if mask is None else mask
    lag = int(np.ceil(evidence_lag * rate - 1e-6))
    eps = 1.0 - np.exp(-switch_rate / rate)
    p = allowed[0] / allowed[0].sum()                      # belief of the filter (without the cue prior)
    post = np.zeros((T, K))
    for k in range(T):
        a = allowed[k]
        if k > 0:
            p = np.where(a, p, 0.0)
            p = p / p.sum() if p.sum() > 0 else a / a.sum()   # (newly allowed hypotheses enter by switching)
            p = np.where(a, (1.0 - eps) * p + eps / a.sum(), 0.0)
            ll = LL[k, min(lag, k)]
            logp = np.where(a, np.log(np.where(a, p, 1.0)) + temperature * (ll - np.max(ll[a])), -np.inf)
            p = np.exp(logp - np.max(logp))
            p /= p.sum()
        with np.errstate(divide="ignore"):
            lp = np.where(a, np.log(p) + kappa_heading * sess["heading"][k] + kappa_gaze * sess["gaze"][k], -np.inf)
        e = np.exp(lp - np.max(lp))
        post[k] = e / e.sum()
    return post


def tick_labels(sess):
    """Per tick: index of the movement it belongs to (-1 idle), its normalized time, the oracle hypothesis (true goal
    with the hand that moves, idle outside the movements) and the mask of the known-goal mode (the goals of the current
    instruction - the next movement's between movements - with every hand that may reach them, and idle)."""
    names, frames = sess["names"], sess["frame"]
    idle = names.index("idle")
    T, K = len(frames), len(names)
    mov = np.full(T, -1)
    phase = np.full(T, np.nan)
    oracle = np.full(T, idle)
    known = np.zeros((T, K), bool)
    known[:, idle] = True
    moves = sess["movements"]
    for m, (seg, on, off, goals, hands) in enumerate(moves):
        inside = (frames >= on) & (frames <= off)
        mov[inside] = m
        phase[inside] = (frames[inside] - on) / max(off - on, 1)
        main = goals[0]   # the goal of the hand that moves most, with that hand if it is a hypothesis
        oracle[inside] = next((i for i, (n, h) in enumerate(zip(names, sess["hands"])) if n == main and h == hands[0]),
                              names.index(main))
        current = (frames <= off) & ((frames > moves[m - 1][2]) if m > 0 else True)
        for i, n in enumerate(names):
            if n in cs.SEGMENT_GOALS.get(seg, goals):
                known[current, i] = True
    return mov, phase, oracle, known


def goal_aware_cov(pj, pc, post, i):
    """Covariance of the right / left wrist (n_t, 2, 3, 3) around hypothesis i's prediction including the goal
    uncertainty: sum_g p_g (Sigma_g + (mu_g - mu_i) (mu_g - mu_i)^T)."""
    mu = pj[:, :, [WRIST["right"], WRIST["left"]]]                      # (K, n_t, 2, 3)
    d = mu - mu[i][None]
    return np.einsum("g,gtsij->tsij", post, pc + d[..., :, None] * d[..., None, :])


def _coverage(pred_wrists, gt_wrists, cov):
    """% of the wrist samples (both wrists) inside their 95 % ellipsoids."""
    err = (gt_wrists - pred_wrists).reshape(-1, 3)
    return 100.0 * float(np.mean(hp.coverage_fraction(err, cov.reshape(-1, 3, 3))))


def tick_metrics(sess, k, i, post_k):
    """[MPJPE, wrists ADE, wrists FDE, coverage (hypothesis cov), coverage (goal-aware cov)] of hypothesis i at tick
    k; post_k: the posterior used for the goal-aware covariance (None: same as the hypothesis covariance)."""
    pj, pc, gt = sess["pj"][k], sess["pc"][k], sess["gt"][k]
    d = np.linalg.norm(pj[i] - gt, axis=-1) * 100.0                     # (n_t, joints)
    wr = d[:, [WRIST["right"], WRIST["left"]]].mean(axis=1)
    w_idx = [WRIST["right"], WRIST["left"]]
    cov_map = _coverage(pj[i][:, w_idx], gt[:, w_idx], pc[i])
    cov_mix = cov_map if post_k is None else _coverage(pj[i][:, w_idx], gt[:, w_idx],
                                                       goal_aware_cov(pj, pc, post_k, i))
    return np.array([d.mean(), wr.mean(), wr[-1], cov_map, cov_mix])


def evaluate(sess, post_inferred, post_known):
    """Per-tick metrics of every method {method: (T, 5)}, and the goal-inference statistics of the inferred mode."""
    mov, phase, oracle, known = tick_labels(sess)
    T = len(mov)
    MAP_inf, MAP_known = post_inferred.argmax(axis=1), post_known.argmax(axis=1)
    res = {"goal inferred": np.array([tick_metrics(sess, k, MAP_inf[k], post_inferred[k]) for k in range(T)]),
           "goal known": np.array([tick_metrics(sess, k, MAP_known[k], post_known[k]) for k in range(T)]),
           "oracle": np.array([tick_metrics(sess, k, oracle[k], None) for k in range(T)])}
    nan2 = np.full((T, 2), np.nan)
    res["constant velocity"] = np.column_stack([sess["base_err"][:, 0], nan2])
    res["frozen"] = np.column_stack([sess["base_err"][:, 1], nan2])
    stats = goal_statistics(sess, post_inferred, mov, phase)
    return res, mov, stats


def goal_statistics(sess, post, mov, phase):
    names = sess["names"]
    T = len(mov)
    ar = np.arange(T)
    MAP = post.argmax(axis=1)
    correct = np.array([mov[k] >= 0 and names[MAP[k]] in sess["movements"][mov[k]][3] for k in range(T)])
    side = np.array([0 if h == "right" else 1 for h in sess["hands"]])
    near = np.linalg.norm(sess["wobs"][ar, side[MAP]] - sess["goals"][MAP], axis=1) < AT_GOAL
    rest_ok = (mov < 0) & ((MAP == names.index("idle")) | near)
    p_true = np.array([sum(post[k, i] for i, n in enumerate(names) if n in sess["movements"][mov[k]][3])
                       if mov[k] >= 0 else np.nan for k in range(T)])
    decision = []
    for m in range(len(sess["movements"])):
        idx = np.where(mov == m)[0]
        if len(idx) == 0:
            continue
        wrong = np.where(~correct[idx])[0]
        first_stable = 0 if len(wrong) == 0 else wrong[-1] + 1
        decision.append(phase[idx[first_stable]] if first_stable < len(idx) else 1.0)
    return {"correct": correct, "rest_ok": rest_ok, "p_true": p_true, "decision": np.array(decision),
            "phase": phase}


def posteriors(sess, LL, cfg, rate):
    """(inferred, known) posteriors of a session with the filter settings cfg."""
    _, _, _, known = tick_labels(sess)
    return (filter_posteriors(sess, LL, rate=rate, **cfg),
            filter_posteriors(sess, LL, rate=rate, mask=known, **cfg))


def aggregate(sessions, LLs, cfgs, rate):
    """Metrics pooled over the sessions; cfgs: {subject: filter settings}."""
    rows = {m: [] for m in METHODS}
    acc_bins, decisions, moving, rest = [], [], [], []
    for sess, LL in zip(sessions, LLs):
        post_inf, post_known = posteriors(sess, LL, cfgs[sess["subject"]], rate)
        res, mov, st = evaluate(sess, post_inf, post_known)
        rest.append((st["rest_ok"].sum(), (mov < 0).sum()))
        for m in METHODS:
            rows[m].append(res[m])
        moving.append(mov >= 0)
        b = np.clip(np.floor(np.nan_to_num(st["phase"], nan=-1.0) * 10), -1, 9)
        for i in range(10):
            sel = (mov >= 0) & (b == i)
            acc_bins.append((i, st["correct"][sel].sum(), sel.sum(), np.nansum(st["p_true"][sel])))
        decisions.append(st["decision"])
    moving = np.concatenate(moving)
    table = {}
    for split, sel in (("all", np.ones_like(moving)), ("moving", moving), ("idle", ~moving)):
        table[split] = {m: np.nanmean(np.concatenate(rows[m])[sel], axis=0) for m in METHODS}
    acc, ptr = np.zeros(10), np.zeros(10)
    for i in range(10):
        c = sum(a[1] for a in acc_bins if a[0] == i)
        n = sum(a[2] for a in acc_bins if a[0] == i)
        acc[i], ptr[i] = 100.0 * c / max(n, 1), sum(a[3] for a in acc_bins if a[0] == i) / max(n, 1)
    dec = np.concatenate(decisions) if decisions else np.array([1.0])
    acc_moving = 100.0 * sum(a[1] for a in acc_bins) / max(sum(a[2] for a in acc_bins), 1)
    acc_rest = 100.0 * sum(r[0] for r in rest) / max(sum(r[1] for r in rest), 1)
    return {"table": table, "accuracy_bins": acc, "p_true_bins": ptr, "accuracy_moving": acc_moving,
            "accuracy_rest": acc_rest, "balanced_accuracy": 0.5 * (acc_moving + acc_rest),
            "decision_median_pct": 100.0 * float(np.median(dec)), "decided_pct": 100.0 * float(np.mean(dec < 1.0))}


def select_settings(sessions, LLs, rate):
    """Filter settings: the grid point with the best balanced accuracy of the goal inference (mean of the MAP
    accuracy on movement ticks and at rest) over `sessions`."""
    keys = list(GRID)
    combos = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    counts = np.zeros((len(combos), 4))   # correct moving, moving, correct rest, rest
    for sess, LL in zip(tqdm(sessions, desc="filter settings"), LLs):
        mov, phase, _, _ = tick_labels(sess)
        for c, cfg in enumerate(combos):
            st = goal_statistics(sess, filter_posteriors(sess, LL, rate=rate, **cfg), mov, phase)
            counts[c] += st["correct"].sum(), (mov >= 0).sum(), st["rest_ok"].sum(), (mov < 0).sum()
    balanced = 0.5 * (counts[:, 0] / counts[:, 1] + counts[:, 2] / counts[:, 3])
    return combos[int(np.argmax(balanced))]


# =============================================================================
# Figures
# =============================================================================
def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, loc="left", fontsize=10.5, color="#0b0b0b", pad=6)
    ax.set_xlabel(xlabel, color=TEXT2)
    ax.set_ylabel(ylabel, color=TEXT2)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID_C, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_C)
    ax.tick_params(colors=TEXT2, labelsize=8.5)


def _label(sess, i):
    n, h = sess["names"][i], sess["hands"][i]
    return f"{n}/{h}" if sess["names"].count(n) > 1 else n


def plot_accuracy(cue_results, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), facecolor=SURFACE)
    x = np.arange(5, 100, 10)
    colors = {"none": GRAYS[1], "heading": "#1baf7a", "gaze": INIT, "heading + gaze": FIT}
    for name, r in cue_results.items():
        ax[0].plot(x, r["accuracy_bins"], "o-", color=colors[name], label=f"cue prior: {name}", markersize=5)
        ax[1].plot(x, r["p_true_bins"], "o-", color=colors[name], label=f"cue prior: {name}", markersize=5)
    ax[0].set_ylim(0, 100)
    ax[1].set_ylim(0, 1)
    _style(ax[0], "Most probable goal is the true one", "movement progress (%)", "% of the ticks")
    _style(ax[1], "Posterior probability of the true goal", "movement progress (%)", "probability")
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_timeline(sess, post, mode, path):
    """Goal uncertainty over a session: posterior over the hypotheses, its normalized entropy, and the 95 % radius of
    the published wrist prediction 0.5 s ahead with the hypothesis covariance and with the goal-aware one."""
    t = sess["t"]
    K = post.shape[1]
    fig, ax = plt.subplots(3, 1, figsize=(15, 8.5), sharex=True, facecolor=SURFACE,
                           gridspec_kw={"height_ratios": [2.2, 1, 1.2]})
    colors = [GOAL_COLORS.get(n, GRAYS[2]) for n in sess["names"]]
    alphas = [0.55 if (sess["names"].count(n) > 1 and h == "left") else 0.9 for n, h in zip(sess["names"], sess["hands"])]
    polys = ax[0].stackplot(t, post.T, colors=colors, labels=[_label(sess, i) for i in range(K)], edgecolor=SURFACE,
                            linewidth=0.3)
    for poly, a in zip(polys, alphas):
        poly.set_alpha(a)
    for seg, on, off, goals, hands in sess["movements"]:
        for a in ax:
            a.axvspan(on * sess["dt"], off * sess["dt"], color=GRID_C, alpha=0.6, zorder=0)
        ax[0].text(0.5 * (on + off) * sess["dt"], 1.03, " / ".join(g.replace("_", " ") for g in goals[:1]),
                   ha="center", fontsize=8, color=TEXT2)
    ax[0].set_ylim(0, 1.1)
    _style(ax[0], f"{sess['name']}: posterior over the hypotheses (goal {mode}; shaded: movements, true goal above)",
           "", "probability")
    ax[0].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(post > 0, post * np.log(post), 0.0), axis=1) / np.log(K)
    ax[1].plot(t, ent, color=FIT, linewidth=1.6)
    ax[1].set_ylim(0, 1)
    _style(ax[1], "Goal uncertainty: normalized entropy of the posterior (0 = certain, 1 = uniform)", "", "entropy")
    k_half = int(np.argmin(np.abs(sess["times"] - 0.5)))
    MAP = post.argmax(axis=1)
    r_map, r_mix = np.zeros(len(t)), np.zeros(len(t))
    for k in range(len(t)):
        i = MAP[k]
        s = 0 if sess["hands"][i] == "right" else 1
        c_map = sess["pc"][k, i, k_half, s]
        c_mix = goal_aware_cov(sess["pj"][k], sess["pc"][k], post[k], i)[k_half, s]
        r_map[k] = 100 * np.sqrt(hp.CHI2_3_95 * np.trace(c_map) / 3.0)
        r_mix[k] = 100 * np.sqrt(hp.CHI2_3_95 * np.trace(c_mix) / 3.0)
    ax[2].plot(t, r_map, color=GRAYS[1], linewidth=1.4, label="hypothesis covariance only")
    ax[2].plot(t, r_mix, color=FIT, linewidth=1.8, label="with the goal uncertainty")
    _style(ax[2], "95 % radius of the published wrist prediction 0.5 s ahead", "time (s)", "cm")
    ax[2].legend(frameon=False, fontsize=8, loc="upper right")
    ax[2].set_xlim(t[0], t[-1])
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _ellipse(ax, mean, cov, **kw):
    """95 % ellipse of the x-y marginal of a 3-D Gaussian."""
    c = cov[:2, :2]
    vals, vecs = np.linalg.eigh(c)
    vals = np.clip(vals, 1e-10, None)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 2 * np.sqrt(5.991 * vals[::-1])          # chi2_2(0.95) = 5.991
    ax.add_patch(Ellipse(mean[:2], w, h, angle=angle, **kw))


def plot_snapshots(sess, post, movement, mode, path, progress=(0.1, 0.3, 0.5, 0.8)):
    """Top view of the cell at several points of one movement: goals (marker area ~ posterior), observed wrist path,
    ground truth over the horizon, the published (most probable) prediction with its 95 % ellipses (goal-aware
    filled, hypothesis-only dashed) at 1/3, 2/3 and 3/3 of the horizon, and the other hypotheses (p > 5 %)."""
    seg, on, off, goals, hands = sess["movements"][movement]
    frames = sess["frame"]
    fig, axes = plt.subplots(1, len(progress), figsize=(5 * len(progress), 5.2), facecolor=SURFACE)
    names, K = sess["names"], len(sess["names"])
    n_t = len(sess["times"])
    marks = [n_t // 3, 2 * n_t // 3, n_t - 1]
    kp = sess["kp"]
    for ax, pr in zip(axes, progress):
        k = int(np.argmin(np.abs(frames - (on + pr * (off - on)))))
        f = frames[k]
        p = post[k]
        i = int(np.argmax(p))
        hand = sess["hands"][i]
        widx = WRIST[hand]
        s = 0 if hand == "right" else 1
        # goals: probability summed over the hands
        for name in dict.fromkeys(names):
            if name == "idle":
                continue
            g = sess["goals"][names.index(name)]
            pg = sum(p[j] for j in range(K) if names[j] == name)
            ax.scatter(g[0], g[1], s=40 + 900 * pg, color=GOAL_COLORS.get(name, GRAYS[2]), alpha=0.35 + 0.6 * pg,
                       edgecolors="white", linewidths=1.2, zorder=2)
            ax.annotate(f"{name.replace('_', ' ')} {100 * pg:.0f}%", (g[0], g[1]), xytext=(6, 6),
                        textcoords="offset points", fontsize=7.5, color=TEXT2)
        for j in range(K):
            if j != i and p[j] > 0.05 and names[j] != "idle":
                w2 = sess["pj"][k, j][:, WRIST[sess["hands"][j]]]
                ax.plot(w2[:, 0], w2[:, 1], color=GRAYS[1], linewidth=1.0, alpha=0.4 + 0.6 * p[j], zorder=3)
        w_obs = kp[max(on - 50, 0): f + 1, cs.hkm.KP_INDEX[f"{hand}_wrist"]]
        ax.plot(w_obs[:, 0], w_obs[:, 1], color=GRAYS[0], linewidth=1.6, zorder=4, label="observed wrist")
        g_fut = sess["gt"][k][:, widx]
        ax.plot(g_fut[:, 0], g_fut[:, 1], color=GT_C, linewidth=1.6, linestyle="--", zorder=5,
                label="ground truth (1 s)")
        w_pred = sess["pj"][k, i][:, widx]
        ax.plot(w_pred[:, 0], w_pred[:, 1], color=FIT, linewidth=2.2, zorder=6,
                label=f"published: {_label(sess, i)} ({100 * p[i]:.0f}%)")
        c_mix = goal_aware_cov(sess["pj"][k], sess["pc"][k], p, i)
        for m in marks:
            _ellipse(ax, w_pred[m], c_mix[m, s], facecolor=FIT, alpha=0.15, edgecolor=FIT, linewidth=0.8, zorder=1)
            _ellipse(ax, w_pred[m], sess["pc"][k, i, m, s], facecolor="none", edgecolor=FIT, linewidth=1.0,
                     linestyle="--", zorder=6)
        # the cell (goals and paths) sets the view; wide ellipses are clipped
        pts = np.concatenate([sess["goals"][np.isfinite(sess["goals"][:, 0])][:, :2], w_obs[:, :2], g_fut[:, :2],
                              w_pred[:, :2]])
        lo, hi = pts.min(axis=0) - 0.12, pts.max(axis=0) + 0.12
        c, half = 0.5 * (lo + hi), 0.5 * np.max(hi - lo)
        ax.set_xlim(c[0] - half, c[0] + half)
        ax.set_ylim(c[1] - half, c[1] + half)
        ax.set_aspect("equal")
        _style(ax, f"{100 * pr:.0f}% of the movement (t = {sess['t'][k]:.2f} s)", "x (m)", "y (m)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle(f"{sess['name']}: movement towards {' / '.join(goals)} (goal {mode}) - 95 % regions at 1/3, 2/3, 3/3 s: "
                 "filled = with the goal uncertainty, dashed = hypothesis only", fontsize=10, color=TEXT2)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


# =============================================================================
# Output
# =============================================================================
def table_html(table, title, uncertainty):
    rows = []
    for split, by_method in table.items():
        rows.append(f"<tr><th colspan='{len(METRICS) + 1}' class='split'>{split} ticks</th></tr>")
        best = {}
        for j, (key, _) in enumerate(METRICS):
            vals = {m: v[j] for m, v in by_method.items() if np.isfinite(v[j]) and m != "oracle"}
            if key.startswith("coverage"):
                best[key] = min(vals, key=lambda m: abs(vals[m] - 95.0)) if vals else None
            else:
                best[key] = min(vals, key=vals.get) if vals else None
        for m, v in by_method.items():
            cells = []
            for j, (key, _) in enumerate(METRICS):
                txt = "–" if not np.isfinite(v[j]) else f"{v[j]:.2f}"
                cells.append(f"<td><b>{txt}</b></td>" if best[key] == m else f"<td>{txt}</td>")
            rows.append(f"<tr><td>{m} <span class='note'>{METHOD_NOTES[m]}</span></td>{''.join(cells)}</tr>")
    head = "".join(f"<th>{label}</th>" for _, label in METRICS)
    return (f"<h2>{title}</h2><p class='note'>published covariance: {uncertainty}</p>"
            f"<table><tr><th>method</th>{head}</tr>{''.join(rows)}</table>")


def write_html(path, meta, results, cue_rows, figures):
    style = """<style>body{font-family:sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;color:#0b0b0b;
background:#fcfcfb}table{border-collapse:collapse;margin:8px 0 24px}td,th{border:1px solid #e6e5e0;padding:4px 10px;
text-align:right}td:first-child{text-align:left}th.split{text-align:left;background:#f0efec}img{max-width:100%}
.note{color:#52514e;font-size:0.85em}</style>"""
    cue = "".join(f"<tr><td>{r[0]}</td>" + "".join(f"<td>{x:.1f}</td>" for x in r[1:]) + "</tr>" for r in cue_rows)
    body = [f"<h1>Online prediction on CARI v2 sessions</h1><p>{meta}</p>",
            "<h2>Goal inference (held-out subjects)</h2><table><tr><th>cue prior</th><th>most probable goal correct, "
            "movements (%)</th><th>correct, rest (%)</th><th>balanced (%)</th><th>decision (% of the movement, "
            f"median)</th><th>movements decided (%)</th></tr>{cue}</table>"]
    body += [f"<img src='{f}'>" for f in figures]
    for title, (res, unc) in results.items():
        body.append(table_html(res["table"], title, unc))
    path.write_text(f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Online prediction</title>{style}"
                    f"</head><body>{''.join(body)}</body></html>")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    os.chdir(ROOT)
    ocfg = cfg.online
    if ocfg.goal_mode not in ("inferred", "known") or ocfg.uncertainty not in ("map", "mixture"):
        raise ValueError("online.goal_mode must be inferred | known, online.uncertainty map | mixture")
    dev = ck.setup_jax(ocfg.device)
    params, pred_noise, source = model_parameters(cfg)
    H, max_iter = int(cfg.model.horizon), int(cfg.model.max_iter)
    train_subjects, test_subjects = ck.split_subjects(cfg.data)
    subjects = list(dict.fromkeys(train_subjects + test_subjects))
    tag = "" if ocfg.source == "cari" else "_kimodo" + ("_noise" if ocfg.noise else "")
    out_dir = ROOT / (ocfg.output_dir or f"output/goal_inference{tag}_{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if ocfg.reuse:
        with open(ROOT / ocfg.reuse / "sessions.pkl", "rb") as f:
            durations, sessions = pickle.load(f)
    else:
        durations = nominal_durations(subjects, cfg.data.velocity)
        if ocfg.source == "cari":
            sources = (cs.load_session(s, cfg.data.velocity) for s in subjects)
        else:
            bank = cs.noise_bank(subjects, cfg.data.velocity) if ocfg.noise else None
            sources = cs.load_kimodo_sessions(ROOT / ocfg.clips, subjects, noise=bank)
        sessions = []
        with jax.default_device(dev):
            for S in tqdm(list(sources), desc="sessions"):
                sessions.append(replay_session(S, ocfg, params, H, max_iter, pred_noise, durations[S.subject]))
        with open(out_dir / "sessions.pkl", "wb") as f:
            pickle.dump((durations, sessions), f)
    LLs = [lagged_loglik(sess) for sess in sessions]
    latency = np.concatenate([sess["latency"][3:] for sess in sessions])
    train = [i for i, s in enumerate(sessions) if s["subject"] in train_subjects]
    test = [i for i, s in enumerate(sessions) if s["subject"] in test_subjects]
    settings = select_settings([sessions[i] for i in train], [LLs[i] for i in train], ocfg.rate)
    cfgs = {s["subject"]: settings for s in sessions}
    pick = lambda idx: ([sessions[i] for i in idx], [LLs[i] for i in idx])
    unc = "goal-aware (mixture)" if ocfg.uncertainty == "mixture" else "most probable hypothesis only"
    results = {f"Held-out subjects ({', '.join(test_subjects)})": (aggregate(*pick(test), cfgs, ocfg.rate), unc),
               "All subjects (the training subjects' demonstrations were used by the IOC fit)":
                   (aggregate(*pick(range(len(sessions))), cfgs, ocfg.rate), unc)}
    cue_results, cue_rows = {}, []
    for name, (kh, kg) in CUES.items():
        c = {**settings, "kappa_heading": settings["kappa_heading"] if kh is None else kh,
             "kappa_gaze": settings["kappa_gaze"] if kg is None else kg}
        r = aggregate(*pick(test), {s["subject"]: c for s in sessions}, ocfg.rate)
        cue_results[name] = r
        cue_rows.append((name, r["accuracy_moving"], r["accuracy_rest"], r["balanced_accuracy"],
                         r["decision_median_pct"], r["decided_pct"]))

    figures = ["goal_accuracy.png"]
    plot_accuracy(cue_results, out_dir / "goal_accuracy.png")
    for i in test:
        sess = sessions[i]
        post_inf, post_known = posteriors(sess, LLs[i], settings, ocfg.rate)
        post = post_inf if ocfg.goal_mode == "inferred" else post_known
        name = sess["name"].replace("/", "_")
        plot_timeline(sess, post, ocfg.goal_mode, out_dir / f"timeline_{name}.png")
        figures.append(f"timeline_{name}.png")
        for m, mv in enumerate(sess["movements"]):
            if mv[0] in (1, 3, 5, 7):
                fn = f"goal_uncertainty_{name}_segment{mv[0]}.png"
                plot_snapshots(sess, post, m, ocfg.goal_mode, out_dir / fn)
                figures.append(fn)

    meta = (f"{'CARI v2 recordings' if ocfg.source == 'cari' else 'Kimodo clips'}: {len(sessions)} sessions "
            f"({cfg.data.velocity}; held out: {', '.join(test_subjects)}), {len(latency)} ticks at {ocfg.rate:g} Hz, "
            f"last {ocfg.observation_time:g} s observed ({ocfg.angles} IK angles), {ocfg.horizon:g} s predicted; "
            f"{len(sessions[0]['names'])} hypotheses ({', '.join(_label(sessions[0], i) for i in range(len(sessions[0]['names'])))}). "
            f"Model: {source}, prediction noise {pred_noise:.3g}. Published prediction: goal {ocfg.goal_mode}, "
            f"covariance {unc}. Prediction of all hypotheses: median {1000 * np.median(latency):.0f} ms on "
            f"{dev.platform}. Filter settings selected on the training subjects: {settings}.")
    write_html(out_dir / "summary.html", meta, results, cue_rows, figures)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subjects", "split", "method"] + [k for k, _ in METRICS])
        for title, (r, _) in results.items():
            for split, by_method in r["table"].items():
                for m, v in by_method.items():
                    w.writerow([title, split, m] + [f"{x:.3f}" for x in v])
    (out_dir / "results.json").write_text(json.dumps({
        "online": OmegaConf.to_container(ocfg, resolve=True), "model": source, "pred_noise": pred_noise,
        "test_subjects": test_subjects, "nominal_duration": durations, "filter_settings": settings,
        "latency_ms_median": 1000 * float(np.median(latency)),
        "cues": {k: {"accuracy_moving": v["accuracy_moving"], "accuracy_rest": v["accuracy_rest"],
                     "balanced_accuracy": v["balanced_accuracy"], "decision_median_pct": v["decision_median_pct"],
                     "decided_pct": v["decided_pct"], "accuracy_bins": v["accuracy_bins"].tolist(),
                     "p_true_bins": v["p_true_bins"].tolist()} for k, v in cue_results.items()},
        "metrics": [k for k, _ in METRICS],
        "tables": {t: {sp: {m: v.tolist() for m, v in bm.items()} for sp, bm in r["table"].items()}
                   for t, (r, _) in results.items()}}, indent=2))
    latest = ROOT / "output" / f"latest_goal_inference{tag}"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out_dir.resolve(), target_is_directory=True)

    print(f"\n{meta}\n")
    for row in cue_rows:
        print(f"cues {row[0]:15s} goal correct moving {row[1]:5.1f} %  rest {row[2]:5.1f} %  balanced {row[3]:5.1f} %"
              f"  decision at {row[4]:5.1f} % of the movement")
    for title, (r, _) in results.items():
        print(f"\n{title}  [MPJPE / wrists ADE / coverage hypothesis cov / coverage goal-aware cov]")
        for split, by_method in r["table"].items():
            print(f"  {split:7s}" + "".join(f"  {m}: {v[0]:.2f}/{v[1]:.2f}/{v[3]:.0f}%/{v[4]:.0f}%"
                                            for m, v in by_method.items()))
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()
