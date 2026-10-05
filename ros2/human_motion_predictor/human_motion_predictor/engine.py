"""Prediction engine of the node: receding-horizon prediction of the goal hypotheses and goal inference
(prophet_ioc.human_prediction), without ROS.

It runs either inside the node, or in a worker process with its own python environment (`python -m
human_motion_predictor.engine`, parameter `worker_python`): ROS 2 Jazzy's python extensions need numpy < 2, which
caps JAX at 0.7.1, ~3x slower than the current JAX on the GPU for this problem (130 vs 45 ms for 8 hypotheses).
Requests and replies go through stdin / stdout as length-prefixed messages (JSON header + raw array buffers: no
pickle, which does not cross numpy 1 / 2).

uncertainty (config): "mixture" (default) publishes wrist covariances that include the goal uncertainty (goal_aware),
"map" those of each hypothesis alone.

Requests: {"op": "init", **config} -> {"ok": True, "compile_s"}; {"op": "predict", "t", "hist" (n, 28), "dt",
"body" (8,), "head" (3,) | None, "gaze" (3,) | None, "goals" (K, 3) | None, "reset"} -> {"post" (K,), "joints"
(K, n_t, 9, 3), "cov" (K, n_t, 3, 3, 3) (wrist, elbow, passive wrist), "target" (K, 3), "arrival" (K,), "temporary"
(K,), "elapsed"}.
"""

import json
import os
import struct
import sys
import time
from typing import Dict, List

# Single-threaded OpenBLAS (LAPACK of JAX's CPU linear algebra): with its thread pool the small factorizations of the
# prediction were up to 100x slower on a loaded machine. Before numpy is imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

COV_KEYS = ("wrist", "elbow", "passive_wrist")


# ----------------------------------------------------------------------------- messages
def _encode(obj: Dict) -> bytes:
    header, blobs, offset = {}, [], 0
    for k, v in obj.items():
        if isinstance(v, np.ndarray):
            b = np.ascontiguousarray(v).tobytes()
            header[k] = {"__nd__": [v.dtype.str, list(v.shape), offset, len(b)]}
            blobs.append(b)
            offset += len(b)
        else:
            header[k] = v
    h = json.dumps(header).encode()
    return struct.pack("<II", len(h), offset) + h + b"".join(blobs)


def _decode(data: bytes) -> Dict:
    nh, _ = struct.unpack("<II", data[:8])
    header = json.loads(data[8: 8 + nh])
    body = memoryview(data)[8 + nh:]
    out = {}
    for k, v in header.items():
        if isinstance(v, dict) and "__nd__" in v:
            dtype, shape, off, n = v["__nd__"]
            out[k] = np.frombuffer(body[off: off + n], dtype=np.dtype(dtype)).reshape(shape).copy()
        else:
            out[k] = v
    return out


def send(stream, obj: Dict) -> None:
    data = _encode(obj)
    stream.write(struct.pack("<Q", len(data)) + data)
    stream.flush()


def recv(stream) -> Dict:
    head = stream.read(8)
    if len(head) < 8:
        raise EOFError("prediction worker closed")
    (n,) = struct.unpack("<Q", head)
    return _decode(stream.read(n))


# ----------------------------------------------------------------------------- engine
class Engine:
    """config: hypotheses [[name, hand, [x, y, z] | None]], params {HumanKinematicParams fields and model switches,
    params_from_config}, H, max_iter, tol (early stopping of the solver, None = always max_iter), settings
    {hp.PredictionSettings fields: covariance, residual, observability, temperature, belief_steps} (default: model
    covariance, fully observed), pred_noise (random-walk covariance only; None = the model covariance), grasp_offset
    (m, wrist goal before the goal locations, hp.wrist_goal; default 0), horizon, nominal_duration, stop_time, times
    (prediction sample times), filter {switch_rate, evidence_lag, temperature, obs_noise}, kappa_heading,
    kappa_gaze, device, warmup_samples, warmup_dt."""

    def __init__(self, config: Dict):
        import jax
        try:
            jax.config.update("jax_default_device", jax.devices(config["device"])[0])
            self.device = config["device"]
        except RuntimeError:
            self.device = jax.devices()[0].platform
        from prophet_ioc import human_prediction as hp
        from prophet_ioc.envs.human_kinematic_reaching import params_from_config
        self.hp, self.c = hp, config
        self.params = params_from_config(config["params"])
        self.settings = hp.PredictionSettings(**(config.get("settings") or {}))
        self.pred_noise = None if config.get("pred_noise") is None or not self.settings.random_walk \
            else float(config["pred_noise"])
        self.hypotheses = [hp.Hypothesis(n, h, None if g is None else tuple(float(x) for x in g))
                           for n, h, g in config["hypotheses"]]
        f = config["filter"]
        self.filter = hp.GoalFilter(self.hypotheses, self.pred_noise, switch_rate=f["switch_rate"],
                                    evidence_lag=f["evidence_lag"], temperature=f["temperature"],
                                    obs_noise=f["obs_noise"])
        self.times = np.asarray(config["times"], dtype=float)
        # warm start: each solve starts from the plan of the previous tick (hp.shift_plan) and runs at most
        # "warm_max_iter" iterations (default 2; the first tick and the ticks after a reset run "max_iter" from zero).
        # Early stopping (tol) does not save time here: the batch of hypotheses iterates until all have converged.
        # On a replayed CARI reach, warm 2 iterations matched cold 3 (wrist 0.15 cm mean difference), 18 % faster.
        self.warm_start = bool(config.get("warm_start", True))
        self.warm_max_iter = int(config.get("warm_max_iter", 2))
        self.plans = {}

    def warmup(self) -> float:
        t0 = time.perf_counter()
        self.plans = {}
        q = np.zeros(28, dtype=np.float32)
        q[2], q[6] = 1.2, 1.0
        n = int(self.c["warmup_samples"])
        hist = np.repeat(q[None], n, axis=0)
        hist[:, 0] += np.linspace(0.0, 0.01, n)
        body = np.array([0.35, 0.45, 0.25, 0.3, 0.27, 0.4, 0.4, 0.2], dtype=np.float32)
        for k in range(3):   # compiles the cold solve and (k >= 1, no reset) the warm-started one
            self.predict({"t": 0.0, "hist": hist, "dt": self.c["warmup_dt"], "body": body, "head": None,
                          "gaze": None, "goals": None, "reset": k == 0})
        self.filter.reset()
        self.plans = {}
        return time.perf_counter() - t0

    def predict(self, req: Dict) -> Dict:
        hp, c = self.hp, self.c
        t0 = time.perf_counter()
        if req.get("goals") is not None:   # known target (from a topic): same hypotheses, current goal
            goals = np.asarray(req["goals"], dtype=float)
            self.hypotheses = [h if h.idle else hp.Hypothesis(h.name, h.hand, tuple(goals[i]))
                               for i, h in enumerate(self.hypotheses)]
            self.filter.hypotheses = self.hypotheses
        if req.get("reset"):
            self.filter.reset()
            self.plans = {}
        preds, hs = hp.predict_hypotheses(np.asarray(req["hist"], dtype=np.float32), float(req["dt"]),
                                          np.asarray(req["body"], dtype=np.float32), self.hypotheses, self.params,
                                          int(c["H"]), self.warm_max_iter if (self.warm_start and self.plans)
                                          else int(c["max_iter"]), horizon=float(c["horizon"]),
                                          nominal_duration=float(c["nominal_duration"]),
                                          stop_time=float(c["stop_time"]), tol=c.get("tol"), settings=self.settings,
                                          grasp_offset=float(c.get("grasp_offset", 0.0)),
                                          warm_start=self.plans if self.warm_start else None, t_now=float(req["t"]))
        if self.warm_start:
            self.plans = {(p.hypothesis.name, p.hypothesis.hand): p.plan for p in preds}
        head = None if req.get("head") is None else np.asarray(req["head"], dtype=float)
        gaze = None if req.get("gaze") is None else np.asarray(req["gaze"], dtype=float)
        log_prior = hp.goal_cue_logprior(self.hypotheses, hs, head, gaze, float(c["kappa_heading"]),
                                         float(c["kappa_gaze"]))
        post = self.filter.update(float(req["t"]), preds, log_prior)
        joints, covs = [], []
        for p in preds:
            j, cv = hp.sample_prediction(p.prediction, self.times, self.pred_noise)
            joints.append(np.stack([j[name] for name in hp.JOINTS], axis=1))
            covs.append(np.stack([cv[k] for k in COV_KEYS], axis=1))
        joints, covs = np.stack(joints), np.stack(covs)
        if c.get("uncertainty", "mixture") == "mixture":
            covs = self.goal_aware(joints, covs, np.asarray(post))
        return {"post": np.asarray(post, dtype=np.float64), "joints": joints, "cov": covs,
                "target": np.stack([p.target for p in preds]).astype(np.float64),
                "arrival": np.array([p.arrival for p in preds]), "temporary": np.array([p.temporary for p in preds]),
                "elapsed": time.perf_counter() - t0}


    def goal_aware(self, joints, covs, post):
        """Wrist covariances of every hypothesis including the goal uncertainty: around hypothesis i,
        sum_g p_g (Sigma_g + (mu_g - mu_i) (mu_g - mu_i)^T) for each wrist (law of total covariance over the
        hypotheses); the elbow keeps the hypothesis covariance."""
        hp = self.hp
        widx = [hp.JOINTS.index("right_wrist"), hp.JOINTS.index("left_wrist")]
        right = np.array([h.hand == "right" for h in self.hypotheses])
        # covariance of the right / left wrist of each hypothesis: its reaching wrist (0) or its passive one (2)
        side_cov = np.stack([np.where(right[:, None, None, None], covs[:, :, 0], covs[:, :, 2]),
                             np.where(right[:, None, None, None], covs[:, :, 2], covs[:, :, 0])], axis=2)
        mu = joints[:, :, widx]                                            # (K, n_t, 2, 3)
        out = covs.copy()
        for i in range(len(self.hypotheses)):
            d = mu - mu[i][None]
            mixed = np.einsum("g,gtsij->tsij", post, side_cov + d[..., :, None] * d[..., None, :])
            own, other = (0, 1) if right[i] else (1, 0)
            out[i, :, 0], out[i, :, 2] = mixed[:, own], mixed[:, other]
        return out


class WorkerClient:
    """The engine in a worker process (python executable of another environment), with the same interface."""

    def __init__(self, python: str, config: Dict, pythonpath: str):
        import os
        import subprocess
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "JAX_PLATFORMS", "VIRTUAL_ENV")}
        env["PYTHONPATH"] = pythonpath
        self.proc = subprocess.Popen([python, "-m", "human_motion_predictor.engine"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, env=env)
        send(self.proc.stdin, {"op": "init", **config})
        reply = recv(self.proc.stdout)
        self.compile_s, self.device = reply["compile_s"], reply["device"]

    def predict(self, req: Dict) -> Dict:
        send(self.proc.stdin, {"op": "predict", **{k: v for k, v in req.items()}})
        return recv(self.proc.stdout)

    def close(self):
        self.proc.terminate()


def main():
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr   # library prints must not corrupt the replies
    engine = None
    while True:
        try:
            req = recv(stdin)
        except EOFError:
            return
        if req.pop("op") == "init":
            engine = Engine(req)
            send(stdout, {"ok": True, "compile_s": engine.warmup(), "device": engine.device})
        else:
            for k in ("head", "gaze", "goals"):
                req.setdefault(k, None)
            send(stdout, engine.predict(req))


if __name__ == "__main__":
    main()
