"""Human Kinematic Model Reaching Environment for NIOC.

Integrates the 28-DOF anthropomorphic human kinematic model (JAX implementation from
`human_kinematic_model_jax`) into the NIOC optimal control and probabilistic motion prediction
framework.

Supports:
- "upper_body" mode (19 active DOFs): Head (2), Chest Position & Orientation (6),
  Thoracic Spine / Shoulder girdle (1), Pelvis / Hip rotation (2), Left & Right Arms (8).
- "full_body" mode (27 active DOFs): Upper body + Left & Right Legs (8).

Kinematic conventions:
- Pure JAX analytical kinematics (differentiable, jittable, vmap-friendly).
- Chest orientation q[3:6]: a rotation vector relative to the chest orientation at the handover (q_chest_ref), so it
  is 0 at the start of a prediction and far from the pi singularity; the quaternion (x, y, z, w) is only formed inside
  the FK (build_q28: q_chest_ref * exp(q[3:6]), analytical map without gradient NaNs at 0).
- Rigid link kinematics strictly preserving human anatomical bone lengths.
- Gauss-Newton quadratization of the reaching costs for stable iLQG convergence (gauss_newton_sq).

Dynamics: per joint, qdd = u - b qd + motor noise, discretized exactly with zero-order hold (prophet_ioc.envs.zoh;
b = params.damping, default 0). Cost: unit terminal wrist-to-target weight, learnable effort / velocity / pelvis
terms, hand-tuned joint-limit penalty (HumanKinematicParams).
"""

from functools import lru_cache
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import jacobian

import human_kinematic_model_jax as hkm
from prophet_ioc.envs.base import Env
from prophet_ioc.envs.zoh import zoh_coefficients, zoh_noise_chol, zoh_process_cov


# Attributes of HumanKinematicReaching that are pytree leaves (traced under jit / batched by vmap); the others are static.
_ENV_LEAVES = ("dt", "dt_ref", "q_chest_ref", "legs_nominal", "body_params", "q0", "x0", "target", "target_vel",
               "right_hand", "target_left", "target_vel_left", "alpha_right", "alpha_left")



class HumanKinematicParams(NamedTuple):
    """Parameters of HumanKinematicReaching: the single source of truth for what the IOC fit may learn.

    LEARNABLE (class attribute) lists the cost weights the fit may infer (prophet_ioc / evaluation/cari_kinematic.py
    select the subset of a configuration: e.g. without the pelvis displacement term base_disp_cost is not learned);
    every other field is a fixed, hand-tuned value read from config/model/human_kinematic.yaml
    (params_from_config) and saved with the fitted weights.

    Cost scale: the terminal wrist-to-target term has weight 1 (it sets the scale of the cost and is not a parameter);
    the defaults are the former values (terminal weight 100) divided by 100, i.e. the same optimal behaviour.
    """
    # --- learnable cost weights ---
    velocity_cost: float = 1e-4        # terminal wrist velocity (to target_vel)
    running_vel_cost: float = 0.0      # joint velocities at every step (velocity_floor below it)
    base_disp_cost: float = 0.1        # pelvis displacement from its position at the start of the horizon
    running_target_cost: float = 0.0   # wrist-to-target distance at every step (not only at T)
    # Effort 0.5 * sum_j w_{g(j)} u_j^2 per joint group g (former c_a = 1e-4 times the former group weight, / 100)
    w_act_pelvis: float = 2e-5         # pelvis translation q[0:3]
    w_act_trunk: float = 3e-6          # chest rotation vector q[3:6]
    w_act_spine: float = 3e-6          # shoulder rot x and hip rot z/x, q[6:9]
    w_act_reach_arm: float = 1e-6      # reaching arm (q[9:13] right / q[13:17] left)
    w_act_passive_arm: float = 3e-6    # the other arm
    w_act_head: float = 2e-6           # head rot x/y, q[17:19]
    w_act_legs: float = 8e-6           # legs q[19:27] (full_body only)
    # --- noise ---
    motor_noise: float = 0.1           # signal-dependent motor noise (std of qd per step: sigma |u| sqrt(dt))
    motor_noise_add: float = 0.0       # additive (signal-independent) white-noise acceleration intensity
    obs_noise: float = 1.0             # observation noise of the partially observed model (y = x + sqrt(dt) sigma_o w)
    # Likelihood-only white-noise acceleration intensity (same ZOH structure as motor_noise_add): absorbs model
    # mismatch in the IOC likelihood (prophet_ioc.infer.multi_env). Not part of the dynamics, so the controller does not
    # plan against it, unlike motor_noise.
    residual_noise: float = 0.0
    # Prediction only: scale of the handover Kalman covariance P0 that the predictive covariance starts from
    # (prophet_ioc.human_prediction); fitted with residual_noise by the predictive stage 2 of train.py (ioc.noise_fit).
    handover_cov_scale: float = 1.0
    # --- fixed, hand-tuned (config/model/human_kinematic.yaml, never learned) ---
    damping: float = 0.0               # b of qdd = u - b qd (1/s), exact ZOH discretization (prophet_ioc.envs.zoh)
    velocity_floor: float = 1e-6       # lower bound of the running joint-velocity weight (keeps Q_xx regular)
    w_lim: float = 1.0                 # joint-limit penalty weight (0 = no penalty)
    chest_rot_limit: float = 1.0       # rad, bound on the norm of the chest rotation vector (joint-limit penalty)

    # Cost weights the IOC fit may learn (the configuration selects a subset, see learnable_params)
    LEARNABLE = ("velocity_cost", "running_vel_cost", "base_disp_cost", "running_target_cost", "w_act_pelvis",
                 "w_act_trunk", "w_act_spine", "w_act_reach_arm", "w_act_passive_arm", "w_act_head", "w_act_legs")

    @staticmethod
    def get_params_type() -> type:
        """Return the parameter record type used by this environment."""
        return HumanKinematicParams

    @staticmethod
    def get_params_bounds() -> Tuple["HumanKinematicParams", "HumanKinematicParams"]:
        """Bounds of the IOC fit (log10 space). Effort weights: their default x [1e-2, 1e2]; the other cost weights:
        the former bounds / 100 (cost scale). The fixed fields keep their defaults (never fitted)."""
        lo = HumanKinematicParams(
            velocity_cost=1e-6, running_vel_cost=1e-7, base_disp_cost=1e-3, running_target_cost=1e-5,
            w_act_pelvis=2e-7, w_act_trunk=3e-8, w_act_spine=3e-8, w_act_reach_arm=1e-8, w_act_passive_arm=3e-8,
            w_act_head=2e-8, w_act_legs=8e-8, motor_noise=1e-2, motor_noise_add=1e-3, obs_noise=1e-3,
            residual_noise=1e-3, handover_cov_scale=1e-2,
        )
        hi = HumanKinematicParams(
            velocity_cost=1e-2, running_vel_cost=1e-2, base_disp_cost=1.0, running_target_cost=10.0,
            w_act_pelvis=2e-3, w_act_trunk=3e-4, w_act_spine=3e-4, w_act_reach_arm=1e-4, w_act_passive_arm=3e-4,
            w_act_head=2e-4, w_act_legs=8e-4, motor_noise=1.0, motor_noise_add=10.0, obs_noise=10.0,
            residual_noise=10.0, handover_cov_scale=1e2,
        )
        return lo, hi


def params_from_config(cfg: Mapping[str, Any]) -> HumanKinematicParams:
    """HumanKinematicParams of a model configuration (config/model/human_kinematic.yaml, or the "params" of a train.py
    params.json): the fields present in cfg, the class defaults for the others; other keys are ignored. The switches
    pelvis_displacement_cost / joint_limit_cost = false set base_disp_cost / w_lim to 0 (term removed)."""
    fields = HumanKinematicParams._fields
    params = HumanKinematicParams(**{k: float(v) for k, v in cfg.items() if k in fields})
    if not cfg.get("pelvis_displacement_cost", True):
        params = params._replace(base_disp_cost=0.0)
    if not cfg.get("joint_limit_cost", True):
        params = params._replace(w_lim=0.0)
    return params


def learnable_params(cfg: Mapping[str, Any], mode: str = "upper_body") -> Tuple[str, ...]:
    """The learnable fields (HumanKinematicParams.LEARNABLE) that the model configuration cfg keeps: without the
    switched-off terms (pelvis_displacement_cost: false removes base_disp_cost; running_target_cost is learned only
    when it is enabled, running_target: true) and, in upper_body mode, without the leg effort weight."""
    names = list(HumanKinematicParams.LEARNABLE)
    if not cfg.get("pelvis_displacement_cost", True):
        names.remove("base_disp_cost")
    if not cfg.get("running_target", False):
        names.remove("running_target_cost")
    if mode == "upper_body":
        names.remove("w_act_legs")
    return tuple(names)


@lru_cache(maxsize=None)
def joint_limits(mode: str = "upper_body") -> Tuple[np.ndarray, np.ndarray]:
    """Lower / upper joint limits (n_dof,) of the active DOFs from hkm.default_joint_limits() (28-DOF layout, the
    anatomical limits of the IK): shoulder rot x, hip, arms, head (and legs in full_body). The pelvis translation q[0:3]
    is unbounded and the chest rotation vector q[3:6] has a bound on its norm instead (chest_rot_limit), so these
    entries are +-UNBOUNDED."""
    lim = np.asarray(hkm.default_joint_limits(), dtype=np.float64)   # (28, 2)
    n = 19 if mode == "upper_body" else 27
    lo, hi = np.full(n, -UNBOUNDED), np.full(n, UNBOUNDED)
    # active DOF index -> 28-DOF index (build_q28)
    pairs = [(6, 7), (7, 8), (8, 9)] + [(9 + k, 10 + k) for k in range(8)] + [(17, 26), (18, 27)]
    if mode == "full_body":
        pairs += [(19 + k, 18 + k) for k in range(8)]
    for i, j in pairs:
        lo[i], hi[i] = lim[j]
    return lo.astype(np.float32), hi.astype(np.float32)


UNBOUNDED = 1e4  # "no limit" of joint_limits (finite: no inf arithmetic in the penalty and its derivatives)


def quat_from_rotvec(w: jnp.ndarray) -> jnp.ndarray:
    """Computes unit quaternion (x, y, z, w) from rotation vector w in R^3.

    Analytically smooth and finite everywhere, especially at w = 0 (no division by zero, no NaNs).
    """
    th2 = jnp.dot(w, w)
    th = jnp.sqrt(th2 + 1e-24)
    half_th = 0.5 * th
    s = jnp.where(th2 < 1e-8, 0.5 - th2 / 48.0, jnp.sin(half_th) / th)
    c = jnp.where(th2 < 1e-8, 1.0 - th2 / 8.0, jnp.cos(half_th))
    return jnp.concatenate([s * w, jnp.array([c], dtype=w.dtype)])


def gauss_newton_sq(residual_fn, x: jnp.ndarray) -> jnp.ndarray:
    """|r(x)|^2 with its exact value and gradient but the Gauss-Newton Hessian 2 J^T J (PSD), for iLQG.

    The full Hessian is d2|r|^2 = 2 J^T J + 2 sum_i r_i d2 r_i. Linearizing r around the current point x0,
    r(x) ~ r0 + J0 (x - x0), and squaring gives the same value and gradient at x0 (2 J0^T r0) but the Hessian
    2 J0^T J0 >= 0. The dropped term can be negative far from the goal: for r = sin(x) - g with g = 0 at x = pi/2
    the full second derivative of r^2 is 2 cos^2 x - 2 sin^2 x = -2, which would make Q_xx indefinite; it vanishes at
    the goal (r = 0), so the Gauss-Newton and exact Hessians agree at a perfect reach.

    jax.jvp, not jax.linearize: the same r0 + J0 (x - x0) (bit for bit), but nested inside the jacfwd(grad) of the
    cost quadratization, vmap and the unrolled solver, jax.linearize goes through JAX's fallback linearize rule
    (_lift_linearized), where the IOC fit crashed the interpreter (segmentation fault / illegal instruction while
    tracing, JAX 0.11.2, Python 3.14, on one of two machines).
    """
    x0 = jax.lax.stop_gradient(x)
    r0, dr = jax.jvp(residual_fn, (x0,), (x - x0,))
    r = jax.lax.stop_gradient(r0) + dr
    return jnp.sum(r**2)


# Canonical keypoint indices matching hkm.KEYPOINT_NAMES
KEYPOINT_NAMES = hkm.KEYPOINT_NAMES
KP_HEAD = 0
KP_LEFT_SHOULDER = 1
KP_LEFT_ELBOW = 2
KP_LEFT_WRIST = 3
KP_LEFT_HIP = 4
KP_LEFT_KNEE = 5
KP_LEFT_ANKLE = 6
KP_RIGHT_SHOULDER = 7
KP_RIGHT_ELBOW = 8
KP_RIGHT_WRIST = 9
KP_RIGHT_HIP = 10
KP_RIGHT_KNEE = 11
KP_RIGHT_ANKLE = 12


@jax.tree_util.register_pytree_node_class
class HumanKinematicReaching(Env):
    """3D Human Reaching Environment based on the 28-DOF Human Kinematic Model.

    Controls an anthropomorphic human body to reach for 3D spatial targets while
    maintaining natural posture and physiological motion constraints.

    The environment is a JAX pytree: the trial-specific arrays (`_ENV_LEAVES`: dt, target, postures, body
    parameters, ...) are leaves, while mode, root joint, reaching hand and the cost options are static. Passing the
    environment as a regular (non-static) jit argument therefore compiles once for all trials with the same static
    configuration, and environments of several trials can be stacked (`stack_envs`) and vmapped.
    """
    # Q_uu regularization of the backward passes (Env.reg_eps): the former 1e-4 / 100, like every cost weight
    reg_eps = 1e-6

    def __init__(
        self,
        mode: str = "upper_body",
        dt: float = 0.02,
        target: Optional[Union[jnp.ndarray, Tuple[float, float, float]]] = None,
        target_vel: Optional[Union[jnp.ndarray, Tuple[float, float, float]]] = None,
        q0: Optional[jnp.ndarray] = None,
        body_params: Optional[jnp.ndarray] = None,
        reaching_hand: str = "right",
        q_chest_ref: Optional[jnp.ndarray] = None,
        legs_nominal: Optional[jnp.ndarray] = None,
        root_joint: str = "pelvis",
        dt_scaled_cost: bool = False,
        dt_ref: float = 0.05,
        target_left: Optional[Union[jnp.ndarray, Tuple[float, float, float]]] = None,
        target_vel_left: Optional[Union[jnp.ndarray, Tuple[float, float, float]]] = None,
        alpha_right: Optional[float] = None,
        alpha_left: Optional[float] = None,
    ):
        """Initializes the human kinematic reaching environment.

        Args:
            mode: "upper_body" (19 active DOFs) or "full_body" (27 active DOFs).
            dt: Sampling time step in seconds (default: 0.02 s = 50 Hz).
            target: [x, y, z] Cartesian target position for the reaching wrist (or right wrist).
            target_vel: [vx, vy, vz] wrist velocity at the end of the horizon (terminal velocity cost; default zero,
                i.e. at rest on the target; non-zero for an intermediate target on the way to a farther goal).
            q0: Initial configuration vector (19 DOFs for upper_body, 27 for full_body).
                When root_joint="pelvis", q[0:3] must be the pelvis 3D position.
                When root_joint="chest", q[0:3] must be the chest 3D position (legacy).
                q0[0:3] is also the reference of the pelvis displacement cost (pelvis at the start of the horizon).
            body_params: 8 body segment parameters in meters:
                [shoulder_dist, chest_hip_dist, hip_dist, upper_arm, lower_arm, thigh, shank, head_dist].
            reaching_hand: "right", "left", "both", or "any": the hand is the traced leaf
                right_hand (1 right, 0 left, or explicit alpha_right/alpha_left).
            q_chest_ref: Reference chest quaternion [x, y, z, w]: chest_quat = q_chest_ref * exp(q[3:6]).
            legs_nominal: Nominal leg joint angles (8,) for upper_body mode.
            root_joint: "pelvis" (default) or "chest".
            dt_scaled_cost: If True, the running cost is multiplied by dt / dt_ref.
            dt_ref: Reference time step of dt_scaled_cost, in seconds.
            target_left: [x, y, z] Cartesian target position for the left wrist (when both/dual mode).
            target_vel_left: [vx, vy, vz] terminal left wrist velocity.
            alpha_right: Weight of right wrist target cost (default: 1.0 for right/both, 0.0 for left).
            alpha_left: Weight of left wrist target cost (default: 1.0 for left/both, 0.0 for right).
        """
        self.mode = mode.lower()
        if self.mode not in ("upper_body", "full_body"):
            raise ValueError(f"mode must be 'upper_body' or 'full_body', got {mode}")
        self.root_joint = root_joint.lower()
        if self.root_joint not in ("pelvis", "chest"):
            raise ValueError(f"root_joint must be 'pelvis' or 'chest', got {root_joint}")

        self.n_dof = 19 if self.mode == "upper_body" else 27
        self.dt = dt
        self.dt_ref = dt_ref
        self.dt_scaled_cost = bool(dt_scaled_cost)
        self.reaching_hand = reaching_hand.lower()
        if self.reaching_hand not in ("right", "left", "both", "any"):
            raise ValueError(f"reaching_hand must be 'right', 'left', 'both' or 'any', got {reaching_hand}")

        # Set target weights alpha_right and alpha_left
        if alpha_right is not None:
            self.alpha_right = jnp.float32(alpha_right)
        else:
            self.alpha_right = jnp.float32(0.0 if self.reaching_hand == "left" else 1.0)

        if alpha_left is not None:
            self.alpha_left = jnp.float32(alpha_left)
        else:
            self.alpha_left = jnp.float32(1.0 if self.reaching_hand in ("left", "both") else 0.0)

        # right_hand leaf for legacy tracing
        self.right_hand = jnp.float32(0.0 if self.reaching_hand == "left" else 1.0)

        if q_chest_ref is not None:
            q_ref = jnp.asarray(q_chest_ref, dtype=jnp.float32)
            self.q_chest_ref = q_ref / (jnp.linalg.norm(q_ref) + 1e-8)
        else:
            self.q_chest_ref = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)

        if legs_nominal is not None:
            self.legs_nominal = jnp.asarray(legs_nominal, dtype=jnp.float32)
        else:
            self.legs_nominal = jnp.zeros(8, dtype=jnp.float32)

        # Default human body parameters (meters)
        if body_params is not None:
            self.body_params = jnp.asarray(body_params, dtype=jnp.float32)
        else:
            self.body_params = jnp.array(
                [0.30, 0.40, 0.25, 0.30, 0.30, 0.35, 0.40, 0.40],
                dtype=jnp.float32,
            )

        # Default initial configuration
        if q0 is not None:
            self.q0 = jnp.asarray(q0, dtype=jnp.float32)
        else:
            self.q0 = self._default_nominal_q0()

        # Initial state: [q, dq] in R^(2 * n_dof)
        self.x0 = jnp.concatenate([self.q0, jnp.zeros(self.n_dof, dtype=jnp.float32)])

        # Target definitions. The default targets need the wrists at x0: computed only when a target is missing, with
        # a compiled forward kinematics (op by op it took ~40 ms, once per prediction, even with given targets)
        need_right = target is None
        need_left = target_left is None and self.reaching_hand != "left"
        if need_right or need_left:
            zero3 = jnp.zeros(3, dtype=jnp.float32)
            self.target = self.target_vel = self.target_left = self.target_vel_left = zero3   # complete the pytree
            self._init_shapes()
            rw0, lw0 = _wrists_at(self, self.x0)

        # Right target (self.target):
        if target is not None:
            self.target = jnp.asarray(target, dtype=jnp.float32)
        else:
            self.target = rw0 + jnp.array([0.25, 0.05, 0.15], dtype=jnp.float32)
        self.target_vel = (jnp.zeros(3, dtype=jnp.float32) if target_vel is None
                           else jnp.asarray(target_vel, dtype=jnp.float32))

        # Left target (self.target_left):
        if target_left is not None:
            self.target_left = jnp.asarray(target_left, dtype=jnp.float32)
        elif self.reaching_hand == "left":
            self.target_left = self.target
        else:
            self.target_left = lw0 + jnp.array([0.25, -0.05, 0.15], dtype=jnp.float32)

        if target_vel_left is not None:
            self.target_vel_left = jnp.asarray(target_vel_left, dtype=jnp.float32)
        elif self.reaching_hand == "left":
            self.target_vel_left = self.target_vel
        else:
            self.target_vel_left = jnp.zeros(3, dtype=jnp.float32)

        self._init_shapes()

    def _init_shapes(self):
        Env.__init__(
            self,
            state_shape=(2 * self.n_dof,),
            action_shape=(self.n_dof,),
            observation_shape=(2 * self.n_dof,),
            # motor noise channels: [signal-dependent q, qd | additive q, qd] (see _dynamics)
            state_noise_shape=(4 * self.n_dof,),
            obs_noise_shape=(2 * self.n_dof,),
        )

    def tree_flatten(self):
        leaves = tuple(getattr(self, name) for name in _ENV_LEAVES)
        aux = (self.mode, self.root_joint, self.reaching_hand, self.dt_scaled_cost)
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        env = object.__new__(cls)
        env.mode, env.root_joint, env.reaching_hand, env.dt_scaled_cost = aux
        env.n_dof = 19 if env.mode == "upper_body" else 27
        for name, value in zip(_ENV_LEAVES, leaves):
            setattr(env, name, value)
        env._init_shapes()
        return env

    def passive_arm_slice(self) -> slice:
        return slice(13, 17) if self.reaching_hand == "right" else slice(9, 13)

    def _by_hand(self, right, left):
        """right if the reaching hand is the right one, else left (traced selection with reaching_hand "any")."""
        if self.reaching_hand == "any":
            return self.right_hand * right + (1.0 - self.right_hand) * left
        return right if self.reaching_hand == "right" else left

    def action_weights(self, params: HumanKinematicParams) -> jnp.ndarray:
        """Effort weight of each DOF (n_dof,): the group weights w_act_* of params.
        Blends between reach and passive arm weights according to alpha_right / alpha_left."""
        ones = lambda k: jnp.ones(k, dtype=jnp.float32)
        w_right = self.alpha_right * params.w_act_reach_arm + (1.0 - self.alpha_right) * params.w_act_passive_arm
        w_left = self.alpha_left * params.w_act_reach_arm + (1.0 - self.alpha_left) * params.w_act_passive_arm
        groups = [params.w_act_pelvis * ones(3), params.w_act_trunk * ones(3), params.w_act_spine * ones(3),
                  w_right * ones(4), w_left * ones(4), params.w_act_head * ones(2)]
        if self.mode == "full_body":
            groups.append(params.w_act_legs * ones(8))
        return jnp.concatenate(groups)



    def _default_nominal_q0(self) -> jnp.ndarray:
        """Constructs a natural upright human resting posture."""
        q = jnp.zeros(self.n_dof, dtype=jnp.float32)
        if self.root_joint == "pelvis":
            # Pelvis position at height z ≈ 0.6 m (pelvis is ~40 cm below the chest at z≈1.0 m)
            q = q.at[0:3].set(jnp.array([0.0, 0.0, 0.6], dtype=jnp.float32))
        else:
            # Legacy chest-as-root: chest at z = 1.0 m
            q = q.at[0:3].set(jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32))

        # Natural arm resting posture (slight forward pitch and elbow flexion)
        # Right arm: q[9:13] = (shoulder rot z, rot x, rot y, elbow rot z)
        q = q.at[9].set(0.15)    # slight shoulder abduction
        q = q.at[10].set(-0.25)  # slight shoulder flexion
        q = q.at[12].set(-0.45)  # slight elbow flexion

        # Left arm: q[13:17]
        q = q.at[13].set(-0.15)
        q = q.at[14].set(-0.25)
        q = q.at[16].set(-0.45)

        return q

    # --------------------------------------------------------------------------
    # Full Kinematics & Keypoint Extraction
    # --------------------------------------------------------------------------

    def build_q28(self, q_dof: jnp.ndarray) -> jnp.ndarray:
        """Expands the active DOF vector into the full 28-DOF configuration expected by hkm.fk.

        Args:
            q_dof: Active DOF vector (19 DOFs for upper_body, 27 for full_body).
                q_dof[0:3] is pelvis position when self.root_joint="pelvis",
                or chest position when self.root_joint="chest".

        Returns:
            q28: Full 28-DOF configuration vector (always with chest position at [0:3]).
        """
        chest_rotvec = q_dof[3:6]
        rotvec_quat = quat_from_rotvec(chest_rotvec)
        chest_quat = hkm.quat_multiply(self.q_chest_ref, rotvec_quat)
        shoulder_rotx = q_dof[6:7]
        hip_rot = q_dof[7:9]
        rarm = q_dof[9:13]
        larm = q_dof[13:17]

        if self.root_joint == "pelvis":
            # q[0:3] is pelvis position. Chest is above the pelvis along the chest "up" axis.
            # From trunk_fk: pelvis_pos = chest_pos + R_chest @ [0, 0, -chest_hip_distance]
            #             => chest_pos  = pelvis_pos + R_chest @ [0, 0, +chest_hip_distance]
            #             => chest_pos  = pelvis_pos + chest_hip_distance * R_chest[:, 2]
            # R_chest[:, 2] is the third column of the chest rotation matrix (the "z" axis of the chest frame).
            pelvis_pos = q_dof[0:3]
            chest_hip_distance = self.body_params[1]
            R_chest = hkm.quat_to_rotmat(chest_quat)
            chest_z_in_world = R_chest[:, 2]  # chest "up" direction in world frame
            chest_pos = pelvis_pos + chest_hip_distance * chest_z_in_world
        else:
            # Legacy: q[0:3] is chest position directly
            chest_pos = q_dof[0:3]

        if self.mode == "upper_body":
            rleg = self.legs_nominal[0:4]
            lleg = self.legs_nominal[4:8]
            head = q_dof[17:19]
        else:
            rleg = q_dof[19:23]
            lleg = q_dof[23:27]
            head = q_dof[17:19]

        return jnp.concatenate([
            chest_pos,      # [0:3]
            chest_quat,     # [3:7]
            shoulder_rotx,  # [7]
            hip_rot,        # [8:10]
            rarm,           # [10:14]
            larm,           # [14:18]
            rleg,           # [18:22]
            lleg,           # [22:26]
            head,           # [26:28]
        ])


    def all_keypoints(self, state: jnp.ndarray) -> jnp.ndarray:
        """Calculates 3D Cartesian coordinates of all 13 canonical keypoints.

        Args:
            state: Full state vector in R^(2 * n_dof) or configuration in R^n_dof.

        Returns:
            (13, 3) array of keypoints ordered as hkm.KEYPOINT_NAMES.
        """
        q_dof = state[: self.n_dof]
        q28 = self.build_q28(q_dof)
        return hkm.fk(q28, self.body_params)

    def extended_keypoints(self, state: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Calculates complete 15-keypoint representation including Chest and Pelvis centers."""
        kpts = self.all_keypoints(state)
        q_dof = state[: self.n_dof]
        q28 = self.build_q28(q_dof)
        chest_pos = q28[0:3]  # Always the actual chest position (differs from q_dof[0:3] in pelvis-root mode)
        pelvis_pos = 0.5 * (kpts[KP_LEFT_HIP] + kpts[KP_RIGHT_HIP])

        result = {name: kpts[i] for i, name in enumerate(KEYPOINT_NAMES)}
        result["chest"] = chest_pos
        result["pelvis"] = pelvis_pos
        return result

    # --------------------------------------------------------------------------
    # Task-Space Outputs (Matching Predictor Conventions)
    # --------------------------------------------------------------------------

    def wrist_right(self, state: jnp.ndarray) -> jnp.ndarray:
        """Returns 3D Cartesian position of the right hand/wrist."""
        return self.all_keypoints(state)[KP_RIGHT_WRIST]

    def wrist_left(self, state: jnp.ndarray) -> jnp.ndarray:
        """Returns 3D Cartesian position of the left hand/wrist."""
        return self.all_keypoints(state)[KP_LEFT_WRIST]

    def e(self, state: jnp.ndarray) -> jnp.ndarray:
        """Returns 3D Cartesian position of the reaching hand/wrist."""
        kpts = self.all_keypoints(state)
        return self._by_hand(kpts[KP_RIGHT_WRIST], kpts[KP_LEFT_WRIST])

    def elbow(self, state: jnp.ndarray) -> jnp.ndarray:
        """Returns 3D Cartesian position of the reaching elbow."""
        kpts = self.all_keypoints(state)
        return self._by_hand(kpts[KP_RIGHT_ELBOW], kpts[KP_LEFT_ELBOW])

    def passive_wrist(self, state: jnp.ndarray) -> jnp.ndarray:
        """Returns 3D Cartesian position of the other (non-reaching) wrist."""
        kpts = self.all_keypoints(state)
        return self._by_hand(kpts[KP_LEFT_WRIST], kpts[KP_RIGHT_WRIST])

    def head(self, state: jnp.ndarray) -> jnp.ndarray:
        """Returns 3D Cartesian position of the head."""
        return self.all_keypoints(state)[KP_HEAD]

    def pelvis(self, state: jnp.ndarray) -> jnp.ndarray:
        """Returns 3D Cartesian position of the pelvis center."""
        kpts = self.all_keypoints(state)
        return 0.5 * (kpts[KP_LEFT_HIP] + kpts[KP_RIGHT_HIP])

    def chest(self, state: jnp.ndarray) -> jnp.ndarray:
        """Returns 3D Cartesian position of the chest center (always correct regardless of root_joint)."""
        q_dof = state[: self.n_dof]
        return self.build_q28(q_dof)[0:3]

    # Task Jacobians
    def gamma(self, state: jnp.ndarray) -> jnp.ndarray:
        """Task-space Jacobian of the reaching hand wrt state: J in R^(3 x 2*n_dof)."""
        return jacobian(self.e)(state)

    def gamma_elbow(self, state: jnp.ndarray) -> jnp.ndarray:
        """Task-space Jacobian of the elbow wrt state: J in R^(3 x 2*n_dof)."""
        return jacobian(self.elbow)(state)

    def gamma_keypoint(self, state: jnp.ndarray, kp_idx: int) -> jnp.ndarray:
        """Task-space Jacobian of any keypoint index wrt state: J in R^(3 x 2*n_dof)."""
        return jacobian(lambda s: self.all_keypoints(s)[kp_idx])(state)

    def wrist_velocity(self, state: jnp.ndarray) -> jnp.ndarray:
        """Analytical velocity of the reaching hand via JVP."""
        q = state[: self.n_dof]
        qd = state[self.n_dof :]
        return jax.jvp(lambda q_: self.e(q_), (q,), (qd,))[1]

    def wrist_vel_right(self, state: jnp.ndarray) -> jnp.ndarray:
        """Analytical velocity of the right wrist via JVP."""
        q = state[: self.n_dof]
        qd = state[self.n_dof :]
        return jax.jvp(lambda q_: self.wrist_right(q_), (q,), (qd,))[1]

    def wrist_vel_left(self, state: jnp.ndarray) -> jnp.ndarray:
        """Analytical velocity of the left wrist via JVP."""
        q = state[: self.n_dof]
        qd = state[self.n_dof :]
        return jax.jvp(lambda q_: self.wrist_left(q_), (q,), (qd,))[1]


    # --------------------------------------------------------------------------
    # NIOC Environment Interface
    # --------------------------------------------------------------------------

    def _dynamics(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        noise: jnp.ndarray,
        params: HumanKinematicParams,
    ) -> jnp.ndarray:
        """Exact ZOH discretization of qdd = u - b qd + sigma_m u w(t) + sigma_add w'(t) (prophet_ioc.envs.zoh).

        Deterministic part: qd' = e qd + a1 u, q' = q + a1 qd + a2 u (b = params.damping, smooth at b = 0).

        Noise (state_noise_shape = 4 n): white-noise accelerations integrated over the step, mapped through the
        Cholesky factor (l11, l21, l22) of M(dt) = [[dt^3/3, dt^2/2], [dt^2/2, dt]]:
            delta q_j  = s_j l11 n1_j,   delta qd_j = s_j (l21 n1_j + l22 n2_j)
        with s = sigma_m u (signal-dependent, channels n1 = noise[:n], n2 = noise[n:2n]) plus the same with
        s = sigma_add (additive, channels noise[2n:3n], noise[3n:]). The motor noise Jacobian V = df/dv at v = 0
        therefore gives V V^T = (sigma_m^2 u_j^2 + sigma_add^2) M(dt) per joint: exactly the covariance of the
        continuous-time noise over the step (zoh module docstring), with a non-singular position block. Its standard deviation is proportional
        to |u| (signal-dependent noise, Harris & Wolpert 1998; the sign of u does not matter for the covariance, and
        using u rather than |u| keeps V differentiable for the gLQR terms Cu = dV/du), and it scales with sqrt(dt)
        like white noise, so sigma_m does not depend on the time grid; sigma_add covers u ~ 0. M is the b = 0
        covariance, used for any damping (small-b approximation).
        """
        n = self.n_dof
        q = state[:n]
        qd = state[n:]
        e, a1, a2 = zoh_coefficients(self.dt, params.damping)
        q_next = q + a1 * qd + a2 * action
        qd_next = e * qd + a1 * action

        l11, l21, l22 = zoh_noise_chol(self.dt)
        s_sig = params.motor_noise * action
        s_add = params.motor_noise_add
        dq = l11 * (s_sig * noise[:n] + s_add * noise[2 * n:3 * n])
        dqd = l21 * (s_sig * noise[:n] + s_add * noise[2 * n:3 * n]) + l22 * (s_sig * noise[n:2 * n]
                                                                              + s_add * noise[3 * n:])
        return jnp.concatenate([q_next + dq, qd_next + dqd])

    def residual_covariance(self, params: HumanKinematicParams) -> jnp.ndarray:
        """Covariance (2n, 2n) of the likelihood-only residual noise: a white-noise acceleration of intensity
        params.residual_noise on every joint, discretized like motor_noise_add (residual_noise^2 M(dt) per joint)."""
        return params.residual_noise ** 2 * zoh_process_cov(self.dt, self.n_dof)

    def _observation(
        self,
        state: jnp.ndarray,
        noise: jnp.ndarray,
        params: HumanKinematicParams,
    ) -> jnp.ndarray:
        """Observation model of the partially observed (belief) model: the full state with additive Gaussian noise of
        std sqrt(dt) sigma_o."""
        return state + jnp.sqrt(self.dt) * params.obs_noise * noise

    def joint_limit_penalty(self, q: jnp.ndarray, params: HumanKinematicParams) -> jnp.ndarray:
        """0.5 * sum_j [max(0, q_j - q_max_j)^2 + max(0, q_min_j - q_j)^2] over the bounded DOFs (joint_limits), plus
        0.5 * max(0, |w| - chest_rot_limit)^2 for the chest rotation vector w = q[3:6]. Zero inside the limits,
        quadratic outside: C1 with a PSD (piecewise constant, diagonal for the joints) Hessian, as iLQG needs."""
        lo, hi = joint_limits(self.mode)
        over = jnp.maximum(q - hi, 0.0) + jnp.maximum(lo - q, 0.0)   # at most one of the two is non-zero
        rot = jnp.sqrt(jnp.sum(q[3:6] ** 2) + 1e-12)                  # smooth norm: finite gradient at w = 0
        return 0.5 * jnp.sum(over ** 2) + 0.5 * jnp.maximum(rot - params.chest_rot_limit, 0.0) ** 2

    def _cost(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: HumanKinematicParams,
    ) -> jnp.ndarray:
        """Running cost, scaled by dt / dt_ref when dt_scaled_cost is set:
            effort        0.5 * sum_j w_{g(j)} u_j^2 (action_weights: one weight per joint group)
            pelvis        0.5 * base_disp_cost * |q[0:3] - q0[0:3]|^2 (displacement from the start of the horizon;
                          base_disp_cost = 0 when the term is switched off, e.g. walking tasks)
            velocity      0.5 * max(running_vel_cost, velocity_floor) * |qd|^2
            joint limits  w_lim * joint_limit_penalty(q) (hand-tuned, 0 = off)
            target        0.5 * running_target_cost * |wrist - target|^2 (Gauss-Newton quadratized, optional)
        There is no posture term: anchoring the joints to the handover posture biased the predicted reach."""
        q = state[: self.n_dof]
        qd = state[self.n_dof :]
        cost_effort = 0.5 * jnp.sum(self.action_weights(params) * action ** 2)
        cost_base = 0.5 * params.base_disp_cost * jnp.sum((q[0:3] - self.q0[0:3]) ** 2)
        cost_vel = 0.5 * jnp.maximum(params.running_vel_cost, params.velocity_floor) * jnp.sum(qd ** 2)
        cost_lim = params.w_lim * self.joint_limit_penalty(q, params)
        cost_target_r = self.alpha_right * gauss_newton_sq(lambda x: self.wrist_right(x) - self.target, state)
        cost_target_l = self.alpha_left * gauss_newton_sq(lambda x: self.wrist_left(x) - self.target_left, state)
        cost_target = 0.5 * params.running_target_cost * (cost_target_r + cost_target_l)
        cost = cost_effort + cost_base + cost_vel + cost_lim + cost_target
        if self.dt_scaled_cost:
            cost = cost * (self.dt / self.dt_ref)
        return cost

    def _final_cost(
        self,
        state: jnp.ndarray,
        params: HumanKinematicParams,
    ) -> jnp.ndarray:
        """Terminal cost: alpha_R (|wrist_R - target_R|^2 + velocity_cost * |wrist_vel_R - target_vel_R|^2) +
        alpha_L (|wrist_L - target_L|^2 + velocity_cost * |wrist_vel_L - target_vel_L|^2)."""
        err_r = gauss_newton_sq(lambda x: self.wrist_right(x) - self.target, state)
        vel_r = gauss_newton_sq(lambda x: self.wrist_vel_right(x) - self.target_vel, state)
        term_r = self.alpha_right * (err_r + params.velocity_cost * vel_r)

        err_l = gauss_newton_sq(lambda x: self.wrist_left(x) - self.target_left, state)
        vel_l = gauss_newton_sq(lambda x: self.wrist_vel_left(x) - self.target_vel_left, state)
        term_l = self.alpha_left * (err_l + params.velocity_cost * vel_l)

        return term_r + term_l


    # Exact quadratization of the cost from its structure (spec.make_lqr_approx uses it instead of jacfwd(grad)):
    # False falls back to automatic differentiation (reference for tests).
    fast_quadratization = True
    # The motor noise of _dynamics is state-independent with covariance motor_noise^2 u_j^2 M(dt) on (q_j, qd_j) (+
    # additive): ilqr_unrolled.backward uses the closed-form noise terms of glqr.backward_joint_signal_noise.
    # False: the generic backward pass on the noise Jacobians (reference for tests).
    joint_signal_noise = True

    def quadratize_cost(self, X: jnp.ndarray, U: jnp.ndarray, params: HumanKinematicParams):
        """(Q, q, P, R, r) of the running cost at (X[:-1], U) and (Qf, qf) of the final cost at X[-1]: the values (and
        the derivatives, for the offline fit) of make_lqr_approx's jacfwd(grad) of _cost / _final_cost, computed from
        the structure of the cost instead of differentiating the gradient once per state dimension:
            effort, pelvis displacement, joint velocity   constant diagonal Hessians
            joint limits                                   the joints only (no kinematics): exact autodiff Hessian
            wrist targets (running, final position, final velocity)
                                                           Gauss-Newton as gauss_newton_sq: gradient 2 J0^T r_lin and
                                                           Hessian 2 J0^T J0 with the residual Jacobian J0 at
                                                           stop_gradient(x) (3 x 19 for the positions: one forward-mode
                                                           Jacobian of the forward kinematics for both wrists)
        About an order of magnitude cheaper per step (the forward kinematics is differentiated 19 times instead of
        ~96 times); identical up to floating-point rounding (tests/test_human_kinematic_reaching.py)."""
        n = self.n_dof
        sg = jax.lax.stop_gradient
        scale = (self.dt / self.dt_ref) if self.dt_scaled_cost else 1.0
        w_u = self.action_weights(params)
        w_qd = jnp.maximum(params.running_vel_cost, params.velocity_floor)
        w_r = params.running_target_cost
        alphas = (self.alpha_right, self.alpha_left)
        targets = (self.target, self.target_left)
        lim = lambda q: self.joint_limit_penalty(q, params)

        def wrists(q):
            kp = self.all_keypoints(q)
            return jnp.stack([kp[KP_RIGHT_WRIST], kp[KP_LEFT_WRIST]])        # (2, 3)

        def gn(r0, J, dx):
            """Gradient and Hessian of |r|^2 for r = stop_gradient(r0) + J dx (gauss_newton_sq)."""
            r = sg(r0) + J @ dx
            return 2.0 * J.T @ r, 2.0 * J.T @ J

        def running(x, u):
            q, qd = x[:n], x[n:]
            q0s = sg(q)
            w0 = wrists(q0s)
            J0 = jax.jacfwd(wrists)(q0s)                                       # (2, 3, n)
            gq = params.w_lim * jax.grad(lim)(q)
            Hq = params.w_lim * jax.hessian(lim)(q)
            gq = gq.at[0:3].add(params.base_disp_cost * (q[0:3] - self.q0[0:3]))
            Hq = Hq.at[0:3, 0:3].add(params.base_disp_cost * jnp.eye(3, dtype=Hq.dtype))
            for k in range(2):
                g_k, H_k = gn(w0[k] - targets[k], J0[k], q - q0s)
                gq = gq + 0.5 * w_r * alphas[k] * g_k
                Hq = Hq + 0.5 * w_r * alphas[k] * H_k
            qx = scale * jnp.concatenate([gq, w_qd * qd])
            Qx = scale * jnp.block([[Hq, jnp.zeros((n, n), Hq.dtype)],
                                    [jnp.zeros((n, n), Hq.dtype), w_qd * jnp.eye(n, dtype=Hq.dtype)]])
            r = scale * w_u * u
            R = scale * jnp.diag(w_u)
            P = jnp.zeros((u.shape[0], x.shape[0]), dtype=Qx.dtype)
            return Qx, qx, P, R, r

        def final(x):
            x0 = sg(x)
            vel = lambda s: jax.jvp(wrists, (s[:n],), (s[n:],))[1]             # (2, 3) wrist velocities
            w0, v0 = wrists(x0[:n]), vel(x0)
            Jp = jax.jacfwd(lambda s: wrists(s[:n]))(x0)                       # (2, 3, 2n)
            Jv = jax.jacfwd(vel)(x0)                                           # (2, 3, 2n)
            vt = (self.target_vel, self.target_vel_left)
            gx = jnp.zeros(2 * n, dtype=x.dtype)
            Hx = jnp.zeros((2 * n, 2 * n), dtype=x.dtype)
            for k in range(2):
                g_p, H_p = gn(w0[k] - targets[k], Jp[k], x - x0)
                g_v, H_v = gn(v0[k] - vt[k], Jv[k], x - x0)
                gx = gx + alphas[k] * (g_p + params.velocity_cost * g_v)
                Hx = Hx + alphas[k] * (H_p + params.velocity_cost * H_v)
            return Hx, gx

        Q, q, P, R, r = jax.vmap(running)(X[:-1], U)
        Qf, qf = final(X[-1])
        return Q, q, P, R, r, Qf, qf

    def _reset(
        self,
        noise: Optional[jnp.ndarray],
        params: HumanKinematicParams,
    ) -> jnp.ndarray:
        """Resets the environment to nominal resting state."""
        return self.x0

    @staticmethod
    def get_params_type() -> type:
        return HumanKinematicParams.get_params_type()

    @staticmethod
    def get_params_bounds() -> Tuple[HumanKinematicParams, HumanKinematicParams]:
        return HumanKinematicParams.get_params_bounds()


@jax.jit
def _wrists_at(env: "HumanKinematicReaching", x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Right and left wrist positions at state x (compiled once per static configuration of the environment)."""
    return env.wrist_right(x), env.wrist_left(x)


def stack_envs(envs: List[HumanKinematicReaching]) -> HumanKinematicReaching:
    """Stacks environments with the same static configuration into one batched environment (leading axis = trial)."""
    return jax.tree.map(lambda *leaves: jnp.stack([jnp.asarray(leaf, dtype=jnp.float32) for leaf in leaves]), *envs)
