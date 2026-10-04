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
- Unconstrained 3D rotation vector representation for trunk/chest orientation with
  singularity-free analytical quaternion mapping (no gradient NaNs).
- Rigid link kinematics strictly preserving human anatomical bone lengths.
- Gauss-Newton quadratization of terminal reaching costs for stable iLQG convergence.
"""

from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import jacobian

import human_kinematic_model_jax as hkm
from prophet_ioc.envs.base import Env


# Attributes of HumanKinematicReaching that are pytree leaves (traced under jit / batched by vmap); the others are static.
_ENV_LEAVES = ("dt", "dt_ref", "w_target", "posture_cost", "base_disp_cost", "q_chest_ref", "legs_nominal",
               "body_params", "q0", "q_posture_ref", "x0", "target", "target_vel", "w_action", "right_hand")


class HumanKinematicParams(NamedTuple):
    action_cost: float = 1e-4
    velocity_cost: float = 1e-2
    posture_cost: float = 1e-3
    w_target: float = 100.0
    motor_noise: float = 0.1
    obs_noise: float = 1.0
    running_vel_cost: float = 0.0
    base_disp_cost: float = 10.0  # Weight penalizing 3D displacement of root base link (chest_pos)
    # Effort weights of the joint groups (relative to the reaching arm, fixed at 1; pelvis translation fixed at 20).
    # The defaults are the former hard-coded values.
    w_act_trunk: float = 3.0         # chest rotation vector q[3:6]
    w_act_spine: float = 3.0         # shoulder rot x and hip rot z/x, q[6:9]
    w_act_passive_arm: float = 3.0   # non-reaching arm
    w_act_head: float = 2.0          # head rot x/y, q[17:19]
    running_target_cost: float = 0.0  # Weight of the wrist-to-target distance at every step (not only at T)
    motor_noise_add: float = 0.0     # Additive (signal-independent) motor noise on the joint velocities
    # Likelihood-only additive std of the joint-velocity transitions (rad/s, m/s for the pelvis): absorbs model
    # mismatch in the IOC likelihood (prophet_ioc.infer.multi_env). Not part of the dynamics, so the controller does not
    # plan against it, unlike motor_noise.
    residual_noise: float = 0.0

    @staticmethod
    def get_params_type() -> type:
        """Return the parameter record type used by this environment."""
        return HumanKinematicParams

    @staticmethod
    def get_params_bounds() -> Tuple["HumanKinematicParams", "HumanKinematicParams"]:
        """Return useful lower and upper bounds for parameter fitting."""
        lo = HumanKinematicParams(
            action_cost=1e-6, velocity_cost=1e-4, posture_cost=1e-5, w_target=1.0, motor_noise=1e-2, obs_noise=0.1,
            running_vel_cost=1e-5, base_disp_cost=0.1, w_act_trunk=0.1, w_act_spine=0.1, w_act_passive_arm=0.1,
            w_act_head=0.1, running_target_cost=1e-3, motor_noise_add=1e-3, residual_noise=1e-3,
        )
        hi = HumanKinematicParams(
            action_cost=1e-2, velocity_cost=1.0, posture_cost=1e-1, w_target=1e3, motor_noise=1.0, obs_noise=10.0,
            running_vel_cost=1.0, base_disp_cost=100.0, w_act_trunk=100.0, w_act_spine=100.0,
            w_act_passive_arm=100.0, w_act_head=100.0, running_target_cost=1e3, motor_noise_add=10.0,
            residual_noise=10.0,
        )
        return lo, hi


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
    """Computes |residual(x)|^2 with exact gradient and positive semi-definite Gauss-Newton Hessian (2 J^T J)."""
    x0 = jax.lax.stop_gradient(x)
    r0, jvp = jax.linearize(residual_fn, x0)
    r = jax.lax.stop_gradient(r0) + jvp(x - x0)
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

    def __init__(
        self,
        mode: str = "upper_body",
        dt: float = 0.02,
        target: Optional[Union[jnp.ndarray, Tuple[float, float, float]]] = None,
        target_vel: Optional[Union[jnp.ndarray, Tuple[float, float, float]]] = None,
        q0: Optional[jnp.ndarray] = None,
        body_params: Optional[jnp.ndarray] = None,
        reaching_hand: str = "right",
        w_target: float = 100.0,
        posture_cost: float = 1e-3,
        base_disp_cost: float = 10.0,
        q_chest_ref: Optional[jnp.ndarray] = None,
        legs_nominal: Optional[jnp.ndarray] = None,
        root_joint: str = "pelvis",
        q_posture_ref: Optional[jnp.ndarray] = None,
        dt_scaled_cost: bool = False,
        dt_ref: float = 0.05,
    ):
        """Initializes the human kinematic reaching environment.

        Args:
            mode: "upper_body" (19 active DOFs) or "full_body" (27 active DOFs).
            dt: Sampling time step in seconds (default: 0.02 s = 50 Hz).
            target: [x, y, z] Cartesian target position for the reaching wrist.
            target_vel: [vx, vy, vz] wrist velocity at the end of the horizon (terminal velocity cost; default zero,
                i.e. at rest on the target; non-zero for an intermediate target on the way to a farther goal).
            q0: Initial configuration vector (19 DOFs for upper_body, 27 for full_body).
                When root_joint="pelvis", q[0:3] must be the pelvis 3D position.
                When root_joint="chest", q[0:3] must be the chest 3D position (legacy).
            body_params: 8 body segment parameters in meters:
                [shoulder_dist, chest_hip_dist, hip_dist, upper_arm, lower_arm, thigh, shank, head_dist].
            reaching_hand: "right" or "left" hand to execute the reaching task, or "any": the hand is the traced leaf
                right_hand (1 right, 0 left), so that environments of both hands can be batched together.
            w_target: Weight penalty on terminal target reaching error.
            posture_cost: Weight penalty on deviating from nominal resting posture.
            base_disp_cost: Weight penalty on 3D displacement of the root base link.
                Anchors the pelvis (root_joint="pelvis") or chest (root_joint="chest").
            q_chest_ref: Reference chest quaternion [x, y, z, w]. If provided, chest_quat = q_chest_ref * rotvec_quat.
            legs_nominal: Nominal leg joint angles (8,) for upper_body mode.
            root_joint: "pelvis" (default) or "chest".
                - "pelvis": q[0:3] stores pelvis position; build_q28 computes chest as
                  pelvis + chest_hip_distance * R_chest[:,2], allowing trunk flexion without
                  displacing the base link anchor.
                - "chest": q[0:3] stores chest position (legacy behaviour).
            q_posture_ref: Reference configuration of the posture and base displacement costs (same layout as q0).
                Defaults to q0, the configuration at the start of the horizon.
            dt_scaled_cost: If True, the running cost is multiplied by dt / dt_ref, so that the cost weights mean the
                same on any time grid (a sum over steps approximating an integral); the weights keep their meaning at
                dt = dt_ref. If False (legacy), the running cost is summed per step and the weights depend on dt.
            dt_ref: Reference time step of dt_scaled_cost, in seconds.
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
        if self.reaching_hand not in ("right", "left", "any"):
            raise ValueError(f"reaching_hand must be 'right', 'left' or 'any', got {reaching_hand}")
        self.right_hand = jnp.float32(0.0 if self.reaching_hand == "left" else 1.0)
        self.w_target = w_target
        self.posture_cost = posture_cost
        self.base_disp_cost = base_disp_cost

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
        self.q_posture_ref = self.q0 if q_posture_ref is None else jnp.asarray(q_posture_ref, dtype=jnp.float32)

        # Initial state: [q, dq] in R^(2 * n_dof)
        self.x0 = jnp.concatenate([self.q0, jnp.zeros(self.n_dof, dtype=jnp.float32)])

        # Target definition
        rw0 = self.e(self.x0)
        if target is not None:
            self.target = jnp.asarray(target, dtype=jnp.float32)
        else:
            # Default target: reach forward +x by 25 cm, +y by 5 cm, +z by 15 cm
            self.target = rw0 + jnp.array([0.25, 0.05, 0.15], dtype=jnp.float32)
        self.target_vel = (jnp.zeros(3, dtype=jnp.float32) if target_vel is None
                           else jnp.asarray(target_vel, dtype=jnp.float32))

        # Fixed part of the effort weighting across body segments: base link 3D translation is penalized heavily to
        # prevent floating body displacement, the reaching arm has weight 1. The trunk, spine, passive arm and head
        # weights are parameters (HumanKinematicParams.w_act_*), see action_weights().
        w_act = jnp.ones(self.n_dof, dtype=jnp.float32)
        w_act = w_act.at[0:3].set(20.0)   # Pelvis / chest 3D translation (base link) heavily penalized
        if self.mode == "full_body":
            w_act = w_act.at[19:27].set(8.0)  # Legs penalized more (standing stability)
        self.w_action = w_act

        self._init_shapes()

    def _init_shapes(self):
        Env.__init__(
            self,
            state_shape=(2 * self.n_dof,),
            action_shape=(self.n_dof,),
            observation_shape=(2 * self.n_dof,),
            state_noise_shape=(2 * self.n_dof,),
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
        """Effort weight of each DOF: fixed base/arm/leg weights plus the parametrized joint groups."""
        w = self.w_action
        w = w.at[3:6].set(getattr(params, "w_act_trunk", 3.0))
        w = w.at[6:9].set(getattr(params, "w_act_spine", 3.0))
        w_passive = getattr(params, "w_act_passive_arm", 3.0)
        if self.reaching_hand == "any":   # right arm q[9:13], left arm q[13:17]; the reaching arm keeps w_action
            w = w.at[13:17].set(self._by_hand(w_passive, w[13:17]))
            w = w.at[9:13].set(self._by_hand(w[9:13], w_passive))
        else:
            w = w.at[self.passive_arm_slice()].set(w_passive)
        w = w.at[17:19].set(getattr(params, "w_act_head", 2.0))
        return w

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
        """Discrete-time joint-space integrator dynamics with joint damping and motor noise."""
        q = state[: self.n_dof]
        qd = state[self.n_dof :]

        # Natural biological joint damping (viscous friction)
        damping = 0.20
        q_next = q + self.dt * qd
        qd_next = (1.0 - damping * self.dt) * qd + self.dt * action

        next_state = jnp.concatenate([q_next, qd_next])

        # Signal-dependent motor noise on joint velocities (noise[n:]), plus an optional additive part that uses the
        # otherwise unused noise[:n] channels
        motor_noise = jnp.sqrt(self.dt) * (
            params.motor_noise * action * noise[self.n_dof :]
            + getattr(params, "motor_noise_add", 0.0) * noise[: self.n_dof]
        )
        return next_state.at[self.n_dof :].add(motor_noise)

    def _observation(
        self,
        state: jnp.ndarray,
        noise: jnp.ndarray,
        params: HumanKinematicParams,
    ) -> jnp.ndarray:
        """Observation model with additive sensory Gaussian noise."""
        return state + jnp.sqrt(self.dt) * params.obs_noise * noise

    def _cost(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        params: HumanKinematicParams,
    ) -> jnp.ndarray:
        """Running cost: joint effort + base link displacement + internal posture & velocity regularization
        (+ optional running target cost), scaled by dt / dt_ref when dt_scaled_cost is set."""
        q = state[: self.n_dof]
        qd = state[self.n_dof :]

        # 1. Base link 3D displacement penalty (anchoring body reference frame)
        w_base = getattr(params, "base_disp_cost", getattr(self, "base_disp_cost", 10.0))
        cost_base = 0.5 * w_base * jnp.sum((q[0:3] - self.q_posture_ref[0:3]) ** 2)

        # 2. Internal posture penalty (trunk orientation / flexion & limb joints)
        p_posture = getattr(params, "posture_cost", self.posture_cost)
        cost_posture = 0.5 * p_posture * jnp.sum((q[3:] - self.q_posture_ref[3:]) ** 2)

        # 3. Running velocity damping
        r_vel = getattr(params, "running_vel_cost", 0.0)
        cost_vel = 0.5 * jnp.where(r_vel > 0.0, r_vel, 1e-4) * jnp.sum(qd ** 2)

        # 4. Joint action / effort cost
        cost_effort = 0.5 * params.action_cost * jnp.sum(self.action_weights(params) * action**2)

        cost = cost_effort + cost_base + cost_posture + cost_vel

        # 5. Running wrist-to-target cost (Gauss-Newton quadratized, like the terminal cost)
        w_run_target = getattr(params, "running_target_cost", 0.0)
        cost = cost + 0.5 * w_run_target * gauss_newton_sq(lambda x: self.e(x) - self.target, state)

        if self.dt_scaled_cost:
            cost = cost * (self.dt / self.dt_ref)
        return cost

    def _final_cost(
        self,
        state: jnp.ndarray,
        params: HumanKinematicParams,
    ) -> jnp.ndarray:
        """Terminal cost: target reaching error + terminal wrist velocity error (to target_vel, zero by default) via
        Gauss-Newton."""
        err = gauss_newton_sq(lambda x: self.e(x) - self.target, state)
        vel = gauss_newton_sq(lambda x: self.wrist_velocity(x) - self.target_vel, state)
        w_t = getattr(params, "w_target", self.w_target)
        return w_t * err + params.velocity_cost * vel

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


def stack_envs(envs: List[HumanKinematicReaching]) -> HumanKinematicReaching:
    """Stacks environments with the same static configuration into one batched environment (leading axis = trial)."""
    return jax.tree.map(lambda *leaves: jnp.stack([jnp.asarray(leaf, dtype=jnp.float32) for leaf in leaves]), *envs)
