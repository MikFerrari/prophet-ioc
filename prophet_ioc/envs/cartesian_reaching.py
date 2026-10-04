from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import jacobian

from prophet_ioc.envs.base import Env


class CartesianReachingParams(NamedTuple):
    action_cost: float = 1e-4
    velocity_cost: float = 1e-2
    motor_noise: float = 1e-1
    obs_noise: float = 1.0


class CartesianReaching(Env):
    """2D Cartesian Space Point-Mass Reaching Task.

    State: [px, py, vx, vy] in R^4
    Action: [ux, uy] in R^2 (Cartesian command force in N)
    Dynamics: m * ddp + b * dp = u + w_motor
    """

    def __init__(
        self,
        dt: float = 0.01,
        target: Optional[jnp.ndarray] = None,
        x0: Optional[jnp.ndarray] = None,
        mass: float = 2.5,
        damping: float = 5.0,
        w_target: float = 10.0,
    ):
        self.dt = dt
        self.mass = mass
        self.damping = damping
        self.w_target = w_target
        self.is_cartesian = True

        if target is not None:
            self.target = jnp.asarray(target, dtype=jnp.float32)
        else:
            self.target = jnp.array([0.05, 0.50], dtype=jnp.float32)

        if x0 is not None:
            self.x0 = jnp.asarray(x0, dtype=jnp.float32)
        else:
            self.x0 = jnp.array([0.0, 0.47, 0.0, 0.0], dtype=jnp.float32)

        super().__init__(
            state_shape=(4,),
            action_shape=(2,),
            observation_shape=(4,),
            state_noise_shape=(4,),
            obs_noise_shape=(4,),
        )

    def e(self, state: jnp.ndarray) -> jnp.ndarray:
        return state[:2]

    def gamma(self, state: jnp.ndarray) -> jnp.ndarray:
        return jnp.eye(2, dtype=jnp.float32)

    def edot(self, state: jnp.ndarray) -> jnp.ndarray:
        return state[2:]

    def _dynamics(self, state, action, noise, params):
        vel = state[2:]
        acc = (action - self.damping * vel) / self.mass

        dx = jnp.concatenate([vel, acc])
        f = self.dt * dx

        motor_noise_mat = params.motor_noise * jnp.diag(action)
        w_acc = (motor_noise_mat @ noise[2:]) / self.mass
        w = jnp.sqrt(self.dt) * jnp.concatenate([jnp.zeros(2, dtype=jnp.float32), w_acc])

        return state + f + w

    def _observation(self, state, noise, params):
        return state + self.dt * params.obs_noise * jnp.eye(self.observation_shape[0]) @ noise

    def _cost(self, state, action, params):
        return 0.5 * params.action_cost * jnp.sum(action**2)

    def _final_cost(self, state, params):
        p_err = state[:2] - self.target
        v_err = state[2:]
        return self.w_target * jnp.sum(p_err**2) + params.velocity_cost * jnp.sum(v_err**2)

    def _reset(self, noise, params):
        return self.x0

    @staticmethod
    def get_params_type():
        return CartesianReachingParams

    @staticmethod
    def get_params_bounds():
        lo = CartesianReachingParams(
            action_cost=1e-5, velocity_cost=1e-3, motor_noise=1e-2, obs_noise=1e-1
        )
        hi = CartesianReachingParams(
            action_cost=1e-1, velocity_cost=1e-1, motor_noise=1.0, obs_noise=100.0
        )
        return lo, hi


class CartesianReaching3DParams(NamedTuple):
    action_cost: float = 1e-4
    velocity_cost: float = 1e-2
    motor_noise: float = 1e-1
    obs_noise: float = 1.0


class CartesianReaching3D(Env):
    """3D Cartesian Space Point-Mass Reaching Task.

    Models reaching kinematics directly in 3D task space (x, y, z):
        State : [px, py, pz, vx, vy, vz] in R^6
        Action: [ux, uy, uz] in R^3 (endpoint command force in N)
        Dynamics: m * ddp + b * dp = u + w_motor
    """

    def __init__(
        self,
        dt: float = 0.01,
        target: Optional[jnp.ndarray] = None,
        x0: Optional[jnp.ndarray] = None,
        mass: float = 2.5,
        damping: float = 5.0,
        w_target: float = 10.0,
    ):
        self.dt = dt
        self.mass = mass
        self.damping = damping
        self.w_target = w_target
        self.is_cartesian = True

        if target is not None:
            self.target = jnp.asarray(target, dtype=jnp.float32)
        else:
            self.target = jnp.array([0.40, 0.12, 0.25], dtype=jnp.float32)

        if x0 is not None:
            self.x0 = jnp.asarray(x0, dtype=jnp.float32)
        else:
            self.x0 = jnp.array([0.34, 0.02, 0.18, 0.0, 0.0, 0.0], dtype=jnp.float32)

        super().__init__(
            state_shape=(6,),
            action_shape=(3,),
            observation_shape=(6,),
            state_noise_shape=(6,),
            obs_noise_shape=(6,),
        )

    def e(self, state: jnp.ndarray) -> jnp.ndarray:
        """Hand 3D Cartesian coordinates [x, y, z] in meters."""
        return state[:3]

    def gamma(self, state: jnp.ndarray) -> jnp.ndarray:
        """Task-space 3D Jacobian J = de/dx (3x3 position block)."""
        return jnp.eye(3, dtype=jnp.float32)

    def edot(self, state: jnp.ndarray) -> jnp.ndarray:
        """Task-space 3D velocity [vx, vy, vz]."""
        return state[3:]

    def keypoints(self, state: jnp.ndarray):
        """Shoulder, Elbow, and Hand keypoints in 3D.

        Since this is an endpoint Cartesian point-mass model, internal elbow posture
        is unmodeled and approximated as the midpoint between shoulder base and hand.
        """
        p_shoulder = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
        p_hand = state[:3]
        p_elbow = 0.5 * (p_shoulder + p_hand)
        return p_shoulder, p_elbow, p_hand

    def _dynamics(self, state, action, noise, params):
        vel = state[3:]
        acc = (action - self.damping * vel) / self.mass

        dx = jnp.concatenate([vel, acc])
        f = self.dt * dx

        motor_noise_mat = params.motor_noise * jnp.diag(action)
        w_acc = (motor_noise_mat @ noise[3:]) / self.mass
        w = jnp.sqrt(self.dt) * jnp.concatenate([jnp.zeros(3, dtype=jnp.float32), w_acc])

        return state + f + w

    def _observation(self, state, noise, params):
        return state + self.dt * params.obs_noise * jnp.eye(self.observation_shape[0]) @ noise

    def _cost(self, state, action, params):
        return 0.5 * params.action_cost * jnp.sum(action**2)

    def _final_cost(self, state, params):
        p_err = state[:3] - self.target
        v_err = state[3:]
        return self.w_target * jnp.sum(p_err**2) + params.velocity_cost * jnp.sum(v_err**2)

    def _reset(self, noise, params):
        return self.x0

    @staticmethod
    def get_params_type():
        return CartesianReaching3DParams

    @staticmethod
    def get_params_bounds():
        lo = CartesianReaching3DParams(
            action_cost=1e-5, velocity_cost=1e-3, motor_noise=1e-2, obs_noise=1e-1
        )
        hi = CartesianReaching3DParams(
            action_cost=1e-1, velocity_cost=1e-1, motor_noise=1.0, obs_noise=100.0
        )
        return lo, hi


class CartesianMultiPointReaching3DParams(NamedTuple):
    action_cost: float = 1e-4
    velocity_cost: float = 1e-2
    motor_noise: float = 1e-1
    obs_noise: float = 1.0


class CartesianMultiPointReaching3D(Env):
    """3D Multi-Point Cartesian Reaching Task.

    Models multiple keypoints of the human arm (Elbow and Hand/Wrist) as
    independent, uncoupled Cartesian point-masses in 3D Euclidean space.
    The shoulder is modeled as a fixed base at the origin [0, 0, 0].

    State : [p_e, p_h, v_e, v_h] in R^12
            p_e = (xe, ye, ze) : 3D Elbow position [m]
            p_h = (xh, yh, zh) : 3D Hand/Wrist position [m]
            v_e = (vxe, vye, vze) : 3D Elbow velocity [m/s]
            v_h = (vxh, vyh, vzh) : 3D Hand velocity [m/s]
    Action: [u_e, u_h] in R^6
            u_e = (Fxe, Fye, Fze) : Elbow virtual control force [N]
            u_h = (Fxh, Fyh, Fzh) : Hand virtual control force [N]
    Dynamics:
            m_e * ddp_e + b_e * dp_e = u_e + w_e
            m_h * ddp_h + b_h * dp_h = u_h + w_h
    """

    def __init__(
        self,
        dt: float = 0.01,
        target_hand: Optional[jnp.ndarray] = None,
        target_elbow: Optional[jnp.ndarray] = None,
        x0: Optional[jnp.ndarray] = None,
        mass_elbow: float = 1.5,
        mass_hand: float = 1.0,
        damping_elbow: float = 4.0,
        damping_hand: float = 3.0,
        w_target_hand: float = 10.0,
        w_target_elbow: float = 10.0,
        l1: float = 0.30,
        l2: float = 0.33,
    ):
        self.dt = dt
        self.mass_elbow = mass_elbow
        self.mass_hand = mass_hand
        self.damping_elbow = damping_elbow
        self.damping_hand = damping_hand
        self.w_target_hand = w_target_hand
        self.w_target_elbow = w_target_elbow
        self.l1 = l1
        self.l2 = l2
        self.is_cartesian = True
        self.is_multipoint = True

        if target_hand is not None:
            self.target_hand = jnp.asarray(target_hand, dtype=jnp.float32)
        else:
            self.target_hand = jnp.array([0.40, 0.12, 0.25], dtype=jnp.float32)

        if target_elbow is not None:
            self.target_elbow = jnp.asarray(target_elbow, dtype=jnp.float32)
        else:
            # Default target elbow from nominal reaching geometry
            self.target_elbow = jnp.array([0.18, 0.05, -0.05], dtype=jnp.float32)

        self.target = self.target_hand  # Primary task target is hand

        if x0 is not None:
            self.x0 = jnp.asarray(x0, dtype=jnp.float32)
        else:
            # Nominal initial state: elbow at (0.24, 0.01, -0.05), hand at (0.34, 0.02, 0.18), zero velocities
            self.x0 = jnp.array([
                0.24, 0.01, -0.05,
                0.34, 0.02, 0.18,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0
            ], dtype=jnp.float32)

        super().__init__(
            state_shape=(12,),
            action_shape=(6,),
            observation_shape=(12,),
            state_noise_shape=(12,),
            obs_noise_shape=(12,),
        )

    def e(self, state: jnp.ndarray) -> jnp.ndarray:
        """Hand 3D Cartesian coordinates [x, y, z] in meters."""
        return state[3:6]

    def elbow(self, state: jnp.ndarray) -> jnp.ndarray:
        """Elbow 3D Cartesian coordinates [x, y, z] in meters."""
        return state[:3]

    def hand(self, state: jnp.ndarray) -> jnp.ndarray:
        """Hand 3D Cartesian coordinates [x, y, z] in meters."""
        return state[3:6]

    def gamma(self, state: jnp.ndarray) -> jnp.ndarray:
        """Task-space Jacobian for hand position de/dx: (3x12)."""
        J = jnp.zeros((3, 12), dtype=jnp.float32)
        return J.at[:, 3:6].set(jnp.eye(3, dtype=jnp.float32))

    def gamma_elbow(self, state: jnp.ndarray) -> jnp.ndarray:
        """Task-space Jacobian for elbow position d(elbow)/dx: (3x12)."""
        J = jnp.zeros((3, 12), dtype=jnp.float32)
        return J.at[:, :3].set(jnp.eye(3, dtype=jnp.float32))

    def edot(self, state: jnp.ndarray) -> jnp.ndarray:
        """Hand 3D velocity [vxh, vyh, vzh]."""
        return state[9:12]

    def keypoints(self, state: jnp.ndarray):
        """Shoulder, Elbow, and Hand keypoints in 3D."""
        p_shoulder = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
        p_elbow = state[:3]
        p_hand = state[3:6]
        return p_shoulder, p_elbow, p_hand

    def _dynamics(self, state, action, noise, params):
        pe = state[:3]
        ph = state[3:6]
        ve = state[6:9]
        vh = state[9:12]

        fe = action[:3]
        fh = action[3:6]

        ae = (fe - self.damping_elbow * ve) / self.mass_elbow
        ah = (fh - self.damping_hand * vh) / self.mass_hand

        dx = jnp.concatenate([ve, vh, ae, ah])
        f = self.dt * dx

        motor_noise_e = params.motor_noise * (fe * noise[6:9]) / self.mass_elbow
        motor_noise_h = params.motor_noise * (fh * noise[9:12]) / self.mass_hand
        w = jnp.sqrt(self.dt) * jnp.concatenate([jnp.zeros(6, dtype=jnp.float32), motor_noise_e, motor_noise_h])

        return state + f + w

    def _observation(self, state, noise, params):
        return state + self.dt * params.obs_noise * noise

    def _cost(self, state, action, params):
        return 0.5 * params.action_cost * jnp.sum(action**2)

    def _final_cost(self, state, params):
        pe_err = state[:3] - self.target_elbow
        ph_err = state[3:6] - self.target_hand
        ve_err = state[6:9]
        vh_err = state[9:12]
        return (
            self.w_target_hand * jnp.sum(ph_err**2)
            + self.w_target_elbow * jnp.sum(pe_err**2)
            + params.velocity_cost * (jnp.sum(ve_err**2) + jnp.sum(vh_err**2))
        )

    def _reset(self, noise, params):
        return self.x0

    @staticmethod
    def get_params_type():
        return CartesianMultiPointReaching3DParams

    @staticmethod
    def get_params_bounds():
        lo = CartesianMultiPointReaching3DParams(
            action_cost=1e-5, velocity_cost=1e-3, motor_noise=1e-2, obs_noise=1e-1
        )
        hi = CartesianMultiPointReaching3DParams(
            action_cost=1e-1, velocity_cost=1e-1, motor_noise=1.0, obs_noise=100.0
        )
        return lo, hi


def joint_to_multipoint_cartesian_3d(states_joint, env_joint):
    """Converts a sequence of articulated joint states [q, qdot] to multi-point Cartesian states [pe, ph, ve, vh].

    Args:
        states_joint: (T, 6) array of joint angles and angular velocities.
        env_joint: Joint space reaching environment instance (providing .keypoints(s) and .dt).

    Returns:
        (T, 12) array of Cartesian states [pe_x, pe_y, pe_z, ph_x, ph_y, ph_z, ve_x, ve_y, ve_z, vh_x, vh_y, vh_z].
    """
    import numpy as np
    states_joint = np.asarray(states_joint, dtype=np.float32)
    dt = getattr(env_joint, "dt", 0.01)

    kps = [env_joint.keypoints(s) for s in states_joint]
    pe = np.array([np.array(kp[1]) for kp in kps], dtype=np.float32)
    ph = np.array([np.array(kp[2]) for kp in kps], dtype=np.float32)

    ve = np.gradient(pe, dt, axis=0) if len(pe) > 1 else np.zeros_like(pe)
    vh = np.gradient(ph, dt, axis=0) if len(ph) > 1 else np.zeros_like(ph)

    return np.hstack([pe, ph, ve, vh]).astype(np.float32)

