import collections
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from trajectory import FigureEightTrajectory


# UR5e gym env - tracks a figure-eight trajectory
class UR5eTrackingEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        obs_noise_std=0.002,
        action_noise_std=0.001,
        control_delay_steps=2,
        dt=0.002,
        max_episode_steps=1000,
        render_mode=None,
    ):
        xml_path = Path(__file__).parent.parent / "assets" / "scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        self.obs_noise_std = obs_noise_std
        self.action_noise_std = action_noise_std
        self.dt = dt
        self.max_episode_steps = max_episode_steps

        # Action delay buffer - simulates arm reaction time
        self.control_delay_steps = control_delay_steps
        self.action_buffer = collections.deque(maxlen=control_delay_steps)

        self.sim_steps_per_action = 10  # multiple physics steps per action for stability
        self.home_qpos = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float32)
        self.joint_vel_limit = np.array([3.14] * 6, dtype=np.float32)
        self.trajectory = FigureEightTrajectory()

        self.render_mode = render_mode
        self.renderer = None
        self.step_count = 0
        self.sim_time = 0.0

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # 21-dim: 6 joint pos + 6 joint vel + 3 target pos + 3 target vel + 3 EE pos
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:6] = self.home_qpos
        self.data.qvel[:6] = 0.0
        self.data.ctrl[:6] = self.home_qpos
        mujoco.mj_forward(self.model, self.data)

        # Let the arm settle before the episode starts
        for _ in range(500):
            self.data.ctrl[:6] = self.home_qpos
            mujoco.mj_step(self.model, self.data)

        self.step_count = 0
        self.sim_time = 0.0

        self.action_buffer = collections.deque(
            [np.zeros(6, dtype=np.float32)] * self.control_delay_steps,
            maxlen=self.control_delay_steps
        )

        observation = self._get_obs()
        info = {"tracking_error": self._tracking_error(), "sim_time": self.sim_time}

        return observation, info

    def _get_end_effector_pos(self):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
        ee_pos = self.data.xpos[body_id].copy()
        return ee_pos.astype(np.float32)

    def _get_obs(self):
        # Joint state - with sensor noise
        qpos = self.data.qpos[:6].copy()
        qpos_noisy = qpos + self.np_random.normal(0, self.obs_noise_std, size=6)

        qvel = self.data.qvel[:6].copy()
        qvel_noisy = qvel + self.np_random.normal(0, self.obs_noise_std, size=6)

        # Trajectory target and current EE position
        target_pos = self.trajectory.get_position(self.sim_time)
        target_vel = self.trajectory.get_velocity(self.sim_time)
        ee_pos = self._get_end_effector_pos()

        obs = np.concatenate(
            [qpos_noisy, qvel_noisy, target_pos, target_vel, ee_pos],
            dtype=np.float32,
        )

        return obs

    def step(self, action):
        scaled_action = action * self.joint_vel_limit

        noisy_action = scaled_action + self.np_random.normal(
            0, self.action_noise_std, size=6
        )

        self.action_buffer.append(noisy_action)

        # Pull oldest action from buffer - what executes now after the delay
        if len(self.action_buffer) == self.control_delay_steps:
            delayed_action = self.action_buffer[0]
        else:
            delayed_action = np.zeros(6, dtype=np.float32)

        # Velocity integration -> target joint positions
        actual_dt = self.dt * self.sim_steps_per_action
        new_qpos = self.data.qpos[:6] + delayed_action * actual_dt
        self.data.ctrl[:6] = new_qpos

        for _ in range(self.sim_steps_per_action):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        self.sim_time += self.dt * self.sim_steps_per_action

        reward = self._compute_reward()
        obs = self._get_obs()

        terminated = False
        truncated = self.step_count >= self.max_episode_steps

        info = {
            "tracking_error": self._tracking_error(),
            "sim_time": self.sim_time,
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        ee_pos = self._get_end_effector_pos()
        target_pos = self.trajectory.get_position(self.sim_time)

        dist = np.linalg.norm(ee_pos - target_pos)

        tracking_reward = np.exp(-5.0 * dist)       # broad shaping
        close_bonus = np.exp(-50.0 * dist) * 2.0    # extra reward for being very close

        # Penalise jerky motion and large control inputs
        smoothness_penalty = -0.001 * np.linalg.norm(self.data.qacc[:6])
        action_penalty = -0.0001 * np.sum(np.square(self.data.ctrl[:6]))

        return float(tracking_reward + close_bonus + smoothness_penalty + action_penalty)

    def _tracking_error(self):
        ee_pos = self._get_end_effector_pos()
        target_pos = self.trajectory.get_position(self.sim_time)
        error = np.linalg.norm(ee_pos - target_pos)
        return float(error)

    def render(self):
        if self.render_mode != "rgb_array":
            return None

        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model)

        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
        self.renderer.update_scene(self.data, camera=camera_id)
        rgb_array = self.renderer.render()

        return rgb_array

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
