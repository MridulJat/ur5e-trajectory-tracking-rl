import numpy as np
import pytest

from trajectory import FigureEightTrajectory
from env.tracking_env import UR5eTrackingEnv


class TestTrajectory:

    def test_trajectory_initialization(self):
        centre = (0.5, 0.0, 0.55)
        scale_x = 0.12
        scale_y = 0.08
        period = 6.0

        traj = FigureEightTrajectory(centre=centre, scale_x=scale_x, scale_y=scale_y, period=period)

        assert np.allclose(traj.centre, centre)
        assert traj.scale_x == scale_x
        assert traj.scale_y == scale_y
        assert traj.period == period

    def test_position_output_shape(self):
        traj = FigureEightTrajectory()
        pos = traj.get_position(t=0.0)

        assert isinstance(pos, np.ndarray)
        assert pos.shape == (3,)
        assert pos.dtype == np.float32

    def test_velocity_output_shape(self):
        traj = FigureEightTrajectory()
        vel = traj.get_velocity(t=0.0)

        assert isinstance(vel, np.ndarray)
        assert vel.shape == (3,)
        assert vel.dtype == np.float32

    def test_z_position_constant(self):
        traj = FigureEightTrajectory(centre=(0.5, 0.0, 0.55))

        times = np.linspace(0, 6.0, 100)
        z_positions = [traj.get_position(t)[2] for t in times]

        assert all(np.isclose(z, 0.55) for z in z_positions)

    def test_z_velocity_zero(self):
        traj = FigureEightTrajectory()

        times = np.linspace(0, 6.0, 100)
        z_velocities = [traj.get_velocity(t)[2] for t in times]

        assert all(np.isclose(vz, 0.0) for vz in z_velocities)

    def test_figure_eight_shape(self):
        traj = FigureEightTrajectory(
            centre=(0.0, 0.0, 0.0), scale_x=1.0, scale_y=0.5, period=2.0
        )

        t_values = [0.0, 0.5, 1.0, 1.5, 2.0]
        positions = [traj.get_position(t) for t in t_values]

        # t=0: origin
        assert np.allclose(positions[0][:2], [0.0, 0.0], atol=1e-5)

        # t=0.5 (quarter period): sin(pi/2)=1, sin(pi)=0
        assert np.isclose(positions[1][0], 1.0, atol=1e-5)
        assert np.isclose(positions[1][1], 0.0, atol=1e-5)

        # t=1.0 (half period): back to origin
        assert np.isclose(positions[2][0], 0.0, atol=1e-5)
        assert np.isclose(positions[2][1], 0.0, atol=1e-5)

    def test_workspace_bounds(self):
        traj = FigureEightTrajectory(
            centre=(0.5, 0.0, 0.55), scale_x=0.12, scale_y=0.08
        )

        times = np.linspace(0, traj.period, 1000)
        positions = np.array([traj.get_position(t) for t in times])

        assert np.all(positions[:, 0] >= 0.5 - 0.12 - 0.01)  # X min
        assert np.all(positions[:, 0] <= 0.5 + 0.12 + 0.01)  # X max
        assert np.all(positions[:, 1] >= 0.0 - 0.08 - 0.01)  # Y min
        assert np.all(positions[:, 1] <= 0.0 + 0.08 + 0.01)  # Y max
        assert np.all(np.isclose(positions[:, 2], 0.55))      # Z fixed

    def test_velocity_derivative(self):
        traj = FigureEightTrajectory()
        dt = 0.001

        for t in np.linspace(0, traj.period, 50):
            vel_analytical = traj.get_velocity(t)

            # Check against finite difference
            pos_forward = traj.get_position(t + dt)
            pos_backward = traj.get_position(t - dt)
            vel_numerical = (pos_forward - pos_backward) / (2 * dt)

            assert np.allclose(vel_analytical, vel_numerical, atol=1e-3)


class TestEnvironment:

    @pytest.fixture
    def env(self):
        env = UR5eTrackingEnv(max_episode_steps=100)
        yield env
        env.close()

    def test_env_initialization(self, env):
        assert env.action_space.shape == (6,)
        assert env.observation_space.shape == (21,)
        assert env.action_space.low.min() == -1.0
        assert env.action_space.high.max() == 1.0

    def test_reset_returns_correct_shapes(self, env):
        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (21,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)
        assert "tracking_error" in info
        assert "sim_time" in info

    def test_reset_clears_counters(self, env):
        env.reset()
        assert env.step_count == 0
        assert env.sim_time == 0.0

    def test_reset_clears_action_buffer(self, env):
        env.reset()
        assert len(env.action_buffer) == env.control_delay_steps
        for action in env.action_buffer:
            assert np.allclose(action, np.zeros(6))

    def test_step_returns_correct_types(self, env):
        env.reset()
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (21,)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_space_bounds(self, env):
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)

    def test_episode_terminates_within_max_steps(self, env):
        env.reset()

        for step in range(env.max_episode_steps + 100):
            action = env.action_space.sample()
            _, _, _, truncated, _ = env.step(action)

            if truncated:
                assert step >= env.max_episode_steps - 1
                break
        else:
            pytest.fail("Episode did not truncate within expected steps")

    def test_tracking_error_non_negative(self, env):
        env.reset()

        for _ in range(50):
            action = env.action_space.sample()
            _, _, _, _, info = env.step(action)

            assert info["tracking_error"] >= 0.0

    def test_observation_affected_by_noise(self):
        env_with_noise = UR5eTrackingEnv(obs_noise_std=0.01, max_episode_steps=100)

        obs1, _ = env_with_noise.reset(seed=42)
        obs2, _ = env_with_noise.reset(seed=99)

        assert not np.allclose(obs1, obs2)

        env_with_noise.close()

    def test_no_observation_noise_when_zero(self):
        env_no_noise = UR5eTrackingEnv(obs_noise_std=0.0, max_episode_steps=100)

        obs1, _ = env_no_noise.reset(seed=42)
        obs2, _ = env_no_noise.reset(seed=42)

        assert np.allclose(obs1, obs2)

        env_no_noise.close()

    def test_home_position_set_on_reset(self, env):
        env.reset()

        qpos = env.data.qpos[:6]
        assert np.allclose(qpos, env.home_qpos, atol=0.01)

    def test_action_scaling(self, env):
        env.reset()

        action = np.ones(6, dtype=np.float32)
        env.step(action)

    def test_sim_time_advances(self, env):
        env.reset()
        assert env.sim_time == 0.0

        for _ in range(10):
            action = np.zeros(6, dtype=np.float32)
            env.step(action)

        assert env.sim_time > 0.0

    def test_step_count_increments(self, env):
        env.reset()
        assert env.step_count == 0

        for expected_step in range(1, 11):
            action = env.action_space.sample()
            env.step(action)
            assert env.step_count == expected_step

    def test_reward_is_finite(self, env):
        env.reset()

        for _ in range(50):
            action = env.action_space.sample()
            _, reward, _, _, _ = env.step(action)

            assert np.isfinite(reward), f"Reward is not finite: {reward}"

    def test_observation_contains_valid_values(self, env):
        env.reset()

        for _ in range(50):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)

            assert not np.any(np.isnan(obs)), "Observation contains NaN values"
            assert np.all(np.abs(obs) < 1e6), "Observation values are unreasonably large"
