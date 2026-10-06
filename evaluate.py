from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.tracking_env import UR5eTrackingEnv


def evaluate_policy(model_path="models/best/best_model", n_episodes=3, render=True):

    env = UR5eTrackingEnv(render_mode="rgb_array" if render else None)
    vec_env = DummyVecEnv([lambda: env])

    vec_env = VecNormalize.load("models/vecnormalize_best.pkl", vec_env)
    vec_env.training = False    # freeze norm stats for inference
    vec_env.norm_reward = False

    model = SAC.load(model_path, env=vec_env)

    all_episodes_data = []
    all_tracking_errors = []
    print(f"Running {n_episodes} episodes...")
    for episode_num in range(n_episodes):
        print(f"  Episode {episode_num + 1}/{n_episodes}")

        obs = vec_env.reset()
        episode_tracking_errors = []
        episode_ee_positions = []
        episode_target_positions = []
        video_frames = []

        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)

            tracking_error = info[0]["tracking_error"]
            episode_tracking_errors.append(tracking_error)
            all_tracking_errors.append(tracking_error)

            # Unwrap to get raw positions for plotting
            raw_env = vec_env.venv.envs[0]
            ee_pos = raw_env._get_end_effector_pos()
            target_pos = raw_env.trajectory.get_position(raw_env.sim_time)
            episode_ee_positions.append(ee_pos.copy())
            episode_target_positions.append(target_pos.copy())

            # Record first episode for video
            if episode_num == 0 and render:
                frame = vec_env.venv.envs[0].render()
                if frame is not None:
                    video_frames.append(frame)

        episode_data = {
            "tracking_errors": np.array(episode_tracking_errors),
            "ee_positions": np.array(episode_ee_positions),
            "target_positions": np.array(episode_target_positions),
            "video_frames": video_frames if episode_num == 0 else None,
        }
        all_episodes_data.append(episode_data)

        mean_err = np.mean(episode_tracking_errors)
        max_err = np.max(episode_tracking_errors)
        print(f"    Mean: {mean_err:.4f}m, Max: {max_err:.4f}m")

    all_tracking_errors = np.array(all_tracking_errors)
    stats = {
        "mean_error": float(np.mean(all_tracking_errors)),
        "std_error": float(np.std(all_tracking_errors)),
        "max_error": float(np.max(all_tracking_errors)),
        "percentile_95": float(np.percentile(all_tracking_errors, 95)),
        "episodes_data": all_episodes_data,
        "all_errors": all_tracking_errors,
    }

    vec_env.close()
    return stats


def plot_results(stats, output_path="results/tracking_results.png"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Tracking Performance", fontsize=16, fontweight="bold")

    # Error per step
    ax = axes[0, 0]
    for i, episode_data in enumerate(stats["episodes_data"]):
        ax.plot(episode_data["tracking_errors"], alpha=0.7, label=f"Episode {i + 1}")
    mean_error = np.mean([ed["tracking_errors"] for ed in stats["episodes_data"]], axis=0)
    ax.plot(mean_error, "k--", linewidth=2, label="Mean")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Error (m)")
    ax.set_title("Tracking Error Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # XY trajectory
    ax = axes[0, 1]
    episode_1_data = stats["episodes_data"][0]
    ee_positions = episode_1_data["ee_positions"]
    target_positions = episode_1_data["target_positions"]
    ax.plot(target_positions[:, 0], target_positions[:, 1], "g-", linewidth=2, label="Target")
    ax.plot(ee_positions[:, 0], ee_positions[:, 1], "b.", alpha=0.5, markersize=3, label="Actual")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Path (Episode 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # Per-axis error
    ax = axes[1, 0]
    ee_pos = episode_1_data["ee_positions"]
    target_pos = episode_1_data["target_positions"]
    ax.plot(np.abs(ee_pos[:, 0] - target_pos[:, 0]), label="X", alpha=0.7)
    ax.plot(np.abs(ee_pos[:, 1] - target_pos[:, 1]), label="Y", alpha=0.7)
    ax.plot(np.abs(ee_pos[:, 2] - target_pos[:, 2]), label="Z", alpha=0.7)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Error (m)")
    ax.set_title("Per-Axis Error (Episode 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error histogram
    ax = axes[1, 1]
    all_errors = stats["all_errors"] * 100
    mean_error_cm = stats["mean_error"] * 100
    percentile_95_cm = stats["percentile_95"] * 100
    ax.hist(all_errors, bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(mean_error_cm, color="r", linestyle="--", linewidth=2, label=f"Mean: {mean_error_cm:.2f}cm")
    ax.axvline(percentile_95_cm, color="orange", linestyle="--", linewidth=2, label=f"95%: {percentile_95_cm:.2f}cm")
    ax.set_xlabel("Error (cm)")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close()


def save_video(stats, output_path="results/tracking_video.mp4", fps=25):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    frames = stats["episodes_data"][0]["video_frames"]

    if frames and len(frames) > 0:
        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
        print(f"Saved video to {output_path}")
    else:
        print("No frames to save")

def save_demo_gif(stats, output_path="results/demo.gif", fps=12, stride=2, max_seconds=20):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    ep = stats["episodes_data"][0]
    frames, ee, target = ep["video_frames"], ep["ee_positions"], ep["target_positions"]
    errors = ep["tracking_errors"]

    if not frames:
        print("No frames to build GIF from")
        return

    n = min(len(frames), len(ee))
    idxs = list(range(0, n, stride))[: int(max_seconds * fps)]
    h = frames[0].shape[0]

    out = []
    for i in idxs:
        fig = Figure(figsize=(h / 100, h / 100), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        ax.plot(target[:, 0], target[:, 1], color="0.85", linewidth=1.5)
        ax.plot(ee[: i + 1, 0], ee[: i + 1, 1], color="tab:green", linewidth=1.2, alpha=0.8, label="Actual")
        ax.plot(target[: i + 1, 0], target[: i + 1, 1], color="tab:red", linewidth=2, label="Target")
        ax.plot(ee[i, 0], ee[i, 1], "o", color="tab:green", markersize=7)

        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Tracking error: {errors[i] * 100:.2f} cm")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        canvas.draw()
        plot_img = np.asarray(canvas.buffer_rgba())[:, :, :3]
        out.append(np.hstack([frames[i], plot_img]))

    imageio.mimsave(output_path, out, fps=fps, loop=0)
    print(f"Saved GIF to {output_path}")


def main():
    print("Evaluating...")

    stats = evaluate_policy(model_path="models/best/best_model", n_episodes=3, render=True)

    print("\n" + "=" * 50)
    print(f"Mean error:  {stats['mean_error'] * 100:.2f} cm")
    print(f"Std:         {stats['std_error'] * 100:.2f} cm")
    print(f"95th %ile:   {stats['percentile_95'] * 100:.2f} cm")
    print(f"Max error:   {stats['max_error'] * 100:.2f} cm")
    print("=" * 50)

    print("\nGenerating plots...")
    plot_results(stats)

    print("Saving video...")
    save_video(stats)

    print("Building demo GIF...")
    save_demo_gif(stats)

    print("\nDone!")


if __name__ == "__main__":
    main()
