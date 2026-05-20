from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.tracking_env import UR5eTrackingEnv


def create_env(training=True):
    env = UR5eTrackingEnv()
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    return env


# Save normalisation stats whenever a new best model is found
class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, train_env, save_path="models/vecnormalize_best.pkl"):
        super().__init__()
        self.train_env = train_env
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _on_step(self):
        if hasattr(self.parent, 'best_mean_reward'):
            if self.parent.best_mean_reward > self.best_mean_reward:
                self.best_mean_reward = self.parent.best_mean_reward
                self.train_env.save(self.save_path)
                print(f"Saved matching vecnormalize (reward={self.best_mean_reward:.2f})")
        return True


def main():
    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    print("Setting up environments...")
    train_env = create_env(training=True)
    eval_env = create_env(training=True)

    print("Creating SAC agent...")
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=1e-3,
        buffer_size=500000,
        learning_starts=2000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        policy_kwargs={"net_arch": [400, 300]},
        verbose=1,
        tensorboard_log=None,
    )

    print("Setting up callbacks...")
    save_vecnorm_callback = SaveVecNormalizeCallback(train_env)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/best/",
        log_path="logs/eval",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        callback_after_eval=save_vecnorm_callback,
    )

    # Checkpoint every 50k steps as a fallback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="checkpoints/",
        name_prefix="rl_model",
        save_replay_buffer=False,
    )

    print("Training for 1,000,000 steps...")
    model.learn(
        total_timesteps=1000000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    print("Saving final model...")
    model.save("models/final_model")
    train_env.save("models/vecnormalize_final.pkl")

    print("Done! Models in models/, logs in logs/")
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
