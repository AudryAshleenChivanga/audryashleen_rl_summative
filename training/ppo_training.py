# training/ppo_training.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from environment.custom_env import GI2DEnv

env = make_vec_env(lambda: GI2DEnv(render_mode=False), n_envs=1)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,              # Balanced learning rate for stable convergence
    n_steps=2048,                    # Large rollout for better advantage estimation
    batch_size=64,                   # Medium batch size worked well with PPO
    n_epochs=10,                     # More epochs per update for improved policy learning
    gamma=0.99,                      # Longer reward horizon helped detect infection paths
    gae_lambda=0.95,                 # Balanced bias and variance for GAE
    clip_range=0.2,                  # PPO-specific clipping to maintain stability
    ent_coef=0.01,                   # Encouraged exploration
    vf_coef=0.5,                     # Balanced value function loss
    max_grad_norm=0.5,              # Gradient clipping for robustness
    verbose=1,
    tensorboard_log="./logs/"
)

model.learn(total_timesteps=100_000)
model.save("models/ppo/ppo_gi2d")
