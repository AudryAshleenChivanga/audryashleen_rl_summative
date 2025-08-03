import os
import sys
import torch
import numpy as np

from stable_baselines3.common.policies import ActorCriticPolicy

# Add root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from environment.custom_env import GI2DEnv
from contrib.reinforce.reinforce import REINFORCE

# Initialize environment
env = GI2DEnv(render_mode=False)

# Set up agent with ActorCriticPolicy instead of "mlp"
agent = REINFORCE(
    env=env,
    policy=ActorCriticPolicy,
    learning_rate=1e-3,
    gamma=0.99,
    verbose=1,
)

# Train
total_timesteps = 500_000
agent.learn(total_timesteps=total_timesteps)

# Save
os.makedirs("models/reinforce", exist_ok=True)
model_path = "models/reinforce/reinforce_gi2d_improved.pth"
torch.save(agent.policy.state_dict(), model_path)

print(f"REINFORCE model saved to {model_path}")
