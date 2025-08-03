import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from environment.custom_env import GI2DEnv
from contrib.reinforce.reinforce import REINFORCE
from stable_baselines3.common.policies import ActorCriticPolicy

# Set up environment
env = GI2DEnv(render_mode=False)

# Initialize REINFORCE agent
agent = REINFORCE(env=env, policy=ActorCriticPolicy, learning_rate=1e-3, gamma=0.99)

# ✅ Setup model before loading weights
agent._setup_model()

# ✅ Load model weights
model_path = "models/reinforce/reinforce_gi2d_improved.pth"
agent.policy.load_state_dict(torch.load(model_path, map_location=agent.device))

# Evaluate and collect cumulative rewards
episode_rewards = []
n_eval_episodes = 1000

for episode in range(n_eval_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = agent.select_action(obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
    episode_rewards.append(total_reward)

# Cumulative sum
cumulative_rewards = np.cumsum(episode_rewards)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(cumulative_rewards, label="REINFORCE Cumulative Reward", color="orange")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("REINFORCE - Cumulative Reward over 1000 Episodes")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save and show plot
os.makedirs("training/plots", exist_ok=True)
plt.savefig("training/plots/reinforce_cumulative_rewards.png")
plt.show()
