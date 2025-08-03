import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env

# Add root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from environment.custom_env import GI2DEnv
from contrib.reinforce.reinforce import REINFORCE

def evaluate_sb3(model, env, episodes=100):
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return rewards

def evaluate_reinforce(model, env, episodes=100):
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.select_action(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return rewards

# Load environment
env = GI2DEnv(render_mode=False)
rewards_dict = {}

# PPO
try:
    model = PPO.load("models/ppo/ppo_gi2d.zip", env=make_vec_env(lambda: GI2DEnv(render_mode=False)))
    rewards_dict["PPO"] = evaluate_sb3(model, env)
except Exception as e:
    print("PPO Evaluation Failed:", e)

# DQN
try:
    model = DQN.load("models/dqn/dqn_gi2d_improved.zip", env=make_vec_env(lambda: GI2DEnv(render_mode=False)))
    rewards_dict["DQN"] = evaluate_sb3(model, env)
except Exception as e:
    print("DQN Evaluation Failed:", e)

# A2C
try:
    model = A2C.load("models/actor_critic/a2c_gi2d.zip", env=make_vec_env(lambda: GI2DEnv(render_mode=False)))
    rewards_dict["A2C"] = evaluate_sb3(model, env)
except Exception as e:
    print("A2C Evaluation Failed:", e)

# REINFORCE
try:
    reinforce_model = REINFORCE(env=env, policy="mlp")
    reinforce_model.load("models/reinforce/reinforce_gi2d_improved.pth")
    rewards_dict["REINFORCE"] = evaluate_reinforce(reinforce_model, env)
except Exception as e:
    print("REINFORCE Evaluation Failed:", e)

# Plotting
plt.figure(figsize=(10, 6))
for name, rewards in rewards_dict.items():
    plt.plot(np.cumsum(rewards), label=name)
plt.title("Cumulative Rewards Over 100 Episodes")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.grid(True)

# Save plot
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/cumulative_rewards_all_models.png")
plt.show()
