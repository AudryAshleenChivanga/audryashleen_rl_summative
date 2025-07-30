from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import Endoscopy3DEnv
import matplotlib.pyplot as plt
import numpy as np
import os

log_dir = "logs/ppo/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(Endoscopy3DEnv(render_mode=False), log_dir)

model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
model.learn(total_timesteps=100_000)
model.save("models/pg/endoscopy_3d_ppo")

# Plot rewards from Monitor logs
results = np.loadtxt(log_dir + "monitor.csv", skiprows=2, delimiter=",")
rewards = results[:, 0]
plt.plot(np.cumsum(rewards))
plt.title("Cumulative Reward Over Time")
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.savefig("training_reward_curve.png")
plt.close()
