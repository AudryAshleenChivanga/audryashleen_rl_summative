import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from contrib.reinforce.reinforce import REINFORCE
from environment.custom_env import GI2DEnv
from training.curriculum.curriculum_manager import CurriculumManager

log_dir = "logs/reinforce_improved/"
model_dir = "models/reinforce/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

env = GI2DEnv(render_mode=False)
curriculum = CurriculumManager()
agent = REINFORCE(env=env, policy='mlp')

n_episodes = 5000
for episode in range(n_episodes):
    obs = env.reset()[0]
    done = False
    log_probs = []
    rewards = []

    while not done:
        action, log_prob = agent.select_action(obs)
        obs, reward, done, _, _ = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)

    agent.update_policy(log_probs, rewards)

    total_reward = sum(rewards)
    curriculum.record_reward(total_reward)
    if curriculum.ready_to_advance():
        curriculum.advance_if_ready()

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1} — Total Reward: {total_reward:.2f}")

agent.save(os.path.join(model_dir, "reinforce_gi2d_improved.pth"))
print("REINFORCE Improved Training Complete.")
