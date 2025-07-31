# training/retrain_all_improved.py

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from environment.custom_env import GI2DEnv
from training.curriculum.curriculum_manager import CurriculumManager

# DQN
from stable_baselines3 import DQN
# PPO
from stable_baselines3 import PPO
# A2C
from stable_baselines3 import A2C
# REINFORCE
from contrib.reinforce.reinforce import REINFORCE


def train_dqn():
    env = GI2DEnv(render_mode=False)
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="logs/dqn/")
    model.learn(total_timesteps=100_000)
    model.save("models/dqn/dqn_gi2d_improved")
    print("DQN Improved Training Complete.")

def train_ppo():
    env = GI2DEnv(render_mode=False)
    curriculum = CurriculumManager()
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/ppo_improved/")
    total_timesteps = 1_000_000
    eval_interval = 5000
    timesteps = 0

    while timesteps < total_timesteps:
        model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
        timesteps += eval_interval
        mean_reward = evaluate_model(model, env)
        curriculum.advance_if_ready(mean_reward)

    model.save("models/ppo/ppo_gi2d_improved")
    print("PPO Improved Training Complete.")

def train_a2c():
    env = GI2DEnv(render_mode=False)
    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="logs/a2c/")
    model.learn(total_timesteps=500_000)
    model.save("models/actor_critic/a2c_gi2d_improved")
    print("A2C Improved Training Complete.")

def train_reinforce():
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
            print(f"[REINFORCE] Episode {episode + 1} — Total Reward: {total_reward:.2f}")

    agent.save("models/reinforce/reinforce_gi2d_improved.pth")
    print("REINFORCE Improved Training Complete.")

def evaluate_model(model, env, n_eval_episodes=5):
    all_rewards = []
    for _ in range(n_eval_episodes):
        obs = env.reset()[0]
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        all_rewards.append(total_reward)
    return sum(all_rewards) / len(all_rewards)

if __name__ == "__main__":
    train_dqn()
    train_ppo()
    train_a2c()
    train_reinforce()
