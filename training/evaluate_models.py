
import os
import sys
import numpy as np
import torch
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from environment.custom_env import GI2DEnv
from contrib.reinforce.reinforce import REINFORCE

def evaluate(model, env, n_eval_episodes=1000):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, render=False)
    return mean_reward, std_reward

env = GI2DEnv(render_mode=False)

print("\nEvaluating models on 1000 episodes each...\n")
results = {}
models = {}

# PPO
try:
    print("Loading PPO...")
    model = PPO.load("models/ppo/ppo_improved/ppo_gi2d_improved.zip", env=env)
    mean_reward, std_reward = evaluate(model, env)
    results["PPO"] = (mean_reward, std_reward)
    models["PPO"] = model
    print(f"PPO -> Mean Reward: {mean_reward:.2f} | Std: {std_reward:.2f}")
except Exception as e:
    print(f"PPO -> Evaluation failed: {e}")

# DQN
try:
    print("Loading DQN...")
    model = DQN.load("models/dqn/dqn_gi2d_improved.zip", env=env)
    mean_reward, std_reward = evaluate(model, env)
    results["DQN"] = (mean_reward, std_reward)
    models["DQN"] = model
    print(f"DQN -> Mean Reward: {mean_reward:.2f} | Std: {std_reward:.2f}")
except Exception as e:
    print(f"DQN -> Evaluation failed: {e}")

# ACTOR_CRITIC (A2C)
try:
    print("Loading ACTOR_CRITIC...")
    model = A2C.load("models/actor_critic/a2c_gi2d.zip", env=env)
    mean_reward, std_reward = evaluate(model, env)
    results["ACTOR_CRITIC"] = (mean_reward, std_reward)
    models["ACTOR_CRITIC"] = model
    print(f"ACTOR_CRITIC -> Mean Reward: {mean_reward:.2f} | Std: {std_reward:.2f}")
except Exception as e:
    print(f"ACTOR_CRITIC -> Evaluation failed: {e}")

# REINFORCE
try:
    print("Loading REINFORCE...")
    reinforce_model = REINFORCE(policy='mlp', env=env)
    reinforce_model.load("models/reinforce/reinforce_gi2d_improved.pth")
    
    rewards = []
    for _ in range(1000):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = reinforce_model.select_action(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    results["REINFORCE"] = (mean_reward, std_reward)
    models["REINFORCE"] = reinforce_model
    print(f"REINFORCE -> Mean Reward: {mean_reward:.2f} | Std: {std_reward:.2f}")
except Exception as e:
    print(f"REINFORCE -> Evaluation failed: {e}")

# Save best model
if results:
    best_model_name, (best_reward, _) = max(results.items(), key=lambda x: x[1][0])
    print(f"\n Best Model: {best_model_name} with mean reward {best_reward:.2f}")

    os.makedirs("models/best", exist_ok=True)

    if best_model_name == "REINFORCE":
        models[best_model_name].save("models/best/best_model.pth")
    else:
        models[best_model_name].save("models/best/best_model.zip")
else:
    print("\n No models evaluated successfully.")
