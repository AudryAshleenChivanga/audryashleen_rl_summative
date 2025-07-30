import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO, DQN, A2C
from environment.custom_env import GI2DEnv
from contrib.reinforce.reinforce import REINFORCE

MODELS = {
    "ppo": ("models/ppo/ppo_gi2d", PPO),
    "dqn": ("models/dqn/gi_dqn", DQN),
    "actor_critic": ("models/actor_critic/a2c_gi2d", A2C),
    "reinforce": ("models/reinforce/reinforce_gi2d", REINFORCE),
}

EPISODES = 1000

def evaluate(model_class, model_path):
    env = GI2DEnv(render_mode=False)
    model = model_class.load(model_path)
    total_rewards = []

    for _ in range(EPISODES):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        total_rewards.append(ep_reward)

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    return mean_reward, std_reward

if __name__ == "__main__":
    print(f"\nEvaluating models on {EPISODES} episodes each...\n")
    results = {}

    for name, (path, model_class) in MODELS.items():
        try:
            if os.path.exists(path + ".zip"):
                mean, std = evaluate(model_class, path)
                results[name] = (mean, std)
                print(f"{name.upper()} -> Mean Reward: {mean:.2f} | Std: {std:.2f}")
            else:
                print(f"{name.upper()} -> Model not found at {path}.zip")
        except Exception as e:
            print(f"{name.upper()} -> Evaluation failed: {e}")

    if results:
        best_model = max(results, key=lambda k: results[k][0])
        print(f"\nBest Model: {best_model.upper()} with mean reward {results[best_model][0]:.2f}")

        labels = list(results.keys())
        means = [results[k][0] for k in labels]
        stds = [results[k][1] for k in labels]

        plt.figure(figsize=(10, 6))
        plt.bar(labels, means, yerr=stds, capsize=5, color='skyblue')
        plt.ylabel('Mean Reward')
        plt.title('Evaluation of RL Models on GI2DEnv')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig("evaluation_results.png")
        plt.show()
