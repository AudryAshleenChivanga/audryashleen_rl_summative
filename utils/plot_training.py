import os
import numpy as np
import matplotlib.pyplot as plt

def plot_eval_results(log_path, title="Evaluation Results"):
    eval_file = os.path.join(log_path, "evaluations.npz")
    if not os.path.exists(eval_file):
        print("No evaluation file found.")
        return

    data = np.load(eval_file)
    timesteps = data["timesteps"]
    results = data["results"]

    mean_rewards = [np.mean(r) for r in results]
    std_rewards = [np.std(r) for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_rewards, label="Mean Reward")
    plt.fill_between(timesteps, np.array(mean_rewards) - np.array(std_rewards),
                     np.array(mean_rewards) + np.array(std_rewards), alpha=0.3, label="Std Dev")

    plt.title(title)
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_eval_results("logs/ppo_improved/", title="Improved PPO Evaluation Performance")
