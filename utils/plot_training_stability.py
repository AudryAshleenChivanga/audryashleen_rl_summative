import os
import pandas as pd
import matplotlib.pyplot as plt

# Define your available log paths
log_paths = {
    "PPO": "logs/ppo_improved/progress.csv",
    "A2C": "logs/a2c/progress.csv",
    "DQN": "logs/dqn_improved/progress.csv"
}

# Output folder for saving plots
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Plot stability curves for each algorithm
for algo, path in log_paths.items():
    if not os.path.exists(path):
        print(f"[{algo}] Log file not found at {path}, skipping.")
        continue

    df = pd.read_csv(path)

    if algo in ["PPO", "A2C"]:
        plt.figure(figsize=(10, 4))
        if "train/policy_gradient_loss" in df.columns:
            plt.plot(df["time/total_timesteps"], df["train/policy_gradient_loss"], label="Policy Gradient Loss")
        if "train/entropy_loss" in df.columns:
            plt.plot(df["time/total_timesteps"], df["train/entropy_loss"], label="Entropy Loss")
        plt.title(f"{algo} Training Stability")
        plt.xlabel("Timesteps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{algo.lower()}_stability.png"))
        plt.close()
        print(f"[{algo}] Stability plot saved to {output_dir}/{algo.lower()}_stability.png")

    elif algo == "DQN":
        if "train/loss" in df.columns:
            plt.figure(figsize=(10, 4))
            plt.plot(df["time/total_timesteps"], df["train/loss"], color="purple", label="DQN Loss")
            plt.title("DQN Training Loss Over Time")
            plt.xlabel("Timesteps")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "dqn_loss.png"))
            plt.close()
            print("[DQN] Loss plot saved to plots/dqn_loss.png")
        else:
            print("[DQN] 'train/loss' not found in progress.csv.")
