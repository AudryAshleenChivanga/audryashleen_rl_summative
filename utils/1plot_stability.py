import os
import pandas as pd

# Define paths to progress.csv files
log_paths = {
    "PPO": "logs/ppo_improved/progress.csv",
    "A2C": "logs/a2c/progress.csv",
    "DQN": "logs/dqn_improved/progress.csv"
}

print("\n--- Training Stability Summary ---\n")

# Iterate through algorithms
for algo, path in log_paths.items():
    if not os.path.exists(path):
        print(f"[{algo}] ❌ Log file not found at: {path}\n")
        continue

    df = pd.read_csv(path)
    print(f"[{algo}] ✅ Loaded {path}")

    if algo in ["PPO", "A2C"]:
        if "train/policy_gradient_loss" in df.columns:
            pg_loss = df["train/policy_gradient_loss"].dropna()
            print(f"  ↳ Policy Gradient Loss → Final: {pg_loss.iloc[-1]:.4f} | Mean: {pg_loss.mean():.4f} | Min: {pg_loss.min():.4f} | Max: {pg_loss.max():.4f}")
        else:
            print("  ↳ 'train/policy_gradient_loss' not found.")

        if "train/entropy_loss" in df.columns:
            ent_loss = df["train/entropy_loss"].dropna()
            print(f"  ↳ Entropy Loss → Final: {ent_loss.iloc[-1]:.4f} | Mean: {ent_loss.mean():.4f} | Min: {ent_loss.min():.4f} | Max: {ent_loss.max():.4f}")
        else:
            print("  ↳ 'train/entropy_loss' not found.")

    elif algo == "DQN":
        if "train/loss" in df.columns:
            loss = df["train/loss"].dropna()
            print(f"  ↳ DQN Loss → Final: {loss.iloc[-1]:.4f} | Mean: {loss.mean():.4f} | Min: {loss.min():.4f} | Max: {loss.max():.4f}")
        else:
            print("  ↳ 'train/loss' not found.")

    print()

print("--- Done ---")
