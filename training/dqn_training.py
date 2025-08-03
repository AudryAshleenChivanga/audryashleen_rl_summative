import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from environment.custom_env import GI2DEnv
import os

# Setup
env = GI2DEnv(render_mode=False)
check_env(env)

# Train
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,             # Faster learning yielded quicker convergence
    buffer_size=50000,              # Larger buffer helped experience diversity
    learning_starts=1000,           # Initial delay before learning improved early performance
    batch_size=32,                  # Balanced learning stability and speed
    gamma=0.99,                     # Future reward weighting worked well for long-term infection detection
    train_freq=4,                   # Regular updates kept training efficient
    target_update_interval=250,     # Stabilized learning with frequent target updates
    verbose=1,
    tensorboard_log="logs/dqn/"
)


model.learn(total_timesteps=100_000)

# Save
os.makedirs("models/dqn", exist_ok=True)
model.save("models/dqn/gi_dqn")
print(" DQN model saved to models/dqn/gi_dqn")
