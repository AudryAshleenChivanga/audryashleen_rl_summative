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
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    train_freq=4,
    target_update_interval=250,
    verbose=1,
    tensorboard_log="logs/dqn/"
)

model.learn(total_timesteps=100_000)

# Save
os.makedirs("models/dqn", exist_ok=True)
model.save("models/dqn/gi_dqn")
print(" DQN model saved to models/dqn/gi_dqn")
