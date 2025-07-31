import os
import sys
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.custom_env import GI2DEnv

log_dir = "logs/dqn_improved/"
model_dir = "models/dqn/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

env = DummyVecEnv([lambda: GI2DEnv(render_mode=False)])
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=model_dir,
    name_prefix="dqn_improved_checkpoint"
)

eval_env = DummyVecEnv([lambda: GI2DEnv(render_mode=False)])
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=model_dir,
    log_path=log_dir,
    eval_freq=5_000,
    deterministic=True,
    render=False
)

model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

model.learn(total_timesteps=500_000, callback=[checkpoint_callback, eval_callback])
model.save(os.path.join(model_dir, "dqn_gi2d_improved"))
print("Improved DQN Training Complete.")
