import os
import sys
import time

import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.custom_env import GI2DEnv
from training.curriculum.curriculum_manager import CurriculumManager

log_dir = "logs/a2c/"
model_dir = "models/actor_critic_improved/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

def make_env(difficulty=0):
    return lambda: GI2DEnv(render_mode=False, difficulty=difficulty)

curriculum = CurriculumManager()
env = DummyVecEnv([make_env(curriculum.current_level())])

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=model_dir,
    name_prefix="a2c_checkpoint"
)

eval_env = DummyVecEnv([make_env(curriculum.current_level())])
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=model_dir,
    log_path=log_dir,
    eval_freq=5_000,
    deterministic=True,
    render=False
)

new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

total_timesteps = 1_000_000
step = 0
while step < total_timesteps:
    model.learn(total_timesteps=10_000, reset_num_timesteps=False, callback=[checkpoint_callback, eval_callback])
    step += 10_000
    curriculum.advance_if_ready()
    env = DummyVecEnv([make_env(curriculum.current_level())])
    model.set_env(env)
    eval_callback.eval_env = DummyVecEnv([make_env(curriculum.current_level())])

model.save(os.path.join(model_dir, "a2c_gi2d_improved"))
print("Training complete.")
