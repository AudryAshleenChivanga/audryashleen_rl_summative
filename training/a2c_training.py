import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from environment.custom_env import GI2DEnv

env = make_vec_env(lambda: GI2DEnv(render_mode=False), n_envs=1)

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
model.learn(total_timesteps=100_000)
model.save("models/actor_critic/a2c_gi2d")
