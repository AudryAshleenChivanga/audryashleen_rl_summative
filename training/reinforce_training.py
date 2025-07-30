# training/reinforce_training.py

import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.custom_env import GI2DEnv
from contrib.reinforce.reinforce import REINFORCE
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env(lambda: GI2DEnv(render_mode=False), n_envs=1)

from stable_baselines3.common.policies import ActorCriticPolicy
model = REINFORCE(ActorCriticPolicy, env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("models/reinforce_gi2d")
