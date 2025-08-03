import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from environment.custom_env import GI2DEnv

env = make_vec_env(lambda: GI2DEnv(render_mode=False), n_envs=1)

model = A2C(
    "MlpPolicy",
    env,
    learning_rate=7e-4,             # A slightly higher learning rate helped stabilize early training
    gamma=0.95,                     # Emphasized shorter-term rewards for responsiveness
    n_steps=5,                      # Small step batch allowed better policy updates per episode
    gae_lambda=0.9,                 # Balanced bias-variance in GAE estimation
    ent_coef=0.01,                  # Added entropy to promote exploration
    vf_coef=0.25,                   # Value function loss had moderate influence
    max_grad_norm=0.5,             # Clipped gradient to prevent instability
    use_rms_prop=True,             # RMSProp worked better than Adam for A2C here
    verbose=1,
    tensorboard_log="./logs/"
)

model.learn(total_timesteps=100_000)
model.save("models/actor_critic/a2c_gi2d")
