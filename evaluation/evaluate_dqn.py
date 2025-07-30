from stable_baselines3 import DQN
from environment.custom_env import HPyloriEnv
from stable_baselines3.common.evaluation import evaluate_policy

env = HPyloriEnv()
model = DQN.load("models/dqn/hpylori_dqn")

mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Evaluation mean reward: {mean_reward}")
