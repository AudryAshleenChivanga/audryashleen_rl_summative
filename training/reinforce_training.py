import numpy as np
import torch
from torch.nn import functional as F
import gymnasium as gym

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.type_aliases import GymStepReturn
from stable_baselines3.common.base_class import BaseAlgorithm


class REINFORCE(BaseAlgorithm):
    def __init__(
        self,
        policy: str,
        env: VecEnv,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        verbose: int = 0,
        **kwargs,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            verbose=verbose,
            supported_action_spaces=(gym.spaces.Discrete,),
            **kwargs,
        )
        self.gamma = gamma

    def _setup_model(self) -> None:
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            lr_schedule=lambda _: self.learning_rate,
        )
        self.policy = self.policy.to(self.device)

    def _train(
        self,
        step: int,
        callback,
        log_interval: int,
        tb_log_name: str,
        reset_num_timesteps: bool,
        progress_bar: bool
    ) -> None:
        obs = self.env.reset()
        episode_rewards = []
        log_probs = []
        rewards = []

        for _ in range(self.n_steps):
            obs_tensor = obs_as_tensor(obs, self.device)
            dist = self.policy.get_distribution(obs_tensor)
            actions = dist.sample()
            log_prob = dist.log_prob(actions)
            log_probs.append(log_prob)

            new_obs, reward, done, info = self.env.step(actions.cpu().numpy())
            rewards.append(torch.tensor(reward, dtype=torch.float32, device=self.device))
            obs = new_obs

            if done.any():
                break

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.stack(returns).detach()

        log_probs = torch.stack(log_probs)
        loss = -torch.sum(log_probs * returns)

        self.policy.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer.step()

        self._update_learning_rate([self.policy.optimizer])
        self.logger.record("train/learning_rate", self.learning_rate)
        self.logger.record("train/loss", loss.item())

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval=1,
        tb_log_name="REINFORCE",
        reset_num_timesteps=True,
        progress_bar=False
    ):
        self._setup_model()
        self.n_steps = 2048
        timesteps = 0
        self._logger = self.logger
        while timesteps < total_timesteps:
            self._train(timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar)
            timesteps += self.n_steps
        return self
