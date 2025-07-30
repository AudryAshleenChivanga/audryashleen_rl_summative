# environment/custom_env.py (2D Free Movement GI Simulation)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 400
AGENT_SIZE = 30
TARGET_SIZE = 30
AGENT_SPEED = 5

class GI2DEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=False):
        super().__init__()
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.action_space = spaces.Discrete(5)  # up, down, left, right, biopsy
        self.observation_space = spaces.Box(low=0, high=WINDOW_WIDTH, shape=(4,), dtype=np.float32)

        self.agent_pos = np.array([100, 100], dtype=np.float32)
        self.target_positions = []
        self.visited_targets = set()

        self._load_assets()

    def _load_assets(self):
        self.asset_dir = os.path.join(os.path.dirname(__file__), "assets")
        self.agent_img = pygame.image.load(os.path.join(self.asset_dir, "pill.png"))
        self.agent_img = pygame.transform.scale(self.agent_img, (AGENT_SIZE, AGENT_SIZE))
        self.target_img = pygame.image.load(os.path.join(self.asset_dir, "infection.png"))
        self.target_img = pygame.transform.scale(self.target_img, (TARGET_SIZE, TARGET_SIZE))
        self.bg_img = pygame.image.load(os.path.join(self.asset_dir, "tract_texture.png"))
        self.bg_img = pygame.transform.scale(self.bg_img, (WINDOW_WIDTH, WINDOW_HEIGHT))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([100, 100], dtype=np.float32)
        self.visited_targets = set()
        self.total_reward = 0
        self.steps = 0

        # One infection per patient (episode), placed randomly
        self.target_positions = [
            np.array([
                np.random.uniform(100, WINDOW_WIDTH - TARGET_SIZE),
                np.random.uniform(100, WINDOW_HEIGHT - TARGET_SIZE)
            ], dtype=np.float32)
        ]

        if self.render_mode:
            self._init_pygame()
            self._render_frame()
        return self._get_obs(), {}

    def _get_obs(self):
        tx, ty = self.target_positions[0]  # Observe one target
        return np.array([*self.agent_pos, tx, ty], dtype=np.float32)

    def step(self, action):
        dx, dy = 0, 0
        if action == 0: dy = -AGENT_SPEED  # up
        elif action == 1: dy = AGENT_SPEED  # down
        elif action == 2: dx = -AGENT_SPEED  # left
        elif action == 3: dx = AGENT_SPEED  # right

        new_pos = self.agent_pos + np.array([dx, dy], dtype=np.float32)
        if 0 <= new_pos[0] <= WINDOW_WIDTH - AGENT_SIZE and 0 <= new_pos[1] <= WINDOW_HEIGHT - AGENT_SIZE:
            self.agent_pos = new_pos

        reward = -0.01
        done = False

        if action == 4:  # biopsy
            for i, t_pos in enumerate(self.target_positions):
                if i not in self.visited_targets and self._is_near(self.agent_pos, t_pos):
                    reward += 10
                    self.visited_targets.add(i)

        if len(self.visited_targets) == len(self.target_positions):
            done = True

        self.total_reward += reward
        self.steps += 1
        if self.steps >= 300:
            done = True

        if self.render_mode:
            self._render_frame()

        return self._get_obs(), reward, done, False, {}

    def _is_near(self, p1, p2):
        return np.linalg.norm(p1 - p2) < 30

    def _init_pygame(self):
        pygame.init()
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("GI Capsule Navigation - 2D Free")
        self.clock = pygame.time.Clock()

    def _render_frame(self):
        self.window.blit(self.bg_img, (0, 0))

        for i, pos in enumerate(self.target_positions):
            if i not in self.visited_targets:
                self.window.blit(self.target_img, (int(pos[0]), int(pos[1])))

        self.window.blit(self.agent_img, (int(self.agent_pos[0]), int(self.agent_pos[1])))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.window:
            pygame.quit()
