import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 400
AGENT_SIZE = 30
TARGET_SIZE = 30
AGENT_SPEED = 15

STAT_PANEL_WIDTH = 300  # Right side for stats and doctor face
TOTAL_WIDTH = WINDOW_WIDTH + STAT_PANEL_WIDTH

class GI2DEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=False, difficulty=0):
    super().__init__()
    self.render_mode = render_mode
    self.difficulty = difficulty
    self.window = None
    self.clock = None

    self.action_space = spaces.Discrete(5)
    self.observation_space = spaces.Box(low=0, high=WINDOW_WIDTH, shape=(4,), dtype=np.float32)

    self.agent_pos = np.array([100, 100], dtype=np.float32)
    self.target_positions = []
    self.visited_targets = set()
    self.prev_distance = None
    self.total_reward = 0
    self.steps = 0
    self.episode = 1
    self.status = ""  # Status message (e.g. "Found infection", etc.)

    self._load_assets()

def _load_assets(self):
    self.asset_dir = os.path.join(os.path.dirname(__file__), "assets")
    self.agent_img = pygame.image.load(os.path.join(self.asset_dir, "pill.png"))
    self.agent_img = pygame.transform.scale(self.agent_img, (AGENT_SIZE, AGENT_SIZE))
    self.target_img = pygame.image.load(os.path.join(self.asset_dir, "infection.png"))
    self.target_img = pygame.transform.scale(self.target_img, (TARGET_SIZE, TARGET_SIZE))
    self.bg_img = pygame.image.load(os.path.join(self.asset_dir, "tract_texture.png"))
    self.bg_img = pygame.transform.scale(self.bg_img, (WINDOW_WIDTH, WINDOW_HEIGHT))

    self.doctor_happy = pygame.image.load(os.path.join(self.asset_dir, "doctor_happy.png"))
    self.doctor_sad = pygame.image.load(os.path.join(self.asset_dir, "doctor_sad.png"))
    self.doctor_neutral = pygame.image.load(os.path.join(self.asset_dir, "doctor_neutral.png"))
    self.doctor_happy = pygame.transform.scale(self.doctor_happy, (100, 100))
    self.doctor_sad = pygame.transform.scale(self.doctor_sad, (100, 100))
    self.doctor_neutral = pygame.transform.scale(self.doctor_neutral, (100, 100))

def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self.agent_pos = np.array([100, 100], dtype=np.float32)
    self.visited_targets = set()
    self.total_reward = 0
    self.steps = 0
    self.status = "Searching for infection..."

    self.target_positions = [
        np.array([
            np.random.uniform(100, WINDOW_WIDTH - TARGET_SIZE),
            np.random.uniform(100, WINDOW_HEIGHT - TARGET_SIZE)
        ], dtype=np.float32)
    ]
    self.prev_distance = self._distance(self.agent_pos, self.target_positions[0])

    if self.render_mode:
        self._init_pygame()
        self._render_frame()

    return self._get_obs(), {}

def _get_obs(self):
    tx, ty = self.target_positions[0]
    return np.array([*self.agent_pos, tx, ty], dtype=np.float32)

def step(self, action):
    dx, dy = 0, 0
    if action == 0: dy = -AGENT_SPEED
    elif action == 1: dy = AGENT_SPEED
    elif action == 2: dx = -AGENT_SPEED
    elif action == 3: dx = AGENT_SPEED

    new_pos = self.agent_pos + np.array([dx, dy], dtype=np.float32)
    if 0 <= new_pos[0] <= WINDOW_WIDTH - AGENT_SIZE and 0 <= new_pos[1] <= WINDOW_HEIGHT - AGENT_SIZE:
        self.agent_pos = new_pos

    reward = 0.0
    done = False
    current_distance = self._distance(self.agent_pos, self.target_positions[0])
    reward += (self.prev_distance - current_distance) * 0.1
    self.prev_distance = current_distance

    if action == 4:
        if self._is_near(self.agent_pos, self.target_positions[0]):
            reward += 20.0
            self.visited_targets.add(0)
            done = True
            self.status = "Infection found!"

    self.total_reward += reward
    self.steps += 1

    if self.steps >= 300:
        if 0 not in self.visited_targets:
            reward -= 10.0
            self.status = "No infection found."
        done = True

    if self.render_mode:
        self._render_frame()

    return self._get_obs(), reward, done, False, {}

def _distance(self, p1, p2):
    return np.linalg.norm(p1 - p2)

def _is_near(self, p1, p2):
    return self._distance(p1, p2) < 30

def _init_pygame(self):
    pygame.init()
    self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT + 100))
    pygame.display.set_caption("GI Capsule Simulation - Infection Detection")
    self.clock = pygame.time.Clock()
    self.font = pygame.font.SysFont("Arial", 24)

def _render_frame(self):
    self.window.fill((255, 255, 255))
    self.window.blit(self.bg_img, (0, 0))

    for i, pos in enumerate(self.target_positions):
        if i not in self.visited_targets:
            self.window.blit(self.target_img, (int(pos[0]), int(pos[1])))

    self.window.blit(self.agent_img, (int(self.agent_pos[0]), int(self.agent_pos[1])))

    # UI Panel
    pygame.draw.rect(self.window, (240, 240, 240), (0, WINDOW_HEIGHT, WINDOW_WIDTH, 100))
    reward_text = self.font.render(f"Reward: {self.total_reward:.2f}", True, (0, 0, 0))
    step_text = self.font.render(f"Steps: {self.steps}", True, (0, 0, 0))
    episode_text = self.font.render(f"Episode: {self.episode}", True, (0, 0, 0))
    status_text = self.font.render(self.status, True, (80, 0, 0))

    self.window.blit(reward_text, (10, WINDOW_HEIGHT + 10))
    self.window.blit(step_text, (10, WINDOW_HEIGHT + 35))
    self.window.blit(episode_text, (10, WINDOW_HEIGHT + 60))
    self.window.blit(status_text, (300, WINDOW_HEIGHT + 10))

    # Dynamic Doctor Avatar
    if "Infection found" in self.status:
        avatar = self.doctor_happy
    elif "No infection" in self.status:
        avatar = self.doctor_sad
    else:
        avatar = self.doctor_neutral

    self.window.blit(avatar, (WINDOW_WIDTH - 120, WINDOW_HEIGHT + 0))

    pygame.display.flip()
    self.clock.tick(30)

def close(self):
    if self.window:
        pygame.quit()