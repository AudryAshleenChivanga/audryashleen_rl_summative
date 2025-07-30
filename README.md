"""
# Autonomous Endoscopy 3D RL Simulation

This project simulates a capsule endoscope navigating a 3D digestive tract using reinforcement learning. It uses PyBullet for 3D physics and visualization, and Stable-Baselines3 for training.

## Features
- PyBullet-powered 3D capsule robot
- Realistic infection detection + biopsy
- PPO + DQN RL agents
- Capsule-view camera, reward overlays
- Log plots and optional GIF/video generation

## Setup
```bash
pip install -r requirements.txt
```

## Run Random Agent (Visual Demo)
```bash
python main.py
```

## Train PPO Agent
```bash
python training/pg_training.py
```

## Train DQN Agent
```bash
python training/dqn_training.py
```
