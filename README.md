# GI2D: Reinforcement Learning in a Simulated Human Digestive Tract

##  Project Overview

This project was developed as part of the ALU Machine Learning Techniques II summative assignment. The objective was to implement, train, and evaluate multiple reinforcement learning (RL) algorithms within a **custom 2D gastrointestinal (GI) tract environment**. The task simulates a medical setting in which a **capsule-like agent** must learn to navigate the GI tract and maximize health-related rewards—such as detecting and responding to infections.

l trained and compared four RL algorithms:
- **DQN (Deep Q-Network)**
- **A2C (Advantage Actor-Critic)**
- **PPO (Proximal Policy Optimization)**
- **REINFORCE (Policy Gradient)**

This repository includes all training scripts, the custom environment, performance evaluation tools, visualizations, and a UI-ready simulation layout.

---

##  Game Simulation: The GI2D Environment

The environment, `GI2DEnv`, is a 2D biological simulation representing a simplified version of the human digestive system. It includes:
- **A capsule agent** navigating the tract
- **Dynamic reward signals** based on health state, infection zones, and exploration
- **Visual HUD** showing stats (e.g., infection count, reward, episode)
- **Doctor avatars** reacting to the agent's performance (happy, neutral, sad)

The environment supports `render_mode=True` for visual simulations and `render_mode=False` for fast training.

---

##  Reinforcement Learning Algorithms

Each algorithm was evaluated based on:
- **Cumulative rewards**
- **Training stability (entropy, policy gradient loss, value loss)**
- **Generalization to unseen initial states**

| Algorithm  | Strengths                            | Weaknesses                         |
|------------|---------------------------------------|-------------------------------------|
| PPO        | Best generalization, stable updates  | Requires hyperparameter tuning     |
| DQN        | Efficient in simpler states          | Less stable in complex dynamics    |
| A2C        | Fast training, simple to implement   | Converges early, poor generalization |
| REINFORCE  | Pure policy gradients, interpretable | Slower convergence, unstable       |

---

##  Visualizations & Results

All logs are stored in the `logs/` directory. You can visualize:
- **Cumulative Rewards**
- **Training Stability Metrics**
- **Generalization Tests**

### Example Training Graphs
- `plots/eval_cumulative_reward.png`
- `utils/plot_training_stability.py`

---

## ▶ Agent Demo

**Watch the trained agent play:**  
[![Watch on YouTube](https://img.youtube.com/vi/MsW2cGXvrT0/0.jpg)](https://www.youtube.com/watch?v=MsW2cGXvrT0)  
_This video shows the PPO agent completing 3 episodes in the GI environment, demonstrating how it maximizes reward by making intelligent decisions._

---


## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train a Model
Train any model from the `training/` folder:
```bash
python training/ppo_training.py
python training/dqn_training.py
python training/a2c_training.py
python training/reinforce_improved_training.py
```

### 3. Evaluate All Models
```bash
python training/evaluate_models.py
```

### 4. Visualize Results
```bash
python utils/plot_training_stability.py
python training/plot_cumulative_rewards.py
```

### 5. Play with Best Agent
```bash
python play_best_model.py
```

---

##  Project Structure

```
├── contrib/                  # Custom REINFORCE implementation
├── environment/              # Custom GI2DEnv environment
├── evaluation/               # Evaluation scripts and summaries
├── logs/                     # TensorBoard logs and CSV metrics
├── models/                   # Trained model files
├── plots/                    # Auto-generated training plots
├── rendering/                # UI and doctor avatars
├── training/                 # All training scripts
├── utils/                    # Plotting utilities
├── README.md
├── requirements.txt
```

---

##  Academic Context

This project was completed as part of ALU's **Machine Learning Techniques II** course. It emphasizes:
- Practical RL model training
- Custom environment design
- Model evaluation and performance analysis
- Scientific writing with a detailed report

---

##  Acknowledgments

Thanks to the ALU instructors for teaching this module well , l learned a lot !!!

---
##  Author
    Audry Ashleen Chivanga 
---
