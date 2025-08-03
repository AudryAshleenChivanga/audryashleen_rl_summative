# play_best_model.py
from stable_baselines3 import PPO
from environment.custom_env import GI2DEnv

def evaluate_trained_model(model_path):
    env = GI2DEnv(render_mode=True)  # Render enabled
    model = PPO.load(model_path)

    obs, _ = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

    env.close()

if __name__ == "__main__":
    evaluate_trained_model("models/best/best_model.zip")
