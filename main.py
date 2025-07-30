# main.py
from environment.custom_env import GI2DEnv
import time

if __name__ == "__main__":
    env = GI2DEnv(render_mode=True)
    NUM_EPISODES = 5

    for episode in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step = 0

        print(f"\n Starting Episode {episode}")

        while not done:
            action = env.action_space.sample()  # Random for now; replace with model.predict(obs) later
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            step += 1

        print(f" Episode {episode} finished after {step} steps with Total Reward: {total_reward:.2f}")
        time.sleep(1)

    env.close()
    print("Simulation complete.")
