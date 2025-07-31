# training/curriculum/curriculum_manager.py

class CurriculumManager:
    def __init__(self):
        self.level = 0
        self.max_level = 5
        self.episode_rewards = []
        self.threshold = -2.0
        self.check_interval = 10

    def current_level(self):
        return self.level

    def record_reward(self, reward):
        self.episode_rewards.append(reward)
        if len(self.episode_rewards) > self.check_interval:
            self.episode_rewards.pop(0)

    def ready_to_advance(self):
        if len(self.episode_rewards) < self.check_interval:
            return False
        return sum(self.episode_rewards) / len(self.episode_rewards) >= self.threshold

    def advance_if_ready(self, last_eval_reward=None):
        if last_eval_reward is not None:
            self.record_reward(last_eval_reward)
        if self.ready_to_advance() and self.level < self.max_level:
            self.level += 1
            self.episode_rewards = []
            print(f"Curriculum advanced to level {self.level}")
