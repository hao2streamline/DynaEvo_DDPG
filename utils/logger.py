import os

class Logger:
    def __init__(self, config):
        self.log_dir = config["log_dir"]
        os.makedirs(self.log_dir, exist_ok=True)
        self.episode_rewards = []

    def log_step(self, reward, info):
        self.episode_rewards.append(reward)

    def log_episode(self):
        total_reward = sum(self.episode_rewards)
        with open(os.path.join(self.log_dir, "log.txt"), "a") as f:
            f.write(f"Total Reward: {total_reward}\n")
        self.episode_rewards = []

    def save(self):
        pass
