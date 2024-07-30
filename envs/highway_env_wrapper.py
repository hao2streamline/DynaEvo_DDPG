import gymnasium as gym
import yaml
import numpy as np


class HighwayEnv:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        self.env = gym.make(id=config["environment"]["id"], render_mode=config["environment"]["render_mode"])
        self.env.unwrapped.configure(config["environment"])
        self.env.unwrapped.configure({
            "action": {
                "type": "ContinuousAction"
            }
        })
        self.env.reset()
        self.action_space = self.env.action_space

    def reset(self):
        obs, info = self.env.reset()
        state, srd = self.split_obs(obs)
        return state, srd

    def step(self, action):
        action = self.limit_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        next_state = obs[0]
        srd = obs[1:4]
        if self.is_off_road(self.env.unwrapped.vehicle):
            reward -= 100  # 给予负奖励
            terminated = True  # 结束当前episode
        return next_state, srd, reward, terminated, truncated


    def is_off_road(self, vehicle):
        road_width = 50
        return vehicle.position[1] < 0 or vehicle.position[1] > road_width

    def limit_action(self, action):
        max_steering = 10  # 最大转向角
        max_acceleration = 0.001  # 最大加速度
        action[0] = np.clip(action[0], -max_steering, max_steering)
        action[1] = np.clip(action[1], -max_acceleration, max_acceleration)
        return action

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()

    def split_obs(self, obs):
        state = obs[0]
        srd = obs[1:]
        return state, srd


#h = HighwayEnv('../config/config.yaml')