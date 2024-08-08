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
        current_speed = self.env.unwrapped.vehicle.velocity
        action = self.limit_action(action, current_speed)
        #print('action',action)

        obs, reward, terminated, truncated, info = self.env.step(action)

        next_state = obs[0]
        srd = obs[1:4]
        reward += 1 / (1 + abs(current_speed[1]))
        if self.is_off_road(self.env.unwrapped.vehicle):
            reward -= 100  # 给予负奖励
            terminated = True  # 结束当前episode

        return next_state, srd, reward, terminated, truncated


    def is_off_road(self, vehicle):
        road_width = 50
        return vehicle.position[1] < 0 or vehicle.position[1] > road_width

    def limit_action(self, action, current_speed):
        #顺序反了 速度角度
        #加速度（-5,5）
        #转动角度（-0.7853981633974483, 0.7853981633974483）
        max_steering = 5  # 最大转向角
        max_acceleration = 0.0005
        # 最大加速度
        min_speed = 40  # 最低速度

        speed = np.linalg.norm(current_speed)  # 计算速度向量的模

        # 计算新的速度
        new_speed = speed + action[0]

        # 确保新的速度不低于最低速度
        if new_speed < min_speed:
            new_speed = min_speed  # 设置为最小速度
            action[0] = new_speed - speed  # 重新计算加速度以确保速度不低于最小速度
        else:
            # 限制加速度范围
            action[0] = np.clip(action[0], -max_acceleration, max_acceleration)

        action[0] = np.clip(action[0], -max_steering, max_steering)
        action[1] = np.clip(action[1], -max_acceleration, max_acceleration)

        #print('current_speed',current_speed)

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