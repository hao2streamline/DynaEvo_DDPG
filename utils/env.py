import numpy as np

class VirtualHighwayEnv:
    def __init__(self, env, model):
        self.env = env
        self.model = model

    def reset(self):
        return self.env.reset()

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
            done = done or truncated  # 合并 done 和 truncated 标志
        elif len(result) == 4:
            obs, reward, done, info = result
        else:
            raise ValueError(f"Unexpected number of values returned from env.step(action): {len(result)}")

        # 使用模型预测奖励
        predicted_reward = self.model.predict(obs).mean().item()  # 将张量转换为标量，这里取平均值
        return obs, predicted_reward, done, info
