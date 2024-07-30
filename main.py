import yaml
import gymnasium as gym
import numpy as np
from models.mlp import MLPModel
from models.actor import Actor
from models.critic import Critic
from algorithms.ea import EA
from algorithms.dyna import Dyna
from utils.replay_buffer import ReplayBuffer
from envs.highway_env_wrapper import HighwayEnv
from algorithms.mopderl import MOPDERL
import torch


def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def initialize_components(config):
    # Initialize environment model
    env_model = MLPModel(config["mlp_model"])

    # Initialize DDPG networks
    actor = Actor(config["ddpg"]["state_dim"], config["ddpg"]["action_dim"], config["ddpg"]["hidden_layers"])
    critic = Critic(config["ddpg"]["state_dim"], config["ddpg"]["action_dim"], config["ddpg"]["hidden_layers"])

    # Initialize EA population
    ea = EA(config["ea"])

    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(config["replay_buffer"]["capacity"])

    return env_model, actor, critic, ea, replay_buffer


def collect_data(env, actor, num_episodes, use_actor=True):
    replay_buffer = ReplayBuffer(200)
    for episode in range(num_episodes):
        state, srd = env.reset()
        terminated= False
        truncated = False
        while not terminated and not truncated:
            if use_actor:
                combined_state = np.concatenate([state, srd.flatten()])
                state_tensor = torch.FloatTensor(combined_state).unsqueeze(0).to(device)
                action = actor(state_tensor).detach().cpu().numpy()[0]
            else:
                action = env.action_space.sample()
            next_state, next_srd, reward, terminated, truncated = env.step(action)
            replay_buffer.add(state, srd, action, reward, next_state, next_srd, terminated, truncated)
            state, srd = next_state, next_srd
    return replay_buffer



if __name__ == "__main__":
    config = load_config('config/config.yaml')

    # 初始化环境
    env = HighwayEnv('config/config.yaml')

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_model = MLPModel(config["mlp_model"]).to(device)
    actor = Actor(config["ddpg"]["state_dim"], config["ddpg"]["action_dim"], config["ddpg"]["hidden_layers"]).to(device)
    critic = Critic(4, 2, config["ddpg"]["hidden_layers"]).to(device)
    target_actor = Actor(config["ddpg"]["state_dim"], config["ddpg"]["action_dim"], config["ddpg"]["hidden_layers"]).to(device)
    target_critic = Critic(4, 2, config["ddpg"]["hidden_layers"]).to(device)
    ea = EA(Actor, config, device)

    # 初始化回放池
    # 应写入config与batchsize匹配
    real_replay_buffer = ReplayBuffer(200)
    sim_replay_buffer = ReplayBuffer(200)

    # 收集初始数据
    real_replay_buffer = collect_data(env, actor, config["training"]["num_episodes_initial"])

    # 初始化Dyna框架
    dyna = Dyna(env, mlp_model, actor, critic, target_actor, target_critic, real_replay_buffer, sim_replay_buffer,
                config, ea)

    # 训练环境模型
    #dyna.update_model(real_replay_buffer.sample(config["ddpg"]["batch_size"]))
    dyna.update_model()

    # 保存环境模型
    mlp_model.save_model('saved_models/env_model')

    # # 收集使用Actor生成动作的数据
    # real_replay_buffer = collect_data(env, actor, config["training"]["num_episodes"])

    # 进行训练
    dyna.train(config["training"]["num_episodes"])
    #
    print("Training completed successfully.")
