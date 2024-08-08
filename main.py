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
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import subprocess
import time
import os




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

def start_tensorboard(logdir):
    subprocess.Popen(['tensorboard', '--logdir', logdir])
    time.sleep(5)  # Give TensorBoard some time to start
    print("TensorBoard is running at http://localhost:6006/")

def dominates(a, b):
    """
    Return True if a dominates b.
    """
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))

def get_pareto_front(fitness_scores):
    """
    Calculate Pareto front based on fitness scores.
    """
    pareto_front = []
    for i, score in enumerate(fitness_scores):
        dominated = False
        for j, other_score in enumerate(fitness_scores):
            if i != j and dominates(other_score, score):
                dominated = True
                break
        if not dominated:
            pareto_front.append(score)
    return pareto_front

def plot_pareto_front(pareto_front, episode):
    """
    Plot the Pareto front and save the plot as an image.
    """
    pareto_front = np.array(pareto_front)
    plt.figure()
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='r', marker='o')
    plt.title(f'Pareto Front - Episode {episode}')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.grid(True)
    plt_path = f'runs/pareto_front_episode_{episode}.png'
    plt.savefig(plt_path)
    plt.close()
    return plt_path

def log_pareto_front(writer, pareto_front, episode):
    """
    Log the Pareto front plot to TensorBoard.
    """
    plt_path = plot_pareto_front(pareto_front, episode)
    writer.add_image(f'Pareto Front - Episode {episode}', plt.imread(plt_path), episode, dataformats='HWC')




if __name__ == "__main__":
   # Create a unique log directory based on the current date and time
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    logdir = os.path.join('runs', f'experiment_{current_time}')
    writer = SummaryWriter(logdir)

    # Start TensorBoard server
    start_tensorboard(logdir)

    # Initialize metrics lists
    sample_efficiency_ddpg = []
    sample_efficiency_dyna = []
    sample_efficiency_mopderl = []
    prediction_errors = []
    population_fitness = []
    learning_rates = []

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
    #ea = EA(Actor, config, device)


    # # 初始化回放池
    # # 应写入config与batchsize匹配
    # real_replay_buffer = ReplayBuffer(200)
    # sim_replay_buffer = ReplayBuffer(200)
    #
    # # 收集初始数据
    # real_replay_buffer = collect_data(env, actor, config["training"]["num_episodes_initial"])
    #
    # # 初始化Dyna框架
    # dyna = Dyna(env, mlp_model, actor, critic, target_actor, target_critic, real_replay_buffer, sim_replay_buffer,
    #             config, ea)
    #
    # # 训练环境模型
    # #dyna.update_model(real_replay_buffer.sample(config["ddpg"]["batch_size"]))
    # dyna.update_model()
    #
    # # 保存环境模型
    # mlp_model.save_model('saved_models/env_model')
    #
    # # # 收集使用Actor生成动作的数据
    # # real_replay_buffer = collect_data(env, actor, config["training"]["num_episodes"])
    #
    # # 进行训练
    # dyna.train(config["training"]["num_episodes"])
    # #
    # print("Training completed successfully.")
    replay_buffer = ReplayBuffer(100)

    # Initialize MOPDERL
    mopderl = MOPDERL(config, Actor, Critic, device)

    for episode in range(config["training"]["num_episodes"]):
        # Collect real experience
        state, srd = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0  # Initialize episode reward
        episode_steps = 0  # Initialize episode steps
        while not terminated and not truncated:
            state_srd = np.concatenate((state.flatten(), srd.flatten()), axis=None)
            state_tensor = torch.FloatTensor(state_srd).unsqueeze(0).to(device)
            action = mopderl.actors[0](state_tensor).detach().cpu().numpy()[0]  # Use the first actor in the population
            next_state, next_srd, reward, terminated, truncated = env.step(action)
            replay_buffer.add(state, srd, action, reward, next_state, next_srd, terminated, truncated)
            state = next_state

            episode_reward += reward  # Accumulate reward
            episode_steps += 1  # Count steps

        # Generate simulated experiences and update models
        mopderl.optimize_population(replay_buffer, config["ddpg"]["batch_size"])

        # Calculate and record sample efficiency (example placeholder values)
        ddpg_efficiency = episode_reward / episode_steps  # Placeholder calculation
        dyna_efficiency = episode_reward / episode_steps  # Placeholder calculation
        mopderl_efficiency = episode_reward / episode_steps  # Placeholder calculation
        sample_efficiency_ddpg.append(ddpg_efficiency)
        sample_efficiency_dyna.append(dyna_efficiency)
        sample_efficiency_mopderl.append(mopderl_efficiency)

        # Log sample efficiency metrics to TensorBoard
        writer.add_scalar('Sample Efficiency/DDPG', ddpg_efficiency, episode)
        writer.add_scalar('Sample Efficiency/Dyna-DDPG', dyna_efficiency, episode)
        writer.add_scalar('Sample Efficiency/MOPDERL', mopderl_efficiency, episode)

        # Log metrics to TensorBoard
        writer.add_scalar('Total Reward/Episode', episode_reward, episode)
        writer.add_scalar('Episode Steps', episode_steps, episode)

        # Log learning rates to TensorBoard
        # for i, optimizer in enumerate(mopderl.optimize):
        #     for param_group in optimizer.param_groups:
        #         writer.add_scalar(f'Learning Rate/Actor_{i}', param_group['lr'], episode)
        #     for param_group in mopderl.critics[i].optimizer.param_groups:
        #         writer.add_scalar(f'Learning Rate/Critic_{i}', param_group['lr'], episode)

        # Periodically evaluate the population
        if episode % config["mopderl"]["freq"] == 0:
            fitness_scores = mopderl.generate(env)
            best_index = np.argmax(fitness_scores)
            best_actor = mopderl.actors[best_index]
            print(f"Episode {episode}, Best fitness: {fitness_scores[best_index]}")

            # Log additional metrics
            writer.add_scalar('Best Fitness', fitness_scores[best_index], episode)

            # Calculate and log Pareto front
            pareto_front = get_pareto_front(fitness_scores)
            log_pareto_front(writer, pareto_front, episode)


        # Save the best actor networks
        for i, actor in enumerate(mopderl.actors):
            torch.save(actor.state_dict(), f'best_actor_{i}.pth')

        # Save the final Pareto front
        final_fitness_scores = mopderl.generate(env)
        final_pareto_front = get_pareto_front(final_fitness_scores)

        # Save Pareto front as a plot
        plot_pareto_front(final_pareto_front, 'final')

        # Optionally, save Pareto front data to a file
        with open('pareto_front_data.txt', 'w') as f:
            for obj1, obj2 in final_pareto_front:
                f.write(f"{obj1}, {obj2}\n")

        # Close the TensorBoard writer

        writer.close()





