# algorithms/mopderl.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from torch.nn import functional as F


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class MOPDERL:
    def __init__(self, config, actor_class, critic_class, device):
        self.config = config
        self.actor_class = actor_class
        self.critic_class = critic_class
        self.device = device
        self.population_size = config["mopderl"]["population_size"]
        self.state_dim = config["ddpg"]["state_dim"]
        self.action_dim = config["ddpg"]["action_dim"]
        self.hidden_layers = config["ddpg"]["hidden_layers"]
        self.actors = [self.actor_class(self.state_dim, self.action_dim, self.hidden_layers).to(device) for _ in
                       range(self.population_size)]
        # self.critics = [self.critic_class(self.state_dim + self.action_dim, 1, self.hidden_layers).to(device) for _ in
        #                 range(self.population_size)]
        self.critics = [self.critic_class(4, 2, self.hidden_layers).to(device) for _ in range(self.population_size)]
        self.init_population()

    def init_population(self):
        # Initialize the population of actors and critics
        for actor in self.actors:
            actor.apply(self.init_weights)
        for critic in self.critics:
            critic.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def evaluate(self, actor, critic, env, episodes=1):
        total_reward = 0
        for _ in range(episodes):
            state, srd = env.reset()
            terminated = False
            truncated = False
            while not terminated and not truncated:
                state_srd = np.concatenate((state.flatten(), srd.flatten()), axis=None)
                state_tensor = torch.FloatTensor(state_srd).unsqueeze(0).to(self.device)
                action = actor(state_tensor).detach().cpu().numpy()[0]
                next_state, next_srd, reward, terminated, truncated = env.step(action)
                total_reward += reward
                state, srd = next_state, next_srd
        return total_reward

    def generate(self, env):
        fitness_scores = []
        for actor, critic in zip(self.actors, self.critics):
            fitness = self.evaluate(actor, critic, env)
            fitness_scores.append(fitness)
        return fitness_scores

    def optimize(self, actor, critic, replay_buffer, batch_size):
        states, srds, actions, rewards, next_states, next_srds, terminated, truncated = replay_buffer.sample(batch_size)

        # Convert numpy arrays to tensors
        # print('states',states)
        srds = srds.reshape(srds.shape[0], -1)
        next_srds = next_srds.reshape(srds.shape[0], -1)
        # print('srds', srds)
        #print('next_states',next_states)
        states = torch.FloatTensor(states).to(self.device)
        srds = torch.FloatTensor(srds).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        next_srds = torch.FloatTensor(next_srds).to(self.device)
        terminated = torch.FloatTensor(terminated).to(self.device).unsqueeze(1)
        truncated = torch.FloatTensor(truncated).to(self.device).unsqueeze(1)

        states_combined = torch.cat([states, srds], dim=-1)
        next_states_combined = torch.cat([next_states, next_srds], dim=-1)

        with torch.no_grad():
            #print('next_states_combined',next_states_combined)
            #next_actions = actor(next_states_combined)
            next_actions = actor(next_states_combined)
            target_q = critic(next_states, next_actions)
            y = rewards + (1 - terminated) * (1 - truncated) * self.config["ddpg"]["gamma"] * target_q

        current_q = critic(states, actions)
        critic_loss = F.mse_loss(current_q, y)

        critic.optimizer.zero_grad()
        critic_loss.backward()
        critic.optimizer.step()

        actor_loss = -critic(states, actor(states_combined)).mean()

        actor.optimizer.zero_grad()
        actor_loss.backward()
        actor.optimizer.step()

        soft_update(actor.target, actor, self.config["ddpg"]["tau"])
        soft_update(critic.target, critic, self.config["ddpg"]["tau"])
    def optimize_population(self, replay_buffer, batch_size):
        for actor, critic in zip(self.actors, self.critics):
            self.optimize(actor, critic, replay_buffer, batch_size)
