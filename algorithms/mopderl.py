import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy


class MOPDERL:
    def __init__(self, actor, critic, env, config):
        self.actor = actor
        self.critic = critic
        self.env = env
        self.config = config
        self.population_size = config["mopderl"]["population_size"]
        self.mutation_rate = config["mopderl"]["mutation_rate"]
        self.crossover_rate = config["mopderl"]["crossover_rate"]
        self.gamma = config["ddpg"]["gamma"]
        self.tau = config["ddpg"]["tau"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化种群
        self.population = [deepcopy(actor) for _ in range(self.population_size)]
        self.critic_population = [deepcopy(critic) for _ in range(self.population_size)]

    def evaluate_fitness(self, actor):
        total_reward = 0
        state, srd, _ = self.env.reset()
        done = False
        while not done:
            state_srd = np.concatenate((state.flatten(), srd.flatten()), axis=None)
            state_tensor = torch.FloatTensor(state_srd).unsqueeze(0).to(self.device)
            action = actor(state_tensor).detach().cpu().numpy()[0]
            next_state, next_srd, reward, done, _ = self.env.step(action)
            total_reward += reward
            state, srd = next_state, next_srd
        return total_reward

    def moea_stage(self):
        fitness_scores = [self.evaluate_fitness(actor) for actor in self.population]
        sorted_indices = np.argsort(fitness_scores)[-self.population_size // 2:]
        selected_population = [self.population[i] for i in sorted_indices]
        if len(selected_population) % 2 != 0:
            selected_population = selected_population[:-1]

        new_population = []
        for i in range(0, len(selected_population), 2):
            parent1, parent2 = selected_population[i], selected_population[i + 1]
            child1, child2 = self.crossover(parent1, parent2)
            new_population.append(self.mutate(child1))
            new_population.append(self.mutate(child2))
        self.population = new_population

    def crossover(self, parent1, parent2):
        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)
        for param1, param2 in zip(child1.parameters(), child2.parameters()):
            if np.random.rand() < self.crossover_rate:
                param1.data, param2.data = param2.data.clone(), param1.data.clone()
        return child1, child2

    def mutate(self, individual):
        for param in individual.parameters():
            if np.random.rand() < self.mutation_rate:
                param.data += torch.randn_like(param) * 0.1
        return individual

    def rl_stage(self, replay_buffer, batch_size):
        states, srds, actions, rewards, next_states, next_srds, dones = replay_buffer.sample(batch_size)
        states_combined = torch.cat([states, srds], dim=-1).to(self.device)
        next_states_combined = torch.cat([next_states, next_srds], dim=-1).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        for actor, critic in zip(self.population, self.critic_population):
            # Critic update
            with torch.no_grad():
                next_actions = actor(next_states_combined)
                target_q = critic(next_states_combined, next_actions)
                y = rewards + (1 - dones) * self.gamma * target_q

            current_q = critic(states_combined, actions)
            critic_loss = nn.MSELoss()(current_q, y)

            critic.optimizer.zero_grad()
            critic_loss.backward()
            critic.optimizer.step()

            # Actor update
            actor_loss = -critic(states_combined, actor(states_combined)).mean()

            actor.optimizer.zero_grad()
            actor_loss.backward()
            actor.optimizer.step()

            # Soft update
            self.soft_update(actor, actor.target, self.tau)
            self.soft_update(critic, critic.target, self.tau)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
