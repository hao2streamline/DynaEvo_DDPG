import numpy as np
import torch
import torch.nn.functional as F
from models.actor import Actor


class Dyna:
    def __init__(self, env, env_model, actor, critic, target_actor, target_critic, real_replay_buffer,
                 sim_replay_buffer, config, ea):
        self.env = env
        self.env_model = env_model
        self.actor = actor
        self.critic = critic
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.real_replay_buffer = real_replay_buffer
        self.sim_replay_buffer = sim_replay_buffer
        self.config = config
        self.ea = ea
        self.actor_model_class = Actor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config["ddpg"]["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config["ddpg"]["critic_lr"])

    def update_model(self):
        batch_size = self.config["ddpg"]["batch_size"]
        print('batch_size',batch_size)
        print('self.real_replay_buffer',len(self.real_replay_buffer))
        states, srds, actions, rewards, next_states, next_srds, terminated, truncated = self.real_replay_buffer.sample(batch_size)
        next_states_rewards = np.hstack((next_states, rewards.reshape(-1, 1)))
        self.env_model.update(states, actions, srds, next_states_rewards)



    def generate_simulated_data(self, num_samples):
        simulated_data = []
        for _ in range(num_samples):
            state, srd = self.env.reset()
            terminated = False
            truncated = False
            s = 0
            maxstep = 20
            while not terminated and not truncated:
                state_srd = np.concatenate((state, srd), axis=None)
                # print('state_srd',state_srd)
                state_tensor = torch.FloatTensor(state_srd).unsqueeze(0).to(self.device)
                # print('state_tensor',state_tensor)
                action = self.actor(state_tensor).detach().cpu().numpy()[0]

                # next_state_reward = self.env_model.predict(state, srd, action)
                # next_state = next_state_reward[:self.config["ddpg"]["state_dim"]]
                # reward = next_state_reward[self.config["ddpg"]["state_dim"]]
                next_state, next_srd, reward = self.env_model.predict(state, srd, action)
                # print('next_state',next_state)
                # print('next_srd',next_srd)
                # print('reward',reward)
                terminated = self.env.is_off_road(self.env.env.unwrapped.vehicle)
                if s > maxstep: truncated = True
                s += 1

                simulated_data.append((state, srd, action, reward, next_state, next_srd, terminated, truncated))
                state, srd = next_state, next_srd
        return simulated_data

    def train(self, num_episodes):
        for episode in range(num_episodes):
            self.update_model()

            simulated_data = self.generate_simulated_data(self.config["dyna"]["planning_steps"])
            for data in simulated_data:
                self.sim_replay_buffer.add(*data)

            self.ddpg_update()
            self.ea_update()

    def ddpg_update(self):
        # Sample from both real and simulated replay buffers
        real_batch = self.real_replay_buffer.sample(self.config["ddpg"]["batch_size"] // 2)
        sim_batch = self.sim_replay_buffer.sample(self.config["ddpg"]["batch_size"] // 2)

        combined_batch = [np.concatenate((real, sim), axis=0) for real, sim in zip(real_batch, sim_batch)]
        # print('combined_batch',combined_batch)
        states, srds, actions, rewards, next_states, next_srds, terminated, truncated = combined_batch
        print('next_states', next_states)


        # print('actions', actions)
        #states 和 srd 维度不同

        states = np.expand_dims(states, axis=1)
        states = states.reshape(-1, states.shape[-1])
        #print('states', states)
        # srds = srds.reshape(4, -1)
        srds = srds.reshape(srds.shape[0], -1)
        #print('srds', srds)
        states_combined = np.concatenate((states, srds), axis=1)
        print('states_combined',states_combined)
        states_combined = torch.FloatTensor(states_combined).to(self.device)

        #states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        #srds = torch.FloatTensor(srds).to(self.device)
        rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        terminated = torch.FloatTensor(terminated).reshape(-1, 1).to(self.device)
        truncated = torch.FloatTensor(truncated).reshape(-1, 1).to(self.device)

        # Update Critic Network
        with torch.no_grad():
            next_actions = self.target_actor(states_combined)
            print('next_states',next_states)
            print('next_actions', next_actions)

            target_q = self.target_critic(next_states, next_actions)
            #存疑
            #y = rewards + (1 - terminated) * self.config["ddpg"]["gamma"] * target_q
            y = rewards + (1 - terminated) * (1 - truncated) * self.config["ddpg"]["gamma"] * target_q


        # print('states_combined', states_combined)
        # print('actions', actions)
        current_q = self.critic(next_states, actions)
        critic_loss = F.mse_loss(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor Network
        # actor_loss = -self.critic(states, self.actor(states)).mean()
        actor_loss = -self.critic(next_states, self.actor(states_combined)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.target_actor, self.actor, self.config["ddpg"]["tau"])
        self.soft_update(self.target_critic, self.critic, self.config["ddpg"]["tau"])

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def ea_update(self):
        # Placeholder for the population of policies
        population = self.ea.population
        #print('population',population)

        # Evaluate fitness of the population
        fitness_scores = self.evaluate_population(population)
        print('fitness_scores',fitness_scores)

        # Select the best individuals
        selected_indices = np.argsort(fitness_scores)[-self.config["ea"]["population_size"] // 2:]
        print('selected_indices', selected_indices)
        selected_population = population[selected_indices]
        print('selected_population', selected_population)

        # Generate new population through crossover and mutation
        new_population = []
        print('len(selected_population)',len(selected_population))
        for i in range(0, len(selected_population), 2):
            #parent1, parent2 = selected_population[i], selected_population[i + 1]
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1]
            child1, child2 = self.crossover(parent1, parent2)
            new_population.append(self.mutate(child1))
            new_population.append(self.mutate(child2))

        # Replace the old population with the new one
        self.ea.population = np.array(new_population)

    def evaluate_population(self, population):
        # Evaluate the fitness of each individual in the population
        fitness_scores = []
        for policy in population:
            fitness_scores.append(self.evaluate_policy(policy))
        return fitness_scores

    def evaluate_policy(self, policy):
        # Evaluate the fitness of a single policy
        total_reward = 0
        state, srd = self.env.reset()
        terminated= False
        truncated = False
        while not terminated and not truncated:
            # 将 state 和 srd 拼接起来
            state_srd = np.concatenate((state.flatten(), srd.flatten()), axis=None)
            state_tensor = torch.FloatTensor(state_srd).unsqueeze(0).to(self.device)

            # print('state_tensor ',type(state_tensor))
            # print('policy(state_tensor).detach().cpu().numpy()[0] ', policy(state_tensor).detach().cpu().numpy()[0])

            # 使用 Actor 网络生成动作
            # action = policy(state_tensor).detach().cpu().numpy()[0]
            actor = self.convert_to_actor(policy)
            action = actor(state_tensor).detach().cpu().numpy()[0]

            state, srd, reward, terminated, truncated = self.env.step(action)
            total_reward += reward
            print('total_reward',total_reward)
        return total_reward

    def crossover(self, parent1, parent2):
        # Simple single-point crossover
        crossover_point = np.random.randint(0, len(parent1))
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2

    def mutate(self, individual):
        # Simple mutation: add Gaussian noise
        mutation_strength = self.config["ea"]["mutation_rate"]
        return individual + np.random.randn(*individual.shape) * mutation_strength


    def array_to_state_dict(self, array, state_dict_template):
        state_dict = {}
        start = 0
        for key, param in state_dict_template.items():
            num_params = param.numel()
            end = start + num_params
            state_dict[key] = torch.FloatTensor(array[start:end]).view(param.shape)
            start = end
        return state_dict

    def convert_to_actor(self, policy_array):
        actor = self.actor_model_class(self.config["ddpg"]["state_dim"],
                                       self.config["ddpg"]["action_dim"],
                                       self.config["ddpg"]["hidden_layers"]).to(self.device)
        state_dict_template = actor.state_dict()
        policy_dict = self.array_to_state_dict(policy_array, state_dict_template)
        actor.load_state_dict(policy_dict)
        return actor