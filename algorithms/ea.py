import numpy as np
import torch
from models.actor import Actor

class EA:
    def __init__(self, actor_model_class, config, device):
        self.actor_model_class = actor_model_class
        self.config = config
        self.device = device
        self.population = self._initialize_population()

    def _initialize_population(self):
        population = []
        for _ in range(self.config["ea"]["population_size"]):
            actor = self.actor_model_class(self.config["ddpg"]["state_dim"],
                                           self.config["ddpg"]["action_dim"],
                                           self.config["ddpg"]["hidden_layers"]).to(self.device)
            state_dict = actor.state_dict()
            # Convert state_dict to numpy array
            state_dict_array = self.state_dict_to_array(state_dict)
            population.append(state_dict_array)
        return np.array(population)
    #需要修改，这里要根据已经学习到的actor网络获取state_dict，产生的population类型应为state_dict

    def state_dict_to_array(self, state_dict):
        array = []
        for key, param in state_dict.items():
            array.extend(param.cpu().numpy().flatten())
        return np.array(array)

    # Add more methods for selection, mutation, crossover, etc.
