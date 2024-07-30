# models/actor.py

import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        super(Actor, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())  # Assuming action space is [-1, 1]
        self.network = nn.Sequential(*layers)
        #self.state_dict = self.state_dict()

    def forward(self, state):
        return self.network(state)

