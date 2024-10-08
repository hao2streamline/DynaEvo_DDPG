import torch
import torch.nn as nn
import torch.optim as optim
import copy

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        super(Critic, self).__init__()
        layers = []
        input_dim = state_dim + action_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.target = copy.deepcopy(self)  # Target network for soft updates

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

