# models/mlp.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib


class MLPModel(nn.Module):
    def __init__(self, config):
        super(MLPModel, self).__init__()
        layers = []
        input_dim = config["input_dim"]
        for hidden_dim in config["hidden_layers"]:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, config["output_dim"]))
        self.network = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.parameters(), lr=config["learning_rate"])
        self.scaler = StandardScaler()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.network(x)

    def predict(self, state, surrounding_vehicles, action):
        # state = state.flatten()
        # surrounding_vehicles = surrounding_vehicles.flatten()
        #
        # inputs = np.hstack([state, action, surrounding_vehicles])
        # inputs = self.scaler.transform([inputs])
        # inputs = torch.FloatTensor(inputs).to(self.device)
        #
        # self.network.eval()
        # with torch.no_grad():
        #     prediction = self.network(inputs)
        # return prediction.cpu().numpy()

        state = state.flatten()
        surrounding_vehicles = surrounding_vehicles.flatten()

        inputs = np.hstack([state, action, surrounding_vehicles])
        inputs = self.scaler.transform([inputs])
        inputs = torch.FloatTensor(inputs).to(self.device)

        self.network.eval()
        with torch.no_grad():
            prediction = self.network(inputs)

        # 拆分预测结果，假设输出为 [next_state, next_srd, reward]
        next_state_dim =4
        next_srd_dim = 12
        next_state = prediction.cpu().numpy()[0][:4]
        next_srd = prediction.cpu().numpy()[0][4:16].reshape(3, 4)
        reward = prediction.cpu().numpy()[0][-1]
        # print('next_state', next_state)
        # print('next_srd', next_srd)
        # print('reward', reward)

        return next_state, next_srd, reward

    def train_model(self, states, surrounding_vehicles, actions,  next_states_rewards, epochs=10):
        # print(states)
        states = np.array([s.flatten() for s in states])
        actions = np.array(actions)
        surrounding_vehicles = np.array([sv.flatten() for sv in surrounding_vehicles])
        next_states_rewards = np.array(next_states_rewards)

        # print(states)
        inputs = np.hstack([states, actions, surrounding_vehicles])
        outputs = next_states_rewards

        inputs = self.scaler.fit_transform(inputs)

        inputs = torch.FloatTensor(inputs).to(self.device)
        outputs = torch.FloatTensor(outputs).to(self.device)
        # print('inputs',inputs)
        # print('outputs', outputs)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.network.parameters(), lr=self.optimizer.defaults['lr'])

        for epoch in range(epochs):
            print('===============================================')
            print('epoch:',epoch)
            self.network.train()
            optimizer.zero_grad()
            predictions = self.network(inputs)
            loss = criterion(predictions, outputs)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    def update(self, states, actions, surrounding_vehicles, next_states_rewards):
        states = np.array([s.flatten() for s in states])
        actions = np.array(actions)
        surrounding_vehicles = np.array([sv.flatten() for sv in surrounding_vehicles])
        next_states_rewards = np.array(next_states_rewards)

        inputs = np.hstack([states, actions, surrounding_vehicles])
        #outputs = next_states_rewards
        outputs = np.hstack([next_states_rewards, surrounding_vehicles])

        inputs = self.scaler.fit_transform(inputs)

        inputs = torch.FloatTensor(inputs).to(self.device)
        outputs = torch.FloatTensor(outputs).to(self.device)

        criterion = nn.MSELoss()

        self.network.train()
        self.optimizer.zero_grad()
        predictions = self.network(inputs)
        loss = criterion(predictions, outputs)
        loss.backward()
        self.optimizer.step()


    def save_model(self, path):
        torch.save(self.network.state_dict(), path)
        joblib.dump(self.scaler, path + '.pkl')

    def load_model(self, path):
        self.network.load_state_dict(torch.load(path))
        self.scaler = joblib.load(path + '.pkl')
        self.network.eval()
