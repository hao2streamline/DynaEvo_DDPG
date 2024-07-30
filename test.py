import numpy as np
from utils.replay_buffer import ReplayBuffer
import yaml

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

config = load_config('config/config.yaml')
replay_buffer = ReplayBuffer(config["replay_buffer"]["capacity"])
replay_buffer.load('./saved_models/replay_buffer.pkl')

print(replay_buffer)
