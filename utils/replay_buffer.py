import random
import numpy as np
import pickle

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, srd, action, reward, next_state, next_srd, terminated, truncated,):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, srd, action, reward, next_state, next_srd, terminated, truncated)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        #print('batch_size',batch_size)
        batch = random.sample(self.buffer, batch_size)
        #batch = self.process_batch(batch)
        #batch = self.check_data(batch)
        #print('batch',batch)
        state, srd, action, reward, next_state, next_srd, terminated, truncated = map(np.stack, zip(*batch))
        # print('sample :',state, srd, action, reward, next_state, terminated, truncated)

        return state, srd, action, reward, next_state, next_srd, terminated, truncated

    def process_batch(self, batch):
        processed_batch = []
        for item in batch:
            if isinstance(item, tuple):
                # 检查第一个元素是否为二维数组，第二个元素是否为字典
                if isinstance(item[0], np.ndarray) and item[0].ndim == 2 and isinstance(item[1], dict):
                    # 将第一个元素变成一维数组，并等于二维数组的第一项
                    new_first_element = item[0][0]
                    # 删除第二项，构建新的tuple
                    new_item = (new_first_element,) + item[2:]
                    processed_batch.append(new_item)
                else:
                    processed_batch.append(item)
            else:
                processed_batch.append(item)

        return processed_batch

    def __len__(self):
        return len(self.buffer)

    def check_data(self,batch):
        for index, i in enumerate(batch):
            if len(i) == 8:
                #print('get!/n/n')
                new_i0 = i[0][0]
                new_item = (new_i0,) + i[2:]
                #print('\nnew_item', new_item)
                batch[index] = new_item
                #print('len(batch[index])', len(batch[index]))
        return batch

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.buffer, f)
        print(f"Replay buffer saved to {file_path}")

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.buffer = pickle.load(f)
        self.position = len(self.buffer) % self.capacity
        print(f"Replay buffer loaded from {file_path}")