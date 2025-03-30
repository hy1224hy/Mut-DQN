from collections import deque
import random
from constants import *
import numpy as np


class ExpPool:
    def __init__(self, state_size, action_size):
        self.pool = deque(maxlen=MAX_EXP)
        self.state_size = state_size
        self.action_size = int(action_size/2)

    def sample(self):
        exps = random.sample(self.pool, BATCH_SIZE)
        state_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []
        next_state_batch = []
        for e in exps:
            state_batch.append(e[0])
            action_batch.append(e[1])
            reward_batch.append(e[2])
            done_batch.append(e[3])
            next_state_batch.append(e[4])
        state_batch = np.array(state_batch).reshape((BATCH_SIZE, self.state_size)).astype(np.float32)
        action_batch = np.array(action_batch).reshape((BATCH_SIZE, self.action_size)).astype(np.int64)
        reward_batch = np.array(reward_batch).reshape((BATCH_SIZE, 1)).astype(np.float32)
        done_batch = np.array(done_batch).reshape((BATCH_SIZE, 1))
        next_state_batch = np.array(next_state_batch).reshape((BATCH_SIZE, self.state_size)).astype(np.float32)
        return state_batch, action_batch, reward_batch, done_batch, next_state_batch
    
    def append(self, x):
        self.pool.append(x)
