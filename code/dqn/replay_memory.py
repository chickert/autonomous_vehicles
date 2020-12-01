from collections import namedtuple, deque
import random
import numpy as np

# could also use randint + numpy to random sample (perhaps time is scaling linearly with rep_mem size)
# Plot time to verify and check

SARSD = namedtuple('SARSD', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory:
    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)

    def add(self, sarsd):
        self.memory.append(sarsd)

    def sample(self, batch_size):
        # assert len(self.memory) > batch_size, "Batch size is greater than memory size"
        return random.sample(self.memory, batch_size)
