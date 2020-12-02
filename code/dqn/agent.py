import torch
import random
import numpy as np


class Agent:
    def __init__(self,
                 q_network,
                 target_network,
                 replay_memory,
                 batch_size,
                 decay_rate):
        self.q_network = q_network
        self.target_network = target_network
        self.replay_memory = replay_memory
        self.batch_size = batch_size
        self.epsilon = 1
        self.eps_decay_rate = decay_rate
        self.current_timestep_number = 1

    def get_e_greedy_action(self, observation, env):
        # self.epsilon = self.epsilon * self.eps_decay_rate
        self.epsilon = np.clip(self.epsilon * self.eps_decay_rate, a_min=0.1, a_max=None)
        if random.random() < self.epsilon:
            # Take random action
            action = env.action_space.sample()
        else:
            # Take action selected by deep q-network
            with torch.no_grad():
                q_vals = self.q_network(observation)
                # assert len(q_vals) == 2, "Action selection issue; model outputs q_vals in wrong format"
                action = q_vals.argmax().item()
        return action
