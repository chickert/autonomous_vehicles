import torch
import random
import numpy as np


class Agent:
    def __init__(self,
                 q_network,
                 target_network,
                 replay_memory,
                 batch_size,
                 decay_rate,
                 decay_starts_at,
                 ):
        self.q_network = q_network
        self.target_network = target_network
        self.replay_memory = replay_memory
        self.batch_size = batch_size
        self.epsilon = 1
        self.eps_decay_rate = decay_rate
        self.current_timestep_number = 1
        self.decay_starts_at = decay_starts_at  # (usually replay memory size; point at which epsilon decay begins)

    def get_e_greedy_action(self, observation, env):
        if self.current_timestep_number > 1:  # arbitrary number to prevent epsilon decay beginning until random initialization of replay memory
            self.epsilon = np.clip(self.epsilon * self.eps_decay_rate, a_min=0.05, a_max=None)
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
