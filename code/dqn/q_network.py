import torch.nn as nn
import torch.optim as optim


class Qnetwork(nn.Module):
    def __init__(self, observation_shape,
                 num_actions,
                 layer_1_nodes,
                 layer_2_nodes,
                 lr):
        super(Qnetwork, self).__init__()
        self.fc1 = nn.Linear(observation_shape, layer_1_nodes)
        self.fc2 = nn.Linear(layer_1_nodes, layer_2_nodes)
        self.fc3 = nn.Linear(layer_2_nodes, num_actions)
        self.deepnet = nn.Sequential(
            nn.Linear(observation_shape, layer_1_nodes),
            nn.ReLU(),
            nn.Linear(layer_1_nodes, layer_2_nodes),
            nn.ReLU(),
            nn.Linear(layer_2_nodes, num_actions)
        )
        self.optimizer = optim.Adam(self.deepnet.parameters(), lr=lr)

    def forward(self, state):
        # Takes in state and outputs q_vals for all actions
        return self.deepnet(state)
