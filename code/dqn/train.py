REPO_ROOT = '../../'  # autonomous_vehicles repo root.

import torch
import torch.nn.functional as F
import wandb
import tqdm
import numpy as np
import random

# Adjust relative path so that this script can find the other code modules:
import sys
sys.path.append(REPO_ROOT+'code/')

from dqn.q_network import Qnetwork
from dqn.agent import Agent
from dqn.replay_memory import SARSD, ReplayMemory
from structures import RingRoad
from animations import Animation
from learning import Game


##### HYPERPARAMETERS #####
LAYER_1_NODES = 512
LAYER_2_NODES = 256
GAMMA = 0.999
# EPS_DECAY_RATE = 0.999985 # (orange, lime green)
# EPS_DECAY_RATE = 0.999977 # (teal)
EPS_DECAY_RATE = 0.999977 # (magenta, green, light blue, pink, gold, grey, peach, dark blue)
# EPS_DECAY_RATE = 0.999700 # (brown)
# LR = 1e-4 # (green + priors)
LR = 5e-4 # (light blue, lime green, brown, grey)
# LR = 3e-4 # (peach, dark blue)
# LR = 1e-3 # (pink)
# LR = 6e-5 # (gold)
BATCH_SIZE = 256
NUM_EPISODES = 6_000
MAX_TIMESTEPS = 200
# REPLAY_MEMORY_SIZE = 20_000 # (orange)
# REPLAY_MEMORY_SIZE = 50_000 # (teal)
REPLAY_MEMORY_SIZE = 50_000 # (magenta, green, light blue, pink, gold, lime green, brown, grey, peach)
# REPLAY_MEMORY_SIZE = 80_000 # (dark blue)
# TIMESTEPS_BEFORE_TARGET_NETWORK_UPDATE = 2_000 # (orange)
# TIMESTEPS_BEFORE_TARGET_NETWORK_UPDATE = 5_000 # (teal)
# TIMESTEPS_BEFORE_TARGET_NETWORK_UPDATE = 2_000 # (magenta, light blue)
# TIMESTEPS_BEFORE_TARGET_NETWORK_UPDATE = 1_000 # (green)
TIMESTEPS_BEFORE_TARGET_NETWORK_UPDATE = 3_000 # (light blue, pink, gold, lime green, brown, grey, dark blue)
WANDB_TSTEP = 50
REWARD_SCALING = 1./100.
SEED = 1
SAVE_PATH = './saved-models/trained_model_'
#SAVE_PATH = REPO_ROOT+'models/trained_model'
#####################


def train(agent, gamma, list_of_rewards_for_all_episodes, env, tstep, wandb_tstep):
    if len(agent.replay_memory.memory) <= agent.batch_size:
        return

    if len(agent.replay_memory.memory) == agent.batch_size + 1:
        print(f"""Replay memory now has {len(agent.replay_memory.memory)} transitions,
            which is sufficient to begin training.
            """)

    transitions = agent.replay_memory.sample(agent.batch_size)

    cur_states = torch.stack([torch.Tensor(t.state) for t in transitions])
    actions_list = [t.action for t in transitions]
    rewards = torch.stack([torch.Tensor([t.reward]) for t in transitions])
    masks = torch.stack([torch.Tensor([0]) if t.done else torch.Tensor([1]) for t in transitions])
    next_states = torch.stack([torch.Tensor(t.next_state) for t in transitions])

    with torch.no_grad():
        # Use no_grad since we don't want to backprop through target network
        # Take the max since we are only interested in the q val for the action taken
        next_state_q_vals = agent.target_network(next_states).max(-1)[0] # (N, num_actions)

    agent.q_network.optimizer.zero_grad()
    q_vals = agent.q_network(cur_states) # (N, num_actions)
    actions_one_hot = F.one_hot(torch.LongTensor(actions_list), env.action_space.n)

    # Here is the TD error from the Bellman equation
    # The torch.sum() component is to select the q_vals ONLY for the action taken
    # without having to use a loop
    loss = ((rewards + gamma * masks[:, 0] * next_state_q_vals - torch.sum(q_vals * actions_one_hot, -1))**2).mean()
    if tstep % wandb_tstep == 0:
        wandb.log({'Loss': loss.detach().item(),
                   'Epsilon': agent.epsilon,
                   'Average reward over last 100 episodes': np.mean(list_of_rewards_for_all_episodes[-100:])},
                  step=agent.current_timestep_number)
    agent.current_timestep_number += 1
    loss.backward()
    agent.q_network.optimizer.step()
    return loss


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Define simulation parameters:
    road_params = {'num_avs': 2,
                   'av_even_spacing': True,
                   'num_vehicles': 10,
                   'ring_length': 100.0,
                   'starting_noise': 1.0,
                   'temporal_res': 0.5,
                   'av_activate': MAX_TIMESTEPS,
                   'seed': 286,
                   'learning_mode': True}
    past_steps = 3
    agent_commands = [-1.0, -0.1, 0.0, 0.1, 1.0] # (brown + priors)
    # agent_commands=[-5.0, -1.0, 0.0, 1.0, 5.0]  # (grey)
    # agent_commands=[-4.0, -1.0, -0.1, 0.0, 0.1, 1.0, 4.0] # (peach)
    # agent_commands=[-2.0, -0.1, 0.0, 0.1, 2.0]  # (dark blue)
    agent_commands_string = "|".join([str(comm) for comm in agent_commands])  # W&B charts can't group by lists.
    config = {
        **road_params,
        'past_steps' : past_steps,
        'agent_commands' : agent_commands,
        'agent_commands_string' : agent_commands_string,
        'LAYER_1_NODES' : LAYER_1_NODES,
        'LAYER_2_NODES' : LAYER_2_NODES,
        'GAMMA' : GAMMA,
        'EPS_DECAY_RATE' : EPS_DECAY_RATE,
        'LR' : LR,
        'BATCH_SIZE' : BATCH_SIZE,
        'NUM_EPISODES' : NUM_EPISODES,
        'MAX_TIMESTEPS' : MAX_TIMESTEPS,
        'REPLAY_MEMORY_SIZE' : REPLAY_MEMORY_SIZE,
        'TIMESTEPS_BEFORE_TARGET_NETWORK_UPDATE' : TIMESTEPS_BEFORE_TARGET_NETWORK_UPDATE,
        'REWARD_SCALING' : REWARD_SCALING,
        'SEED' : SEED,
    }

    # Initialize wandb
    wandb.init(project="cs286", name="tuning_dqn-avs", config=config)

    # Define a ring road environment:
    road = RingRoad(**road_params)
    env = Game(road = road,
               agent_commands = agent_commands,
               past_steps = past_steps,
               max_seconds = None)

    replay_memory = ReplayMemory(memory_size=REPLAY_MEMORY_SIZE)
    q_network = Qnetwork(
        observation_shape=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        layer_1_nodes=LAYER_1_NODES,
        layer_2_nodes=LAYER_2_NODES,
        lr=LR
    )
    target_network = Qnetwork(
        observation_shape=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        layer_1_nodes=LAYER_1_NODES,
        layer_2_nodes=LAYER_2_NODES,
        lr=LR
    )
    agent = Agent(
        q_network=q_network,
        target_network=target_network,
        replay_memory=replay_memory,
        batch_size=BATCH_SIZE,
        decay_rate=EPS_DECAY_RATE
    )

    # full_path = f"{SAVE_PATH}{wandb.run.id}_q_network.pt"
    # torch.save(agent.q_network, full_path)
    # wandb.save(full_path)  # Save full model (not just state) to wandb.

    list_of_rewards_for_all_episodes = []

    tq = tqdm.tqdm()

    for i_episode in range(NUM_EPISODES):
        observation = env.reset()
        ith_episode_rewards = []

        for t in range(MAX_TIMESTEPS):

            action = agent.get_e_greedy_action(observation=torch.Tensor(observation), env=env)
            next_observation, reward, done, _ = env.step(action)
            # Append reward before normalizing. Allows for better tracking (we can see actual score) while
            # also making it easier for network to predict Q-val (for 200-timestep long episode, Q-net
            # should end up predicting "2" rather than "200")
            ith_episode_rewards.append(reward)
            reward = reward * REWARD_SCALING

            sarsd = SARSD(state=observation,
                          action=action,
                          reward=reward,
                          next_state=next_observation,
                          done=done)
            agent.replay_memory.add(sarsd=sarsd)

            train(agent=agent,
                  gamma=GAMMA,
                  list_of_rewards_for_all_episodes=list_of_rewards_for_all_episodes,
                  env=env,
                  tstep=agent.current_timestep_number,
                  wandb_tstep=WANDB_TSTEP)

            observation = next_observation

            if agent.current_timestep_number % TIMESTEPS_BEFORE_TARGET_NETWORK_UPDATE == 0:
                tq.update(1)
                print("\nUpdating target network")
                print(f"\tOn episode {i_episode + 1}")
                print(f"\tOn overall timestep {agent.current_timestep_number}")
                print(f"\tReplay memory now has {len(agent.replay_memory.memory)} transitions")
                agent.target_network.load_state_dict(agent.q_network.state_dict())
                full_path = "{}{}_{}.pt".format(SAVE_PATH, wandb.run.id, i_episode+1)
                torch.save(agent.q_network.state_dict(), full_path)
                print(f'\tModel saved to {full_path}')
                wandb.save(full_path)  # Save model to wandb.

            if done:
                # print("Episode finished after {} timesteps".format(t + 1))
                break

        # Add episode rewards:
        list_of_rewards_for_all_episodes.append(np.sum(ith_episode_rewards))

    print("\n****Training Complete****\n")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    main()

