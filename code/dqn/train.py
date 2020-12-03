REPO_ROOT = '../../'  # autonomous_vehicles repo root.

import torch
import torch.nn.functional as F
import wandb
import tqdm
import numpy as np
import random
import json

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
EPS_DECAY_RATE = 0.999977 # (purple)
# EPS_DECAY_RATE = 0.999986 # (big run)
LR = 5e-4 # (purple)
# LR = 3e-4   # (big run)
BATCH_SIZE = 256
NUM_EPISODES = 200_000
MAX_TIMESTEPS = 200
REPLAY_MEMORY_SIZE = 50_000 # (purple)
# REPLAY_MEMORY_SIZE = 250_000 # (big run)
TIMESTEPS_BEFORE_TARGET_NETWORK_UPDATE = 3_000 # (purple)
# TIMESTEPS_BEFORE_TARGET_NETWORK_UPDATE = 6_000 # (big run)
INIT_REPLAY_MEMORY = REPLAY_MEMORY_SIZE  # purple run: Fill full memory with random actions before learning starts.
EPS_DECAY_START = INIT_REPLAY_MEMORY  # purple run: Don't start epsilon decay until learning starts.
WANDB_TSTEP = 1_000
REWARD_SCALING = 1./100.
SEED = 1
SAVE_PATH = './saved-models/trained_model_'
#SAVE_PATH = REPO_ROOT+'models/trained_model'
TUNING_DESCRIPTION = "Fill replay memory before training starts."
#####################


def train(agent, gamma, list_of_rewards_for_all_episodes, env, wandb_tstep):

    if len(agent.replay_memory.memory) <= agent.batch_size:
        return
    
    # if len(agent.replay_memory.memory) < REPLAY_MEMORY_SIZE:
    #     if len(agent.replay_memory.memory) % 5_000 == 0:
    #         print(f'Replay Memory now has {len(agent.replay_memory.memory)} transitions')
    #     return

    # if len(agent.replay_memory.memory) == agent.batch_size + 1:
    # if len(agent.replay_memory.memory) == REPLAY_MEMORY_SIZE:
    #     print(f"""Replay memory now has {len(agent.replay_memory.memory)} transitions,
    #         which is sufficient to begin training.
    #         """)

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
    if agent.current_timestep_number % wandb_tstep == 0:
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
                   'av_activate': 0,
                   'seed': 286,
                   'learning_mode': True}
    game_params = {
        'crowding_penalty' : 0,
        'crash_penalty' : 10,
        'past_steps' : 3,
        'agent_commands' : [-1.0, -0.1, 0.0, 0.1, 1.0], # (purple + big run)
    }

    # Define a ring road environment:
    road = RingRoad(**road_params)
    env = Game(road = road, max_seconds = None, **game_params)

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
        decay_rate=EPS_DECAY_RATE,
        decay_starts_at=EPS_DECAY_START,
    )

    # Collect network and training parameters in dictionaries (for archiving):
    network_architecture = {
        'observation_shape' : env.observation_space.shape[0],
        'num_actions' : env.action_space.n,
        'layer_1_nodes' : LAYER_1_NODES,
        'layer_2_nodes' : LAYER_2_NODES,
        'lr' : LR,
    }
    training_params = {
        'LAYER_1_NODES' : LAYER_1_NODES,
        'LAYER_2_NODES' : LAYER_2_NODES,
        'GAMMA' : GAMMA,
        'EPS_DECAY_RATE' : EPS_DECAY_RATE,
        'LR' : LR,
        'BATCH_SIZE' : BATCH_SIZE,
        'NUM_EPISODES' : NUM_EPISODES,
        'MAX_TIMESTEPS' : MAX_TIMESTEPS,
        'REPLAY_MEMORY_SIZE' : REPLAY_MEMORY_SIZE,
        'INIT_REPLAY_MEMORY' : INIT_REPLAY_MEMORY,
        'TIMESTEPS_BEFORE_TARGET_NETWORK_UPDATE' : TIMESTEPS_BEFORE_TARGET_NETWORK_UPDATE,
        'REWARD_SCALING' : REWARD_SCALING,
        'SEED' : SEED,
    }

    # Merge all the parameters into a single dictionary (for uploading to wandb):
    config = dict()
    config.update(road_params)
    config.update(network_architecture)
    config.update(game_params)
    config.update(training_params)

    # Initialize wandb
    wandb.init(
        project="cs286", name="tuning_dqn-avs",
        config=config,  # Upload hyperparameters (just of archive purposes -- they don't actually control anything in wandb)
        notes = TUNING_DESCRIPTION,  # Description of run.
    )

    # Save configuration info to a json file (for replay):
    replay_info = {
        'road_info' : road_params,
        'game_info' : game_params,
        'network_info' : network_architecture,
    }
    replay_path = f"{SAVE_PATH}{wandb.run.id}_replay_info.json"
    with open(replay_path, 'w') as f:
        json.dump(replay_info, f, indent=4)
    wandb.save(replay_path)  # Upload to wandb for archiving.

    # Save full model (not just state) locally:
    full_path = f"{SAVE_PATH}{wandb.run.id}_q_network.pt"
    torch.save(agent.q_network, full_path)
    # Upload to wandb for archiving:
    wandb.save(full_path)

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
                  wandb_tstep=WANDB_TSTEP)

            observation = next_observation

            if agent.current_timestep_number % TIMESTEPS_BEFORE_TARGET_NETWORK_UPDATE == 0:
                tq.update(1)
                print("\nUpdating target network")
                print(f"\tOn episode {i_episode + 1}")
                print(f"\tOn overall timestep {agent.current_timestep_number}")
                print(f"\tReplay memory now has {len(agent.replay_memory.memory)} transitions")
                agent.target_network.load_state_dict(agent.q_network.state_dict())
                # Save model state locally:
                full_path = "{}{}_{}.pt".format(SAVE_PATH, wandb.run.id, i_episode+1)
                torch.save(agent.q_network.state_dict(), full_path)
                print(f'\tModel saved to {full_path}')
                # Upload model state to wandb:
                wandb.save(full_path)

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

