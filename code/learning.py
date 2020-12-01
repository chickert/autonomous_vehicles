"""
Wrapper classes for formulating a reinforcement learning problem.
"""

import itertools

import numpy as np

from structures import RingRoad

# Hide warnings about safe distance violation (still in development):
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Game:

    def __init__(self, road, past_steps=3, agent_commands=[-2,-1,0,1,2], max_seconds=None):
        
        # Bind RingRoad object to Game:
        self.road = road
        assert self.road.learning_mode, "Requires a RingRoad with learning_mode=True."
        if self.road.av_activate != 0:
            print(f"WARNING: av_activate>0, i.e. AVs are not in autonomous mode until t={self.road.av_activate}.")

        # Game properies:
        self.max_seconds = max_seconds
        self.max_steps = None if max_seconds is None else int(np.ciel( self.max_seconds / self.road.dt ))
        self.observation_space = ObservationSpace(self)  # Simple wrapper object defined below.
        self.action_space = ActionSpace(self)  # Simple wrapper object defined below.
        self.agent_commands = agent_commands  # List of valid commands for each A.V.
        self.num_agents = self.road.num_avs  # Number of A.V.s being controlled.
        self.past_steps = past_steps  # How many steps into the past to include in observation.

        # Game state:
        self.done = False
        self.step_count = 0

        # Initialize:
        self.reset()

    def reset(self):
        """
        Reset episode state.
        """
        # Reset game state:
        self.done = False
        self.step_count = 0
        # Reset the ring road:
        self.road.reset_road()
        # Build a dummy observation (only its shape matters):
        dummy_observation = self.build_observation()
        self.observation_space = np.zeros(dummy_observation.shape)

    def build_observation(self):
        """
        Build an observation vector from the ring road's current state.
        """
        past_steps = self.past_steps
        road_step = self.road.step
        steps = range( max(0,road_step-past_steps), road_step )
        # Get position history for all vehicles:
        table = self.road.get_vehicle_pos_table(steps=steps)
        # Convert to array and pad it with zeros if needed:
        array = array = table.to_numpy()  # Cols are vehicles, rows are time steps.
        if array.shape[0] < past_steps:
            padding = np.zeros( (past_steps-array.shape[0], array.shape[1]) )
            array = np.vstack([padding,array])
        # Flatten observations to a vector:
        vector = array.flatten()
        # Normalize:
        vector = vector / self.road.L
        return vector

    def reward(self):
        """
        Calculate a reward value from the ring road's current state.
        """
        # Get mean velocity:
        mean_velocity = np.mean([vehicle.vel for vehicle in self.road.vehicles])
        # if self.road.max_speed:  # Normalize by max velocity, if applicable.
        #     mean_velocity = mean_velocity / self.road.max_speed

        # Check for crashes and crowding:
        crashes = 0
        crowding = 0
        for vehicle in self.road.vehicles:
            if self.road.check_crash(vehicle=vehicle, raise_error=False):
                crashes += 1  # Check if the vehicle collided with its lead.
            if self.road.check_crowding(vehicle=vehicle, raise_warning=False, pct=0.1):
                crowding += 1  # Check if the vehicle left less than 10% of its lead's safety buffer.

        # Calculate total reward:
        reward = mean_velocity - 1 * crowding - 10 * crashes

        return reward


    def step(self, action):

        assert not self.done, "Episode is already done."

        # Decode action into a command for each A.V:
        av_commands = self.action_space.decode(action)

        # Apply A.V. commands (if A.V.s are in autonomous mode):
        if self.road.t >= self.road.av_activate:
            for av_index, av_command in zip(self.road.av_indices, av_commands):
                av = self.road.vehicles[av_index]
                av.controller.command = av_command

        # Run step:
        self.road.run_step()

        next_observation = self.build_observation()
        reward = self.reward()
        info = dict()
        done = False

        # End episode of max steps is reached:
        if (self.max_steps is not None) and (self.step_count >= self.max_steps):
            done = True

        # End episode if there is a crash:
        if self.road.check_crash(raise_error=False):
            done = True

        # Update game state:
        self.done = done

        return next_observation, reward, done, info


class ObservationSpace:
    """
    A wraper that calculates and caches the shape of the observation space
    by getting a dummy observation from the environment. The lazy evaluation
    ensures that it is not called until after the Game has been initialized.
    """

    def __init__(self, game):
        self.game = game
        self._shape = None
    
    @property
    def shape(self):
        if self._shape is None:
            dummy_observation = self.game.build_observation()
            self._shape = dummy_observation.shape
        return self._shape


class ActionSpace:
    """
    A wraper that stores the state of the action space.
    """

    def __init__(self, game):
        self.game = game
        self.actions = None

    def _check_build(self):
        if self.actions is None:
            self.actions = itertools.product(self.game.agent_commands, repeat=self.game.num_agents)
            self.actions = list(self.actions)  # List of tuples.

    def __call__(self):
        self._check_build()
        return self.actions
    
    @property
    def n(self):
        self._check_build()
        return len(self.actions)

    def encode(self, agent_commands):
        """
        Convert a tuple of agent commands into a positional index in the RL action space.
        """
        self._check_build()
        return self.actions.index(tuple(agent_commands))

    def decode(self, action_index):
        """
        Convert an index in the RL action space into a tuple of agent commands.
        """
        self._check_build()
        return self.actions[action_index]
