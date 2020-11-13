"""
Harvard CS286 Final Project, Fall 2020.
Data structures for simulation of autonomous vehicle controllers presented by Delle Monarche et al. (2019):
'Feedback Control Algorithms for the Dissipation of Traffic Waves with Autonomous Vehicles'
https://doi.org/10.1007/978-3-030-25446-9_12
"""

import numpy as np


class Position:

    TOL = 1e-9  # Tolerance for floating point comparisons.

    def __init__(self, x, L):
        """
        Representation of a position x on ring road of length L, with modular arithmetic.
        Positions are always positive, as stored as a float in the interval [ 0.0, L ).
        """
        if hasattr(x, 'env'): # Check if the input is alread a Vehicle.
            assert L == x.env.length, "Cannot mix positions with different L values."
            x = x.x
        elif hasattr(x, 'L'): # Check if the input is a Position.
            assert L == x.L, "Cannot mix positions with different L values."
            x = x.x
        assert L>0, "Must have non-negative length."
        self._L = L
        self._x = 1.0*( x % L )

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = 1.0*( x % self.L )

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, L):
        raise AttributeError("The length property is immutable.")
    
    def distance_to(self, other, reverse=False):
        """
        Return the distance from self to other (i.e. by how much does other lead self?)
        If reverse=True, returns the distance from other to self, travelling in direction of the road.
        Distances are always positive and are measured in the direction of traffic.
        """
        # Convert to Position (should work event if other is already a Position):
        other = Position(other, self.L)
        # Apply reverse case:
        if reverse:
            return other.distance_to(self)
        # Get positions as numeric values:
        s = self.x
        o = other.x
        # Get difference:
        dist = (o - s) % self.L
        return dist

    def __eq__(self, other):
        """
        Check if two positions are equal (within a set tolerance).
        """
        s = self.x
        o = Position(x=other, L=self.L).x  # Convert other to a Position if needed.
        return np.isclose(s,o, atol=Position.TOL)

    def __add__(self, other):
        try:
            o = float(other)
        except:
            raise ValueError("Only numerical values can be added to Position objects.")
        s = self.x
        new_x = (s + o) % self.L
        return Position(x=new_x, L=self.L)
    def __radd__(self, other):
        raise NotImplementedError("Reverse operations are ambigous for Position objects -- try Position + scalar instead.")

    def __sub__(self, other):
        try:
            o = float(other)
        except:
            raise ValueError("Only numerical values can be subtracted from Position objects -- try pos1.to_distance(pos2).")
        s = self.x
        new_x = (s - o) % self.L
        return Position(x=new_x, L=self.L)
    def __rsub__(self, other):
        raise NotImplementedError("Reverse operations are ambigous for Position objects -- try Position - scalar instead.")

    def __repr__(self):
        return "Position(x={}, L={})".format(self.x, self.L)

class RingRoad:

    def __init__(self, num_vehicles=22, ring_length=260.0, seed=None):
        
        # Store properties:
        self.num_vehicles = num_vehicles  # Total number of vehicles (including A.V.).
        self.ring_length = ring_length  # Length of ring road (meters).
        self.vehicle_length = 4.5  # Length of vehicles (meters).
        self.safe_distance = 4  # Safe distance between vehicles (meters).
        self.max_speed = 9.75  # Max speed (meters/second).
        self.temporal_res = 0.001  # Time between updates (in seconds).
        self.spatial_res = None
        self.traffic_a = 0.5  # Coefficient for the FTL model (meters/second).
        self.traffic_b = 20  # Coefficient for the Bando-OV model (1/second).
        self.seed = seed

        # Store state information:
        self.state = None
        self.history = dict()  # History of states, keyed by time.

        # Initialize:
        self.random = np.random.RandomState(seed)
        self.reset_state()

    def reset_state(self):
        assert self.num_vehicles >= 2, "Need at least 1 human and 1 robot."
        d_start = self.ring_length / self.num_vehicles
        robot = Robot(
            env=self,
            active_controller = BandoFTL(env=self, a=self.traffic_a, b=self.traffic_b),
            passive_controller = PID(env=self),
            init_pos = 0.0,
            init_vel = 0.0,
            init_acc = 0.0,
            length = self.vehicle_length,
        )
        robot.active = False
        vehicles = [robot]
        for index in range(1,self.num_vehicles):
            human = Human(
                env=self,
                controller = BandoFTL(env=self, a=self.traffic_a, b=self.traffic_b),
                init_pos = index * d_start,
                init_vel = 0.0,
                init_acc = 0.0,
                length = self.vehicle_length,
            )
            vehicles.append(human)
        self.state = {
            'time' : 0.0,
            'vehicles' : vehicles,  # List of vehicles in 0,...,(N-1) index order, with A.V. at index 0.
            'all_vehicles' : set(vehicles),  # Set of all vehicles that have been part of the simulation.
        }

    def copy_state(self):
        state = self.state.copy()
        state['vehicles'] = state['vehicles'].copy()
        state['all_vehicles'] = state['all_vehicles'].copy()
        return state

    def get_vehicle_index(self, vehicle):
        """
        Returns the index (or None) of a given Vehicle.
        """
        index = None
        for i,v in enumerate(self.state['vehicles']):
            if v.id == vehicle_id:
                index = i
                break
        return index

    def get_lead_index(self, vehicle):
        """
        Returns the index of the Vehicle that leads a given Vehicle.
        """
        if isinstance(vehicle, int):
            this_index = vehicle
        else:
            this_index = self.get_vehicle_index(vehicle)
        if this_index is None:
            raise RuntimeError("Vehicle not found: {}".format(vehicle))
        if self.N < 2:
            raise RuntimeError("Vehicle is alone on the road: {}".format(vehicle))
        lead_index = (this_index + 1) % self.N
        return lead_index

    def get_lead_vehicle(self, vehicle):
        lead_index = self.get_lead_index(vehicle)
        lead_vehicle = self.state['vehicles'][lead_index]
        return lead_vehicle

    @property
    def vehicles(self):
        return self.state['vehicles'].copy()

    @property
    def t(self):
        return self.state['time']

    @property
    def dt(self):
        return self.temporal_res

    @property
    def L(self):
        return self.ring_length

    @property
    def l_v(self):
        return self.vehicle_length

    @property
    def N(self):
        return len(self.state['vehicles'])

    def __repr__(self):
        return "RingRoad(num_vehicles={}, ring_length={}, seed={})".format(self.num_vehicles, self.ring_length, self.seed)

    def __str__(self):
        s = ""
        s += self.__repr__() + " at time {}:".format(self.t) + "\n"
        for index,vehicle in enumerate(self.state['vehicles']):
            s += "    [{}] ".format(index) + vehicle.__str__() + "\n"
        return s

class Vehicle:

    all_vehicles = []

    def __init__(self, env, controller, init_pos=0.0, init_vel=0.0, init_acc=0.0, length=4.5):

        # Generate unique ID and add to master list:
        self.id = len(Vehicle.all_vehicles)
        Vehicle.all_vehicles.append(self)

        # Store properties:
        self.env = env
        self.controller = controller
        self.init_pos = init_pos
        self.init_vel = init_vel
        self.init_acc = init_acc
        self.length = length
        self.type = None  # 'robot' or 'human'

        # Store state information:
        self.state = None
        self.history = dict()  # History of states, keyed by time.

        # Initialize:
        self.reset_state()

    def reset_state(self):
        self.state = {
            'pos' : Position(x=self.init_pos, L=self.env.L),
            'vel' : self.init_vel,
            'acc' : self.init_acc,
            'controller_type' : self.controller.type,
        }

    def copy_state(self):
        return self.state.copy()

    def archive_state(self):
        self.history[self.env.t] = self.copy_state()

    @property
    def x(self):
        return self.state['pos'].x

    @property
    def l(self):
        return self.length

class Human(Vehicle):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = 'human'

    def __str__(self):
        s = "Human driver at position {} with velocity {} and acceleration {}.".format(
            self.state['pos'].x, self.state['vel'], self.state['acc'],
        )
        return s

class Robot(Vehicle):

    def __init__(self, env, active_controller, passive_controller, *args, **kwargs):
        
        # Initialize Vehicle:
        super().__init__(env=env, controller=active_controller, *args, **kwargs)

        # Store additional properties:
        self.type = 'robot'
        self.active_controller = active_controller
        self.passive_controller = passive_controller
        
        # Initialize:
        self.reset_state()
    
    def reset_state(self):
        super().reset_state()
        self.state['active'] = True  # Flag to determine autonomous control.

    @property
    def active(self):
        return self.state['active']

    @active.setter
    def active(self, active):
        self.state['active'] = active
        if active:
            self.controller = self.active_controller
        else:
            self.controller = self.passive_controller

    def __str__(self):
        s = "AV ({}) at position {} with velocity {} and acceleration {}.".format(
            'active' if self.active else 'passive',
            self.state['pos'].x, self.state['vel'], self.state['acc'],
        )
        return s

class Controller:

    def __init__(self, env):
        self.env = env
        self.type = None  # 'BandoFTL' or 'PID', etc

class BandoFTL(Controller):

    def __init__(self, env, a, b):
        super().__init__(env)
        self.type = 'BandoFTL'
        self.a = a
        self.b = b

class PID(Controller):

    def __init__(self, env):
        super().__init__(env)
        self.type = 'PID'
