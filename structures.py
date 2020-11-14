"""
Harvard CS286 Final Project, Fall 2020.
Data structures for simulation of autonomous vehicle controllers presented by Delle Monarche et al. (2019):
'Feedback Control Algorithms for the Dissipation of Traffic Waves with Autonomous Vehicles'
https://doi.org/10.1007/978-3-030-25446-9_12
"""

import numpy as np
import pandas as pd

import plotly.io
import matplotlib.animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


from controllers import Controller, BandoFTL, PID


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

    def __repr__(self):
        return "Position(x={}, L={})".format(self.x, self.L)

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

class RingRoad:

    def __init__(self, num_vehicles=22, ring_length=260.0, seed=None):
        
        # Store properties:
        self.num_vehicles = num_vehicles  # Total number of vehicles (including A.V.).
        self.ring_length = ring_length  # Length of ring road (meters).
        self.vehicle_length = 4.5  # Length of vehicles (meters).
        self.safe_distance = 4  # Safe distance between vehicles (meters).
        self.max_speed = 9.75  # Max speed (meters/second).
        self.temporal_res = 0.1  # Time between updates (in seconds).
        self.spatial_res = None
        self.traffic_a = 0.5  # Coefficient for the FTL model (meters/second).
        self.traffic_b = 20  # Coefficient for the Bando-OV model (1/second).
        self.seed = seed

        # Store state information:
        self.state = None
        self.history = dict()  # History of states, keyed by time step.
        self.all_vehicles = set()  # Set of all vehicles that were on the road at any point in time.

        # Initialize:
        self.random = np.random.RandomState(seed)
        self.reset_state()
        self.archive_state()

    @property
    def vehicles(self):
        return self.state['vehicles'].copy()

    @property
    def step(self):
        return self.state['step']

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
        s += self.__repr__() + " at step {} (t={}):".format(self.step, self.t) + "\n"
        for index,vehicle in enumerate(self.state['vehicles']):
            s += "    [{}] ".format(index) + vehicle.__str__() + "\n"

    def reset_state(self):
        assert self.num_vehicles >= 2, "Need at least 1 human and 1 robot."
        d_start = self.ring_length / self.num_vehicles
        robot = Robot(
            env=self,
            active_controller = PID(env=self, safe_distance=self.safe_distance, gamma=2.0, m=38),
            passive_controller = BandoFTL(env=self, a=self.traffic_a, b=self.traffic_b),
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
        for vehicle in vehicles:
            self.all_vehicles.add(vehicle)
        self.state = {
            'step' : 0,
            'time' : 0.0,
            'vehicles' : vehicles,  # List of vehicles in 0,...,(N-1) index order, with A.V. at index 0.
        }

    def copy_state(self):
        state = self.state.copy()
        state['vehicles'] = state['vehicles'].copy()
        return state

    def archive_state(self):
        for vehicle in self.state['vehicles']:
            vehicle.archive_state()
        self.history[self.step] = self.copy_state()

    def get_vehicle_state_table(self, key, steps=None):
        """
        Get a DataFrame of a state value (specified by key) for each vehicle (column) at each time step (row).
        If steps is specified (as an iterable), gets specific time steps; otherwise gets all available time steps.
        """
        table = []
        # Get list of all vehicles, sorted by id:
        vehicles = sorted(self.all_vehicles, key=lambda vehicle: vehicle.id)
        for vehicle in vehicles:
            df = vehicle.get_state_table(keys=['step','time',key], steps=steps)
            df = df.rename(columns={key:vehicle.id}).set_index(['step','time'])
            table.append( df ) 
        table = pd.concat(table, axis=1)
        table.columns.name = 'vehicle_id'
        return table

    def get_vehicle_pos_table(self, steps=None):
        """Get a DataFrame of a position for each vehicle (column) at each time step (row)."""
        return self.get_vehicle_state_table(key='pos', steps=steps)

    def get_vehicle_vel_table(self, steps=None):
        """Get a DataFrame of a velocity for each vehicle (column) at each time step (row)."""
        return self.get_vehicle_state_table(key='vel', steps=steps)

    def get_vehicle_acc_table(self, steps=None):
        """Get a DataFrame of a acceleration for each vehicle (column) at each time step (row)."""
        return self.get_vehicle_state_table(key='acc', steps=steps)

    def get_vehicle_index(self, vehicle):
        """
        Returns the index (or None) of a given Vehicle.
        """
        index = None
        for i,v in enumerate(self.state['vehicles']):
            if v.id == vehicle.id:
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

    def run_step(self):
        """
        Perform simulation update for one time step.
        """

        # Calcualte control for each vehicle:
        control = dict()  # Keyed by index.
        for index,vehicle in enumerate(self.state['vehicles']):
            control[index] = vehicle.controller.calculate(vehicle)

        # Apply control for each vehicle:
        for index,vehicle in enumerate(self.state['vehicles']):
            vehicle.state['index'] = index
            vehicle.state['step'] = self.state['step']
            vehicle.state['time'] = self.state['time']
            vehicle.acc = control[index]
            vehicle.vel += vehicle.acc/self.dt
            vehicle.pos += vehicle.vel*self.dt

        # Increment time step for next iteration:
        self.state['step'] += 1
        self.state['time'] += self.dt

        # Archive environment state:
        self.archive_state()

    def run(self, steps=100):
        for s in range(steps):
            self.run_step()

    def visualize(self, step=None, draw_cars_to_scale=False, label_step=True, label_cars=True):

        # Plot latest step by default:
        if step is None:
            step = self.step

        # Get corresponding state:
        state = self.history[step]

        # Set plotting options:
        road_width = 20.
        scaled_car_width = 10.
        point_car_size = 8.
        road_color = 'silver'
        hv_color = 'firebrick'
        av_color = 'seagreen'

        # Create plot (or get current axes if animating):
        if not hasattr(self, '_animation'):
            self._animation = None
        if self._animation:
            fig = self._animation['fig']
            ax = self._animation['ax']
            ax.clear()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection='polar')

        # Find the radius of the ring given the RingRoad length
        road_radius = self.ring_length / (2 * np.pi)

        # Collect artists (for pyplot animation):
        artists = []
        
        # Plot a circle: https://stackoverflow.com/a/19828753
        polar_transform = ax.transProjectionAffine + ax.transAxes
        ring_road = plt.Circle((0, 0), road_radius, color=road_color, lw=road_width, fill=False, transform=polar_transform)
        ax.add_artist(ring_road)
        artists.append(ring_road)

        # Now plot the cars after transforming the 1-dimensional location of each to the polar coordinate system
        for car in state['vehicles']:

            # Transform the 1-D coord to polar system
            normalized_pos = car.x / self.ring_length
            car_theta = normalized_pos * (2 * np.pi)

            if car.type=='human':
                car_color = hv_color
            elif car.type=='robot':
                car_color = av_color
            else:
                raise NotImplementedError

            # Now plot the cars, whether to scale or not, with color according to whether each is an AV or human driver
            # Note: for large ring roads, it is likely better to NOT draw to scale, for easier visualization
            if draw_cars_to_scale:
                normalized_car_length = self.vehicle_length / self.ring_length
                polar_car_length = normalized_car_length * (2 * np.pi)
                car_arc_theta = np.arange(start=car_theta - polar_car_length/2,
                                    stop=car_theta + polar_car_length/2,
                                    step=0.005)
                car_arc_radius = np.repeat(road_radius, len(car_arc_theta))
                
                car_arc = ax.plot(car_arc_theta, car_arc_radius, color=car_color, lw=scaled_car_width)
                artists.append(car_arc)

            else:
                car_point, = ax.plot(car_theta, road_radius, color=car_color, marker='s', markersize=point_car_size)
                artists.append(car_point)

            # Add text:
            if label_cars:
                label = ax.text(car_theta, road_radius*1.15, "{}".format(car.id), fontsize=10, ha='center', va='center')
                artists.append(label)

        # Add text:
        if label_step:
            label = ax.text(0,0, "t = {:.1f} s".format(state['time']), fontsize=14, ha='center', va='center')
            artists.append(label)
        
        # Hide ticks and gridlines:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.spines['polar'].set_visible(False)
        ax.grid(False)

        ax.set_xlim((0,np.pi*2))
        ax.set_ylim((0,road_radius*1.05))
        
        # Return artists or figure:
        if self._animation:
            return tuple(artists)
        else:
            return fig, ax

    def animate(self, steps=None, *args, **kwargs):
        """
        Plot the history of the simulation (optionally specifying steps as an iterable).
        (Takes same options as `visualize` function).
        """
        
        # Create workspace for animation:
        self._animation = dict()

        # Get start and end steps:
        if steps is None:
            steps = range(self.step+1)

        # Define plot update function:
        def func(frame):
            return self.visualize(step=frame, *args, **kwargs)
        def init_func():
            return self.visualize(step=0, *args, **kwargs)

        # Build animation:
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        self._animation['fig'] = fig
        self._animation['ax'] = ax
        anim = matplotlib.animation.FuncAnimation(
            fig,
            func,
            frames=steps,
            #init_func=init_func,
            interval=200,
            blit=True,
            repeat=False,
        )

        # Clear workspace:
        self._animation = None

        return anim

class Vehicle:

    all_vehicles = []

    def __init__(self, env, controller=None, init_pos=0.0, init_vel=0.0, init_acc=0.0, length=4.5):

        # Generate unique ID and add to master list:
        self.id = len(Vehicle.all_vehicles)
        Vehicle.all_vehicles.append(self)

        # Use null contoller as placeholder:
        if controller is None:
            controller = Controller(env)

        # Store properties:
        self.type = None  # 'robot' or 'human'
        self.env = env
        self.controller = controller
        self.init_pos = init_pos
        self.init_vel = init_vel
        self.init_acc = init_acc
        self.length = length
        
        # Set constraints:
        self.min_vel = 0
        self.max_vel = float("+Inf")
        self.min_acc = float("-Inf")
        self.max_acc = float("+Inf")

        # Store state information:
        self.state = None
        self.history = dict()  # History of states, keyed by time step.

        # Initialize:
        self.reset_state()
    
    def __repr__(self):
        typ = 'Vehicle' if self.type is None else self.type.capitalize()
        return "<{}(id={})@x={:.2f}>".format(typ, self.id, self.x)

    @property
    def l(self):
        return self.length

    @property
    def x(self):
        return self.state['pos'].x

    @property
    def pos(self):
        return self.state['pos']

    @property
    def vel(self):
        return self.state['vel']

    @property
    def acc(self):
        return self.state['acc']

    @pos.setter
    def pos(self, pos):
        self.state['pos'] = Position(x=pos, L=self.env.L)

    @vel.setter
    def vel(self, vel):
        vel = max(vel, self.min_vel)
        vel = min(vel, self.max_vel)
        self.state['vel'] = vel

    @acc.setter
    def acc(self, acc):
        acc = max(acc, self.min_acc)
        acc = min(acc, self.max_acc)
        self.state['acc'] = acc

    def reset_state(self):
        self.state = {
            'index' : None,  # Set by environment.
            'pos' : Position(x=self.init_pos, L=self.env.L),
            'vel' : self.init_vel,
            'acc' : self.init_acc,
            'controller_type' : self.controller.type,
        }

    def copy_state(self):
        return self.state.copy()

    def archive_state(self):
        state = self.copy_state()
        state['pos'] = state['pos'].x  # Extract position from Position object.
        state['time'] = self.env.t
        state['step'] = self.env.step
        self.history[self.env.step] = state

    def get_state_table(self, keys=['step', 'time', 'pos', 'vel', 'acc'], steps=None):
        """
        Build a DataFrame of the state history
        (with specified keys as columns and all available time steps as rows).
        If steps is specified (as an iterable), gets specific time steps; otherwise gets all available time steps.
        """
        table = []
        if steps is None:
            steps = self.history.keys()
        for step in steps:
            state = self.history[step]
            table.append( {key : state[key] for key in keys if key in state.keys()} )
        table = pd.DataFrame(table, columns=keys, index=steps)
        return table

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
    
    def reset_state(self):
        super().reset_state()
        self.state['active'] = True  # Flag to determine autonomous control.
