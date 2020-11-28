"""
Harvard CS286 Final Project, Fall 2020.
Data structures for simulation of autonomous vehicle controllers presented by Delle Monarche et al. (2019):
'Feedback Control Algorithms for the Dissipation of Traffic Waves with Autonomous Vehicles'
https://doi.org/10.1007/978-3-030-25446-9_12
"""

import os
import warnings

import numpy as np
import pandas as pd

import matplotlib.animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  

from controllers import Controller, BandoFTL, PID


class Position:

    TOL = 1e-9  # Tolerance for floating point comparisons.

    def __init__(self, x, L):
        """
        Representation of a position x on ring road of length L, with modular arithmetic.
        Positions are always positive, as stored as a float in the interval [ 0.0, L ).
        """
        if hasattr(x, 'env'): # Check if the input is alread a Vehicle.
            assert L == x.env.ring_length, "Cannot mix positions with different L values."
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

    def __init__(self, num_vehicles=22, ring_length=230.0, starting_noise=0.5, av_activate=60.0, temporal_res=0.1, seed=None):
        
        # Store properties:
        self.num_vehicles = num_vehicles  # Total number of vehicles (including A.V.).
        self.ring_length = ring_length  # Length of ring road (meters).
        self.vehicle_length = 4.5  # Length of vehicles (meters).
        self.safe_distance = 4.0  # Safe distance between vehicles (meters).
        self.min_speed = 0.00  # Min velocity (meters/second).
        self.max_speed = 9.75  # Max velocity (meters/second).
        self.min_accel = -7  # Min acceleration (meters/second^2).
        self.max_accel = 6.50  # Max acceleration (meters/second^2).
        self.temporal_res = temporal_res  # Time between updates (in seconds).
        self.spatial_res = None
        self.traffic_a = 0.5  # Coefficient for the FTL model (meters/second).
        self.traffic_b = 20  # Coefficient for the Bando-OV model (1/second).
        self.av_activate = av_activate  # When to activate AV controller (seconds).
        self.starting_noise = starting_noise  # Add noise (in meters) to starting positions.
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
        try:
            return self.state['step']
        except:  # Before state initialization:
            return 0

    @property
    def t(self):
        try:
            return self.state['time']
        except:  # Before state initialization:
            return 0.0

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
            s += "  [{}] ".format(index) + vehicle.__str__() + "\n"
        return s

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
        robot.state['index'] = 0
        robot.active = (self.av_activate==0)
        vehicles = [robot]
        for index in range(1,self.num_vehicles):
            noise = self.starting_noise
            noise = self.random.uniform(-noise/2,noise/2)  # 1 centimeter.
            human = Human(
                env=self,
                controller = BandoFTL(env=self, a=self.traffic_a, b=self.traffic_b),
                init_pos = index * d_start + noise,
                init_vel = 0.0,
                init_acc = 0.0,
                length = self.vehicle_length,
            )
            human.state['index'] = index
            vehicles.append(human)
        for vehicle in vehicles:
            # Adjust kinematics:
            vehicle.min_vel = self.min_speed
            vehicle.max_vel = self.max_speed
            vehicle.min_acc = self.min_accel
            vehicle.max_acc = self.max_accel# + np.round(self.random.uniform(-0.25,0.25),2)
            vehicle.min_acc =  - vehicle.max_acc
            # Add vehicle:
            self.all_vehicles.add(vehicle)
        self.state = {
            'step' : 0,
            'time' : 0.0,
            'vehicles' : vehicles,  # List of vehicles in 0,...,(N-1) index order, with A.V. at index 0.
            'av_active' : robot.active,
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

    def get_vehicle_control_table(self, steps=None):
        """Get a DataFrame of a (unconstrained) control for each vehicle (column) at each time step (row)."""
        return self.get_vehicle_state_table(key='control', steps=steps)

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

    def check(self):
        """
        Makes sure no vehicles have passed another or gotten too close.
        """

        # Sort vehicles in order of actual position (regardless of index):
        vehicles = sorted(self.vehicles, key=lambda vehicle: vehicle.pos.x)

        # Loop through vehicles to check that they are also in index order:
        for j in range(len(vehicles)):
            this_vehicle = vehicles[j]
            lead_vehicle = vehicles[(j+1)%self.N]
            this_index = self.get_vehicle_index(this_vehicle)
            lead_index = self.get_vehicle_index(lead_vehicle)
            if (this_index+1) % self.N != lead_index:
                raise RuntimeError("Illegal passing occured at step={} around index {} : {}".format(
                    self.step,
                    this_index,
                    this_vehicle, #this_vehicle.__repr__(),
                ))
            # Check safety distance:
            if this_vehicle.distance_to(lead_vehicle) - lead_vehicle.length < self.safe_distance:
                warning = "WARNING: Safe distance violation at step {}:".format(self.step)
                warning += "  [{}] {}".format(this_index,this_vehicle)
                warning += "  [{}] {}".format(lead_index,lead_vehicle)
                warnings.warn(warning)
            # Store values for next iteration:
            lead_vehicle = this_vehicle
            lead_index = this_index

    def run_step(self):
        """
        Perform simulation update for one time step.
        """

        # Calcualte control for each vehicle:
        controls = dict()  # Keyed by index.
        for index,vehicle in enumerate(self.state['vehicles']):
            if (vehicle.type == 'robot') and (not vehicle.active) and (self.t >= self.av_activate):
                vehicle.active = True
            controls[index] = vehicle.controller.calculate(vehicle)

        # Apply control for each vehicle:
        for index,vehicle in enumerate(self.state['vehicles']):
            vehicle.state['index'] = index
            vehicle.state['step'] = self.state['step']
            vehicle.state['time'] = self.state['time']
            vehicle.state['control'] = controls[index]  # Unconstrainted command.
            vehicle.acc = controls[index]  # Enforces kinematic constraints.
            vehicle.vel += vehicle.acc*self.dt
            vehicle.pos += vehicle.vel*self.dt

        # Make sure there has been no illegal passing or tailgaiting.
        self.check()

        # Increment time step for next iteration:
        self.state['step'] += 1
        self.state['time'] += self.dt

        # Archive environment state:
        self.archive_state()

    def run(self, steps=100):
        for s in range(steps):
            self.run_step()

    def start_animation(self, fig, axs):
        # if hasattr(self, '_animation') and (self._animation is not None):
        #     raise RuntimeError("Animation already in progress.")
        self._animation = {
            'fig' : fig,
        }
        try:
            axs[0]  # For single ax.
        except:
            axs = tuple([axs])
        self._animation['axs'] = axs
        plt.ion()
        plt.show(block=False)

    def stop_animation(self):
        self._animation = None
        plt.ioff()
        plt.close()

    def visualize(self, step=None, draw_cars_to_scale=False, draw_safety_buffer=False, label_step=True, label_cars=True, ax=None):

        # Plot latest step by default:
        if step is None:
            step = self.step

        # Get corresponding state:
        state = self.history[step]

        # Set plotting options:
        road_width = 6.
        car_width = 3.
        point_car_size = 6.
        road_color = 'silver'
        hv_color = 'firebrick'
        av_color = 'seagreen'

        # Create plot (or get current axes if animating):
        if not hasattr(self, '_animation'):
            self._animation = None
        if ax is not None:
            fig = ax.figure
        elif self._animation:
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
        #ring_road = plt.Circle((0, 0), road_radius, color=road_color, zorder=1, lw=road_width, fill=False, transform=polar_transform)
        ring_road = plt.Rectangle(xy=(0, road_radius-road_width/2), width=2*np.pi, height=road_width, lw=0, color=road_color, zorder=1, fill=True)
        ax.add_artist(ring_road)
        artists.append(ring_road)
        ax.bar(0, 1).remove()  # Hack: https://github.com/matplotlib/matplotlib/issues/8521#issue-223234274

        # Now plot the cars after transforming the 1-dimensional location of each to the polar coordinate system
        for car in state['vehicles']:

            # Get relevant state variables (each row is a time step and each column is a variable):
            car_state = car.get_state_table(keys=['index','pos'], steps=[step])
            car_state = car_state.iloc[0].to_dict()  # Convert the single table row to a dictionary.
            car_state['index'] = int(car_state['index'])  # Make sure index is an integer.
            if car.type=='human':
                car_color = hv_color
                car_zorder = 2
            elif car.type=='robot':
                car_color = av_color
                car_zorder = 3
            else:
                raise NotImplementedError

            # Transform the 1-D coord to polar system
            car_theta = (2*np.pi) * car_state['pos'] / self.ring_length

            # Now plot the cars, whether to scale or not, with color according to whether each is an AV or human driver
            # Note: for large ring roads, it is likely better to NOT draw to scale, for easier visualization
            if draw_cars_to_scale:
                
                # Draw car:
                car_polar_length = (2*np.pi) * car.length / self.ring_length
                car_rectangle = plt.Rectangle(xy=(car_theta-car_polar_length, road_radius-car_width/2), width=car_polar_length, height=car_width, lw=0, color=car_color, zorder=car_zorder, fill=True)
                ax.add_artist(car_rectangle)
                artists.append(car_rectangle)
                
                # Draw safety zone behind car:
                if draw_safety_buffer:
                    car_polar_buffer = (2*np.pi) * self.safe_distance / self.ring_length
                    car_buffer = plt.Rectangle(xy=(car_theta-car_polar_length-car_polar_buffer, road_radius-car_width/2), width=car_polar_buffer, height=car_width, lw=0, color='gold', zorder=car_zorder-0.1, alpha=0.4, fill=True)
                    ax.add_artist(car_buffer)
                    artists.append(car_buffer)

            else:

                # Draw car:
                car_point, = ax.plot(car_theta, road_radius, color=car_color, zorder=car_zorder, marker='o', markersize=point_car_size)
                artists.append(car_point)

            # Add text:
            if label_cars:
                car_label = "{}".format(car_state['index'])
                label = ax.text(car_theta, (road_radius+road_width*1.25), car_label, fontsize=10, ha='center', va='center')
                artists.append(label)

        # Add text:
        if label_step:
            step_label = "t = {:.1f} s".format(state['time'])
            if label_cars:
                step_label = " \n\n"+step_label+"\n\n"+"A.V. = 0"
            label = ax.text(0,0, step_label, fontsize=14, ha='center', va='center')
            artists.append(label)
        
        # Hide ticks and gridlines:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.spines['polar'].set_visible(False)
        ax.grid(False)

        ax.set_xlim((0,np.pi*2))
        ax.set_ylim((0,(road_radius+road_width/2)*1.05))
        
        # Return artists or figure:
        if self._animation:
            return tuple(artists)
        else:
            return fig, ax

    def plot_positions(self, steps=None, ax=None):
        """
        Plot positions of vehicles (y axis) over time (x axis).
        Optionally, specify step with an iterable (for animation).
        """
        
        # Set plotting options:
        hv_color = 'firebrick'
        av_color = 'seagreen'
        
        # Create plot (or get current axes if animating):
        if not hasattr(self, '_animation'):
            self._animation = None        
        if ax is not None:
            fig = ax.figure
        elif self._animation:
            fig = self._animation['fig']
            ax = self._animation['ax']
            ax.clear()
        else:
            fig,ax = plt.subplots(1,1, figsize=(16,4))

        # Collect artists (for pyplot animation):
        artists = []

        # Get steps to plot:
        if steps is None:
            steps = range(0,self.step)
        
        # Plot each vehicle:
        for vehicle in self.all_vehicles:
            # Get a table of state history for this vehicle:
            table = vehicle.get_state_table(keys=['step','time','pos'], steps=steps)
            # Set plotting options:
            if vehicle.type=='human':
                color = hv_color
                alpha = 0.5
                zorder = 2
            elif vehicle.type=='robot':
                color = av_color
                alpha = 0.75
                zorder = 3
            else:
                raise NotImplementedError
            # Plot a separate chunk for each revolution:
            prev_break = 0
            prev_row = None
            for i in range(len(table)):
                this_row = table.iloc[i]
                # Determine whether to plot new chunk:
                if prev_row is None:
                    new_chunk = False  # First row.
                elif i == len(table)-1:
                    new_chunk = True  # Last row.
                elif this_row['pos'] < prev_row['pos']:
                    new_chunk = True  # Row with wrap around.
                else:
                    new_chunk = False  # All other rows.
                # Plot new chunk if needed:
                if new_chunk:
                    df = table.iloc[prev_break:i]
                    lines, = ax.plot(df['time'],df['pos'], color=color, alpha=alpha, zorder=zorder)
                    artists.append(lines)
                    prev_break = i
                prev_row = this_row

        # Add line for AV activation:
        y_min,y_max = 0, self.L
        if self.av_activate < self.t:
            ax.plot([self.av_activate,self.av_activate],[y_min,y_max], ls=':', color='black', alpha=1, zorder=5)
        ax.set_ylim((y_min,y_max))
                
        # Set axes:
        #ax.set_title("Position over time")
        ax.set_xlabel("time (seconds)")
        ax.set_ylabel("position (meters)")
        
        # Return artists or figure:
        if self._animation:
            return tuple(artists)
        else:
            return fig, ax

    def plot_velocities(self, steps=None, show_sigma=False, ax=None):
        """
        Plot velocities of vehicles (y axis) over time (x axis).
        Optionally, specify step with an iterable (for animation).
        """
        
        # Set plotting options:
        hv_color = 'firebrick'
        av_color = 'seagreen'
        
        # Create plot (or get current axes if animating):
        if not hasattr(self, '_animation'):
            self._animation = None
        if ax is not None:
            fig = ax.figure
        elif self._animation:
            fig = self._animation['fig']
            ax = self._animation['ax']
            ax.clear()
        else:
            fig,ax = plt.subplots(1,1, figsize=(16,4))

        # Collect artists (for pyplot animation):
        artists = []

        # Get steps to plot:
        if steps is None:
            steps = range(0,self.step)
        
        # Plot each vehicle:
        for vehicle in self.all_vehicles:
            # Get a table of state history for this vehicle:
            table = vehicle.get_state_table(keys=['step','time','vel'], steps=steps)
            # Set plotting options:
            if vehicle.type=='human':
                color = hv_color
                alpha = 0.5
                zorder = 2
            elif vehicle.type=='robot':
                color = av_color
                alpha = 0.75
                zorder = 3
            else:
                raise NotImplementedError
            # Plot:
            lines, = ax.plot(table['time'],table['vel'], color=color, alpha=alpha, zorder=zorder)
            artists.append(lines)

        # Plot standard deviation across vehicles:
        if show_sigma:
            table = self.get_vehicle_vel_table(steps=steps).std(axis=1).to_frame(name='sigma').reset_index()
            ax.plot(table['time'], table['sigma'], lw=1, color='grey', label="Standard deviation\nacross all vehicles")
            ax.legend(loc='center right', fontsize=6)

        # Add line for AV activation:
        #y_min,y_max = ax.get_ylim()
        y_min,y_max = 0, min(30,self.max_speed)*1.05
        if self.av_activate < self.t:
            ax.plot([self.av_activate,self.av_activate],[y_min,y_max], ls=':', color='black', alpha=1, zorder=5)
        ax.set_ylim((y_min,y_max))
                
        # Set axes:
        #ax.set_title("Velocity over time")
        ax.set_xlabel("time (seconds)")
        ax.set_ylabel("velocity (meters/second)")
        
        # Return artists or figure:
        if self._animation:
            return tuple(artists)
        else:
            return fig, ax

    def plot_dashboard(self, step=None, total_steps=None, **plot_options):
        """
        Plot a combination of plots for a specific step.
        """

        # Create plot (or get current axes if animating):
        if not hasattr(self, '_animation'):
            self._animation = None
        if self._animation:
            fig = self._animation['fig']
            axs = self._animation['axs']
            ax1,ax2,ax3 = axs
            for ax in axs:
                ax.clear()
        else:
            fig = plt.figure(figsize=(16,6))
            ax1 = fig.add_subplot(1, 2, 1, facecolor='white', frameon=False, projection='polar')
            ax2 = fig.add_subplot(2, 2, 2, facecolor='white')
            ax3 = fig.add_subplot(2, 2, 4, facecolor='white')
        
        if step is None:
            step = self.step

        # Parse options:
        ax1_options = dict()
        ax2_options = dict()
        ax3_options = dict()
        for k,v in plot_options.items():
            if k in {'draw_cars_to_scale','draw_safety_buffer','label_cars','label_step'}:
                ax1_options[k] = v
            if k in {}:
                ax2_options[k] = v
            if k in {'show_sigma'}:
                ax3_options[k] = v

        artists = []  # Collect arists for animation.
        artists.extend( self.visualize(ax=ax1, step=step, **ax1_options) )
        artists.extend( self.plot_positions(ax=ax2, steps=range(0,step), **ax2_options) )
        artists.extend( self.plot_velocities(ax=ax3, steps=range(0,step), **ax3_options) )
        
        if total_steps is not None:
            ax2.set_xlim(0,total_steps*self.dt)
            ax3.set_xlim(0,total_steps*self.dt)

        # Return artists or figure:
        if self._animation:
            return tuple(artists)
        else:
            return fig, (ax1,ax2,ax3)

    def build_animation(self, speedup=1.0, interval=1, kind='dashboard', axs=None, **plot_options):
        """
        speedup:
            (float) Multiplier for how fast to show animation relative to actual time
            (e.g. speedup of 1.0 is realtime, a speedup of 10.0 is ten times faster).
        interval:
            (int) Interval between steps that are included as animation frames
            (e.g. an interval of 1 plots every frame, and interval of 10 plots every tenth frame).
        kind:

        """

        valid_kinds = {'positions','velocities','dashboard','ring'}
        assert kind in valid_kinds, "{} is not a valid kind: {} .".format(kind,valid_kinds)

        # Check inputs:
        assert speedup > 0, "speedup should be a positive multiplier."
        assert (interval>0) and (interval == int(interval)), "interval should be a positive integer."
        interval = int(interval)

        # Calculate frame rate (used when saving GIF):
        fps = 1/self.dt/interval*speedup

        if axs:
            fig = axs[0].figure
        else:
            fig = plt.figure(figsize=(9,7))
            ax1 = fig.add_subplot(1, 2, 1, facecolor='white', frameon=False, projection='polar')
            ax2 = fig.add_subplot(2, 2, 2, facecolor='white')
            ax3 = fig.add_subplot(2, 2, 4, facecolor='white')
            axs = (ax1,ax2,ax3)


        self.start_animation(fig=fig, axs=axs)
        self._animation['fps'] = fps
        self._animation['speedup'] = speedup
        self._animation['interval'] = interval

        def init_func():
            pass

        def func(i):
            if kind=='dashboard':
                self.plot_dashboard(
                    step = i,
                    total_steps = self.step,  # Final step.
                    **plot_options  # Keyword arugments.
                )
            elif kind=='positions':
                self.plot_positions(steps=range(0,i+1), **plot_options)
            elif kind=='velocities':
                self.plot_velocities(steps=range(0,i+1), **plot_options)
            elif kind=='ring':
                self.visualize(step=i, **plot_options)
            else:
                raise NotImplementedError("Kind {} is not yet implemented.".format(kind))

        frames = np.arange(0,self.step,interval)

        anim = FuncAnimation(fig=fig, func=func, frames=frames, init_func=init_func)
        self._animation['anim'] = anim

        # self.stop_animation()

        return anim

    def save_gif(self, filepath, overwrite=False):

        # Get animation:
        if (not hasattr(self, '_animation')) or (self._animation is None):
            raise RuntimeError("No animation in progress.")
        elif not ('anim' in self._animation):
            raise RuntimeError("Animation has not been built.")
        anim = self._animation['anim']
        fps = self._animation['fps']

        # Save to GIF (sometimes slow):
        if os.path.isfile(filepath) and (not overwrite):
            raise FileExistsError("overwrite=False and file already exists: {}".format(filepath))
        writer = PillowWriter(fps=fps)
        anim.save(filepath, writer=writer)
        print("Saved : {} .".format(filepath))


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
            'time' : self.env.t,
            'step' : self.env.step,
            'index' : None,  # Set by environment.
            'pos' : Position(x=self.init_pos, L=self.env.L),
            'vel' : self.init_vel,
            'acc' : self.init_acc,
            'control' : self.init_acc,
            'controller_type' : self.controller.type,
        }

    def copy_state(self):
        return self.state.copy()

    def archive_state(self):
        state = self.copy_state()
        state['time'] = self.env.t
        state['step'] = self.env.step
        state['pos'] = state['pos'].x  # Extract position from Position object.
        self.history[self.env.step] = state

    def get_state_table(self, keys=['step', 'time', 'index', 'pos', 'vel', 'acc', 'control'], steps=None):
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

    def distance_to(self, other):
        """
        Return the distance from self to other (i.e. by how much does other lead self?)
        If reverse=True, returns the distance from other to self, travelling in direction of the road.
        Distances are always positive and are measured in the direction of traffic.
        """
        other = Position(other, self.env.L)  # Convert to position.
        return self.pos.distance_to(other)  # Call Position.distance_to(Position) .

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
