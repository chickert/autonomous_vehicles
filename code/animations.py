"""
Wrapper class to produce animations of a RingRoad object
by iterative calling its built-in plotting functions.
Reference: https://www.c-sharpcorner.com/article/create-animated-gif-using-python-matplotlib/
"""

import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


class Animation:

    def __init__(self, env, speedup=1.0, interval=1, mode='notebook'):
        """
        Initialize an animation from the given environment.

        speedup:
            (float) Multiplier for how fast to show animation relative to actual time
            (e.g. speedup of 1.0 is realtime, a speedup of 10.0 is ten times faster).
        interval:
            (int) Interval between steps that are included as animation frames
            (e.g. an interval of 1 plots every frame, and interval of 10 plots every tenth frame).
        """

        # Check inputs:
        assert speedup > 0, "speedup should be a positive multiplier."
        assert (interval>0) and (interval == int(interval)), "interval should be a positive integer."
        interval = int(interval)
        # Calculate frame rate (used when saving GIF):
        frames = 1 / interval
        seconds = env.dt / speedup
        fps = frames / seconds

        # Check mode:
        valid_modes = ['script','notebook']
        mode = valid_modes[0] if mode is None else mode
        assert mode in valid_modes, f"{mode} is not a valid mode: {valid_modes}"

        # Store properties:
        self.env = env
        self.speedup = speedup
        self.interval = interval
        self.fps = fps
        self.mode = mode

        # Store state:
        self.anim = None  # Most recent animation.

    def start(self):
        """
        Start interactive plotting mode:
        """
        if self.mode=='script':
            plt.ion()
            plt.show(block=False)
        elif self.mode=='notebook':
            plt.ioff()
        else:
            raise NotImplementedError(f"Mode `{self.mode}` is not implemented.")

    def stop(self):
        """
        Stop interactive plotting mode:
        """
        if self.mode=='script':
            plt.ioff()
            plt.close()
        elif self.mode=='notebook':
            pass
        else:
            raise NotImplementedError(f"Mode `{self.mode}` is not implemented.")

    def animate_ring(self, ax=None, **plot_options):
        """
        Animate the result of the RingRoad.plot_ring function.
        """

        # Activate:
        self.start()
        # Create axes (or use existing ones):
        if ax:
            fig = ax.figure
        else:
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(facecolor='white', frameon=False, projection='polar')
        # Define plot initialization function:
        def init_func():
            pass
        # Define plot update function:
        def func(i):
            ax.clear()
            self.env.plot_ring(
                step=i,  # Current step.
                ax = ax, animation_mode=True,
                **plot_options  # Keyword arugments.
            )
        # Define frames:
        frames = np.arange(0,self.env.step,self.interval)
        # Animate:
        self.anim = FuncAnimation(fig=fig, func=func, frames=frames, init_func=init_func)
        # Deactivate:
        # self.stop()
        # Return:
        return self.anim

    def animate_positions(self, ax=None, **plot_options):
        """
        Animate the result of the RingRoad.plot_positions function.
        """

        # Activate:
        self.start()
        # Create axes (or use existing ones):
        if ax:
            fig = ax.figure
        else:
            fig,ax = plt.subplots(1,1, figsize=(9,7))
        # Define plot initialization function:
        def init_func():
            pass
        # Define plot update function:
        def func(i):
            ax.clear()
            self.env.plot_positions(
                steps=range(0,i+1),  # This and all previous steps.
                total_steps = self.env.step,  # Final step.
                ax = ax, animation_mode=True,
                **plot_options  # Keyword arugments.
            )
        # Define frames:
        frames = np.arange(0,self.env.step,self.interval)
        # Animate:
        self.anim = FuncAnimation(fig=fig, func=func, frames=frames, init_func=init_func)
        # Deactivate:
        # self.stop()
        # Return:
        return self.anim

    def animate_velocities(self, ax=None, **plot_options):
        """
        Animate the result of the RingRoad.plot_velocities function.
        """

        # Activate:
        self.start()
        # Create axes (or use existing ones):
        if ax:
            fig = ax.figure
        else:
            fig,ax = plt.subplots(1,1, figsize=(9,7))
        # Define plot initialization function:
        def init_func():
            pass
        # Define plot update function:
        def func(i):
            ax.clear()
            self.env.plot_velocities(
                steps=range(0,i+1),  # This and all previous steps.
                total_steps = self.env.step,  # Final step.
                ax = ax, animation_mode=True,
                **plot_options  # Keyword arugments.
            )
        # Define frames:
        frames = np.arange(0,self.env.step,self.interval)
        # Animate:
        self.anim = FuncAnimation(fig=fig, func=func, frames=frames, init_func=init_func)
        # Deactivate:
        # self.stop()
        # Return:
        return self.anim

    def animate_dashboard(self, axs=None, **plot_options):
        """
        Animate the result of the RingRoad.plot_dashboard function.
        """

        # Activate:
        self.start()
        # Create axes (or use existing ones):
        if axs:
            fig = axs[0].figure
            assert len(axs)==3, "Expect axs as a tuple of three axes."
            ax1, ax2, ax3 = axs
        else:
            fig = plt.figure(figsize=(9,7))
            ax1 = fig.add_subplot(1, 2, 1, facecolor='white', frameon=False, projection='polar')
            ax2 = fig.add_subplot(2, 2, 2, facecolor='white')
            ax3 = fig.add_subplot(2, 2, 4, facecolor='white')
            axs = (ax1,ax2,ax3)
        # Define plot initialization function:
        def init_func():
            pass
        # Define plot update function:
        def func(i):
            ax1.clear()
            ax2.clear()
            ax3.clear()
            self.env.plot_dashboard(
                step = i,  # Current step.
                total_steps = self.env.step,  # Final step.
                axs = axs, animation_mode=True,
                **plot_options  # Keyword arugments.
            )
        # Define frames:
        frames = np.arange(0,self.env.step,self.interval)
        # Animate:
        self.anim = FuncAnimation(fig=fig, func=func, frames=frames, init_func=init_func)
        # Deactivate:
        # self.stop()
        # Return:
        return self.anim

    def show(self):

        # Get animation:
        if not self.anim:
            raise RuntimeError("Animation has not been built.")

        if self.mode=='script':
            plt.show(block=True)
        elif self.mode=='notebook':
            plt.show(block=True)
        else:
            raise NotImplementedError(f"Mode `{self.mode}` is not implemented.")

    def save_gif(self, filepath, overwrite=False):

        # Get animation:
        if not self.anim:
            raise RuntimeError("Animation has not been built.")

        # Save to GIF (sometimes slow):
        if os.path.isfile(filepath) and (not overwrite):
            raise FileExistsError("overwrite=False and file already exists: {}".format(filepath))
        writer = PillowWriter(fps=self.fps)
        self.anim.save(filepath, writer=writer)
        print("Saved : {} .".format(filepath))
