import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from structures import *

warnings.filterwarnings("ignore", category=UserWarning)

env = RingRoad(
    num_vehicles = 22,
    ring_length = 230.0,
    starting_noise = 4.0,
    temporal_res = 0.3,
    av_activate = 30,
    seed = 286,
)
total_steps = int(np.ceil(50/env.dt))
env.run(steps=total_steps)

# Show animation:
speedup = 100.0  # Speed up animation relative to real time (not accounting for time it takes to draw plots).
interval = 1  # Only plot every n-th frame.
fig = plt.figure(figsize=(16,6))
ax1 = fig.add_subplot(1, 2, 1, facecolor='white', frameon=False, projection='polar')
ax2 = fig.add_subplot(2, 2, 2, facecolor='white')
ax3 = fig.add_subplot(2, 2, 4, facecolor='white')
axs = (ax1,ax2,ax3)
env.start_animation(fig=fig, axs=axs)
for step in np.arange(0,env.step,interval):
    #env.visualize(step=step, draw_cars_to_scale=True, draw_safety_buffer=False)
    env.plot_dashboad(step=step, total_steps=total_steps, draw_cars_to_scale=True, draw_safety_buffer=False, show_sigma=True)
    # Pause between frames (doesn't account for how long the plotting takes, so it is slower than realtime at high frame rates.).
    plt.pause(env.dt*interval/speedup)  # In seconds
env.stop_animation()

# Plot final state:
fig, axs = env.plot_dashboad(step=None, draw_cars_to_scale=True, draw_safety_buffer=True, show_sigma=True)
plt.show()
