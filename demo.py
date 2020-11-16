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
    av_activate = 40,
    seed = 286,
)
env.run(steps=int(80/env.dt))

# Show animation:
fig = plt.figure()
ax = fig.add_subplot(projection='polar', facecolor='white', frameon=False)
env.start_animation(fig, ax)
speedup = 100
for step in range(0,env.step,1):
    env.visualize(step=step, draw_cars_to_scale=True, draw_safety_buffer=False)
    plt.pause(env.dt/speedup)
env.stop_animation()

# Plot final state:
fig, ax = env.visualize(step=None, draw_cars_to_scale=True, draw_safety_buffer=True)
plt.show()
