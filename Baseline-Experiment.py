import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Adjust relative path so that this script can find the code modules:
import sys
sys.path.append('code/')

from structures import *

# Hide warnings about safe distance violation (still in development):
warnings.filterwarnings("ignore", category=UserWarning)

# Define a ring road environment:
env = RingRoad(
    num_vehicles = 22,
    ring_length = 230.0,
    starting_noise = 4.0,
    temporal_res = 0.3,
    av_activate = 30,
    seed = 286,
)

# Run the simulation for set number of time steps:
total_time = 50  # In seconds.
total_steps = int(np.ceil(total_time/env.dt))
env.run(steps=total_steps)

# Plot initial state:
fig,ax = env.visualize(step=0, draw_cars_to_scale=True, draw_safety_buffer=False)
filename = "outputs/environment_initial_state.png"
plt.savefig(filename)
print("Saved : {} .".format(filename))

# Plot state at activation:
fig,ax = env.visualize(step=int(np.ceil(env.av_activate/env.dt))-1, draw_cars_to_scale=True, draw_safety_buffer=False)
filename = "outputs/environment_before_activation.png"
plt.savefig(filename)
print("Saved : {} .".format(filename))

# Plot stable state:
fig,ax = env.visualize(step=env.step, draw_cars_to_scale=True, draw_safety_buffer=False)
filename = "outputs/environment_stable_state.png"
plt.savefig(filename)
print("Saved : {} .".format(filename))

# Plot position history:
fig,ax = env.plot_positions()
filename = "outputs/positions_history.png"
plt.savefig(filename)
print("Saved : {} .".format(filename))

# Plot velocity history:
fig,ax = env.plot_velocities(show_sigma=True)
filename = "outputs/velocities_history.png"
plt.savefig(filename)
print("Saved : {} .".format(filename))

# Calculate standard deviations:
speeds_before = env.get_vehicle_vel_table( range(0,env.av_activate) )
speeds_after = env.get_vehicle_vel_table( range(env.av_activate, env.step) )
df = pd.concat([
    pd.DataFrame(speeds_before.std(axis=0).to_frame(name='before AV control')),
    pd.DataFrame(speeds_after.std(axis=0).to_frame(name='after AV control')),
], axis=1)
df_mean = df.mean(axis=0).to_frame(name='mean').transpose()
df = pd.concat([df_mean,df], axis=0)
filename = "outputs/velocity_standard_deviations.csv"
df.to_csv(filename, index=False)
print("Saved : {} .".format(filename))

print("Mean standard deviation before: {}".format(df.loc['mean','before AV control']))
print("Mean standard deviation after: {}".format(df.loc['mean','after AV control']))

print("Done.")
