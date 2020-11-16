# Harvard CS286 Final Project, Fall 2020.

Simulation of autonomous vehicle controllers from Delle Monarche et al. (2019):
'Feedback Control Algorithms for the Dissipation of Traffic Waves with Autonomous Vehicles'
https://doi.org/10.1007/978-3-030-25446-9_12

## Baseline Results:

- The `Baseline-Experiments.ipynb` notebook presents results from out baseline simulation.
- The `Baseline-Animation.py` script shows an animation (which runs for about a minute in an interactive pyplot window) of the vehicle positons and velocities during the baseline experiment.

## Notes:

### Assumptions:

- All vehicles have same length.
- All vehicles have same dynamics.

### Experiments:

- Try different spatial/temporal/controller resolutions.

### Possible Extensions:

- Add more lanes.
- Add entry/exit points.
- Add constraints on vehicle kinematics/dynamics.
- Add error/uncertainty in human driving.
- Add error/uncertainty in robot sensing.
- Add more robots (with and without knowledge of each other).
- Add different driving styles.
