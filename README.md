# Harvard CS286 Final Project, Fall 2020.

Simulation of autonomous vehicle controllers from Delle Monarche et al. (2019):
'Feedback Control Algorithms for the Dissipation of Traffic Waves with Autonomous Vehicles'
https://doi.org/10.1007/978-3-030-25446-9_12

# Data structures:

- [`code/structures.py`](code/structures.py) : Implementation of the `RingRoad` environment and `Vehicles`.
- [`code/controllers.py`](code/controllers.py) : Implementation of controllers for HV and AV driving behavior.
- [`code/animations.py`](code/animations.py) : Implementation of a wrapper class to animate a `RingRoad` simulation.
- [`code/learning.py`](code/learning.py) : Implementation of wrapper for reinforcement learning in a `RingRoad` environment.

# Running the experiments:

**Baseline and Three Simulation Extensions:**

The [`environment.yml`](environment.yml) file includes the dependences for running the baseline and three extension simulations.

The each experiment in presented in a Jupyter notebook:

- [`notebooks/Baseline-Results.ipynb`](notebooks/Baseline-Results.ipynb) presents results from out baseline simulation.
- [`notebooks/Extension-One.ipynb`](notebooks/Extension-One.ipynb) explores what happens to the baseline when we increase the number of AVs.
- [`notebooks/Extension-Two.ipynb`](notebooks/Extension-Two.ipynb) explores what happens to the baseline when we make HV behavior heterogeneous
- [`notebooks/Extension-Three.ipynb`](notebooks/Extension-Three.ipynb) explores what happens to the baseline when we add uncertainty to the AV's sensing.
- [`notebooks/Animations.ipynb`](notebooks/Animations.ipynb) presents `matplotlib` animations (which can be saved as `.gif`s) of each experiment.

The same experiments can also be run from the corresponding python files:

- [`code/baseline.py`](code/baseline.py)
- [`code/extension_one.py`](code/extension_one.py)
- [`code/extension_two.py`](code/extension_two.py)
- [`code/extension_three.py`](code/extension_three.py)

The code runs best when executed from the code directory, e.g.:
```sh
cd code
python baseline.py
```

**Reinforcement Learning Framework:**

- [`code/dqn/environment.yml`](code/dqn/environment.yml) file includes additional dependences (e.g. `pytorch`) for performing Q-Learning.
- [`code/dqn/`](code/dqn/) contains code for Deep Q-Network (DQN) learning adapted from: https://github.com/chickert/reinforcement_learning/tree/master/DQN
- [`code/learning.py`](code/learning.py) implements wrappers to connect the ring road simulation to the DQN framework.
- [`code/dqn/train.py`](code/dqn/train.py) is a script that executes a DQN training run (and uploads the hyperparameters and results to [Weights & Biases](https://www.wandb.com/)).
- [`code/notebooks/Learning.ipynb`](notebooks/Learning.ipynb) provides simple demos of the data structures created for the RL framework.

## Main assumptions:

- The vehicles drive in a circle with no passing.
- All vehicles have same length and dynamics.
- There is a single AV (except for Extension 1 and Q-Learning)
- All human vehicles follow the same traffic model (except for Extension 2).
- The AV has perfect sensing of itself and lead vehicle (except for Extension 3 and Q-learning)
- The HVs have perfect sensing of their own velocity and distance to the vehicle in from.
