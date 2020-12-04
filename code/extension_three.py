import matplotlib.pyplot as plt
import numpy as np
from structures import RingRoad


def extension_three(max_sigma_pct, granularity, show_fig=True):
    """
    Runs an experiment investigating the role of sensor noise on AV and system performance
    """
    max_sigma_pct = max_sigma_pct / 100
    granularity = granularity / 100
    sigma_pcts = np.arange(0., max_sigma_pct, granularity)
    mean_system_vels = []               # mean velocity across all vehicles and all timesteps after AV activation
    variance_across_vehicles = []       # (mean is taken across 22 std devs -- one std dev from each vehicle)
    variance_across_timesteps = []      # (mean is taken across all timesteps -- one std dev from each timestep)

    print(f"Running {len(sigma_pcts)} simulations")
    counter = 0
    for sigma_pct in sigma_pcts:

        # Increment counter
        counter += 1
        if counter % 20 == 0:
            print(f'On simulation #{counter}')

        # Build env
        env = RingRoad(
            num_vehicles=22,  # The vechicles at index 0 is an A.V.
            ring_length=230.0,  # The road is a cicle.
            starting_noise=4.0,  # Uniformly add noise to starting positions.
            temporal_res=0.3,  # Set the size of simulation steps (seconds).
            av_activate=30,  # Set when the PID controller is activated.
            seed=286,  # Set a random seed.
            uncertain_avs=True,
            sigma_pct=sigma_pct
        )

        # Run each simulation for set number of time steps:
        total_time = 50  # In seconds.
        total_steps = int(np.ceil(total_time / env.dt))
        env.run(steps=total_steps)

        # Collect and system metrics for plotting:
        steps_after = range(env.av_activate, env.step)
        speeds_after = env.get_vehicle_vel_table(steps_after)

        # Store std dev and mean of each vehicle's velocity averaged across all vehicles after controller was activated
        mean_system_vels.append(speeds_after.mean(axis=0).mean())           # mean velocity across all vehicles and all timesteps after AV activation
        variance_across_vehicles.append(speeds_after.std(axis=0).mean())    # (mean is taken across 22 std devs -- one std dev from each vehicle)
        variance_across_timesteps.append(speeds_after.std(axis=1).mean())   # (mean is taken across all timesteps -- one std dev from each timestep)

    # Plot results
    sigma_pcts = sigma_pcts * 100
    plt.plot(sigma_pcts, mean_system_vels)
    plt.xlabel("Sensor Noise\n(as a % of $\Delta_x$ and lead vehicle velocity measurements)")
    plt.ylabel("Mean Velocity (after AV activation)\n across All Vehicles")
    plt.title("System's Mean Velocity\n as Uncertainty Increases")
    plt.savefig("../outputs/ext3-meanvel")
    if show_fig:
        plt.show()
    plt.close()

    plt.plot(sigma_pcts, variance_across_vehicles)
    plt.xlabel("Sensor Noise\n(as a % of $\Delta_x$ and lead vehicle velocity measurements)")
    plt.ylabel("Velocity Std. Dev. (after AV activation)\n across All Vehicles")
    plt.title("Mean across Individual Vehicle Velocity Std Devs\n as Uncertainty Increases")
    plt.savefig("../outputs/ext3-stddev1")
    if show_fig:
        plt.show()
    plt.close()

    plt.plot(sigma_pcts, variance_across_timesteps)
    plt.xlabel("Sensor Noise\n(as a % of $\Delta_x$ and lead vehicle velocity measurements)")
    plt.ylabel("Velocity Std. Dev. (after AV activation)\n across All Timesteps")
    plt.title("Mean across Individual Timestep System Velocity Std Devs\n as Uncertainty Increases")
    plt.savefig("../outputs/ext3-stddev2")
    if show_fig:
        plt.show()
    plt.close()

    if not show_fig:
        print("\nExperiment complete. See the ../outputs/ directory for resulting visualizations and plots.\n"
              "To display the visualizations as the code runs, ensure the 'show_fig' argument is set to True")


if __name__ == '__main__':
    import warnings

    plt.style.use('seaborn-darkgrid')
    warnings.filterwarnings("ignore", category=UserWarning)

    extension_three(max_sigma_pct=4000, granularity=100, show_fig=False)
