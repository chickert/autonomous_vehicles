import matplotlib.pyplot as plt
import numpy as np
from structures import RingRoad


def extension_one(num_vehicles, show_fig=True):
    """
    Runs an experiment investigating how the AV:HV ratio affects performance
    """
    mean_system_vels = []  # mean velocity across all vehicles and all timesteps after AV activation
    variance_across_vehicles = []  # (mean is taken across 22 std devs -- one std dev from each vehicle)
    variance_across_timesteps = []  # (mean is taken across all timesteps -- one std dev from each timestep)

    pct_avs = []

    for num_avs in range(1, num_vehicles + 1):
        env = RingRoad(
            num_vehicles=num_vehicles,  # The vehicles at index 0 is an A.V.
            ring_length=230.0,  # The road is a circle.
            starting_noise=4.0,  # Uniformly add noise to starting positions.
            temporal_res=0.3,  # Set the size of simulation steps (seconds).
            av_activate=30,  # Set when the PID controller is activated.
            seed=286,  # Set a random seed.
            num_avs=num_avs
        )

        # Plot initial conditions for select cases for illustrative purposes
        if num_avs == 1 or (num_avs % int(num_vehicles // 2)) == 0:
            env.visualize(step=0, draw_cars_to_scale=True, draw_safety_buffer=True)
            plt.savefig("../outputs/ext1-num_av-"+str(num_avs))
            if show_fig:
                plt.show()
            plt.close()

        # Run each simulation for set number of time steps:
        total_time = 50  # In seconds.
        total_steps = int(np.ceil(total_time / env.dt))
        env.run(steps=total_steps)

        # Collect and system metrics for plotting:
        steps_after = range(env.av_activate, env.step)
        speeds_after = env.get_vehicle_vel_table(steps_after)

        # Store std dev and mean of each vehicle's velocity averaged across all vehicles after controller was activated
        mean_system_vels.append(
            speeds_after.mean(axis=0).mean())  # mean velocity across all vehicles and all timesteps after AV activation
        variance_across_vehicles.append(
            speeds_after.std(axis=0).mean())  # (mean is taken across 22 std devs -- one std dev from each vehicle)
        variance_across_timesteps.append(
            speeds_after.std(axis=1).mean())  # (mean is taken across all timesteps -- one std dev from each timestep)
        pct_avs.append((num_avs / num_vehicles) * 100)

    # Plot results
    plt.plot(pct_avs, mean_system_vels)
    plt.xlabel("% of Vehicles that are AVs")
    plt.ylabel("Mean Velocity (after AV activation)\n across All Vehicles")
    plt.title("System's Mean Velocity\n as Proportion of AVs Increases")
    plt.savefig("../outputs/ext1-meanvel")
    if show_fig:
        plt.show()
    plt.close()

    plt.plot(pct_avs, variance_across_vehicles)
    plt.xlabel("% of Vehicles that are AVs")
    plt.ylabel("Velocity Std. Dev. (after AV activation)\n across All Vehicles")
    plt.title("Mean across Individual Vehicle Std Devs\n as Proportion of AVs Increases")
    plt.savefig("../outputs/ext1-stddev1")
    if show_fig:
        plt.show()
    plt.close()

    plt.plot(pct_avs, variance_across_timesteps)
    plt.xlabel("% of Vehicles that are AVs")
    plt.ylabel("Velocity Std. Dev. (after AV activation)\n across All Timesteps")
    plt.title("Mean across Individual Timestep Std Devs\n as Proportion of AVs Increases")
    plt.savefig("../outputs/ext1-stddev2")
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

    num_vehicles = 22
    # Scale the proportion of AVs in the system (from 0% to 100%) and record the results for each
    # Then, plot the results
    extension_one(num_vehicles, show_fig=False)
