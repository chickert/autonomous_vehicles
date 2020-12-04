from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from structures import RingRoad


def extension_two(a_sigma_start, a_sigma_stop, a_sigma_granularity, b_sigma_start, b_sigma_stop, b_sigma_granularity, show_fig=True):
    """
    Runs an experiment investigating how the AVs and system respond to system heterogeneity
    """
    # Initialize lists to store results
    mean_system_vels = []           # mean velocity across all vehicles and all timesteps after AV activation
    variance_across_vehicles = []   # (mean is taken across 22 std devs -- one std dev from each vehicle)
    variance_across_timesteps = []  # (mean is taken across all timesteps -- one std dev from each timestep)
    # Define grid of values to scan over
    a_sigma_grid = np.arange(a_sigma_start, a_sigma_stop, a_sigma_granularity)
    b_sigma_grid = np.arange(b_sigma_start, b_sigma_stop, b_sigma_granularity)
    counter = 0
    print(f"Computing values for {len(a_sigma_grid) * len(b_sigma_grid)} combinations")
    for a_sigma in a_sigma_grid:
        for b_sigma in b_sigma_grid:
            counter += 1
            mean_buffer = []
            veh_var_buffer = []
            sys_var_buffer = []
            if counter % 10 == 0:
                print(f"On combintion #{counter}")
            for seed in range(3):
                # Define the env with the selected a_sigma and b_sigma values, and with hv_heterogeneity set to True
                env = RingRoad(
                    num_vehicles=22,  # The vechicles at index 0 is an A.V.
                    ring_length=230.0,  # The road is a cicle.
                    starting_noise=4.0,  # Uniformly add noise to starting positions.
                    temporal_res=0.3,  # Set the size of simulation steps (seconds).
                    av_activate=30,  # Set when the PID controller is activated.
                    seed=seed,  # Set a random seed.
                    a_sigma=a_sigma,
                    b_sigma=b_sigma,
                    hv_heterogeneity=True,
                )

                # Run the simulation for set number of time steps:
                total_time = 50  # In seconds.
                total_steps = int(np.ceil(total_time / env.dt))
                env.run(steps=total_steps)
                # Collect and system metrics for plotting:
                steps_after = range(env.av_activate, env.step)
                speeds_after = env.get_vehicle_vel_table(steps_after)

                mean_buffer.append(speeds_after.mean(axis=0).mean())
                veh_var_buffer.append(speeds_after.std(axis=0).mean())
                sys_var_buffer.append(speeds_after.std(axis=1).mean())

            # Store std dev and mean of each vehicle's velocity averaged
            # across all vehicles after controller was activated
            mean_vel = np.mean(mean_buffer)
            veh_var = np.mean(veh_var_buffer)
            sys_var = np.mean(sys_var_buffer)
            mean_system_vels.append(mean_vel)   # mean velocity across all vehicles and all timesteps after AV activation
            variance_across_vehicles.append(veh_var)  # (mean is taken across 22 std devs -- one std dev from each vehicle)
            variance_across_timesteps.append(sys_var)  # (mean is taken across all timesteps -- one std dev from each timestep)

    # Now plot results
    plot_3d(a_sigma_grid, b_sigma_grid, mean_system_vels,
            title="Mean System Velocity", filetitle="ext2-meanvel", show_fig=show_fig)
    plot_3d(a_sigma_grid, b_sigma_grid, variance_across_vehicles,
            title="Mean across Individual Vehicle Std Devs", filetitle="ext2-stddev1", show_fig=show_fig)
    plot_3d(a_sigma_grid, b_sigma_grid, variance_across_timesteps,
            title="Mean across Individual Timestep Std Devs", filetitle="ext2-stddev2", show_fig=show_fig)

    if not show_fig:
        print("\nExperiment complete. See the ../outputs/ directory for resulting visualizations and plots.\n"
              "To display the visualizations as the code runs, ensure the 'show_fig' argument is set to True")


def plot_3d(a_sigma_grid,
            b_sigma_grid,
            z_list,
            title,
            filetitle,
            x_axis_label=r'$\sigma_{a}$ as a % of $\mu_{a}$',
            y_axis_label=r'$\sigma_{b}$ as a % of $\mu_{b}$',
            show_fig=True,
            ):
    fig = plt.figure(figsize=[8.5,6.5])
    ax = fig.gca(projection='3d')
    a_sigma_grid = a_sigma_grid / 0.5 * 100     # Normalize by the mean 'a' value (from FTL-Bando) for more readable plot
    b_sigma_grid = b_sigma_grid / 20 * 100
    X, Y = np.meshgrid(a_sigma_grid, b_sigma_grid)
    Z = np.array(z_list).reshape((X.shape[0], X.shape[1]))
    # Plot the surface.
    ax.plot_surface(X, Y, Z, antialiased=False, cmap='viridis')
    # Customize the z axis.
    # ax.set_zlim(9.6, 9.9)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)
    plt.savefig("../outputs/"+filetitle)
    if show_fig:
        plt.show()
    plt.close()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    extension_two(a_sigma_start=0.,
                  a_sigma_stop=2.0,
                  a_sigma_granularity=0.1,
                  b_sigma_start=0.,
                  b_sigma_stop=60.0,
                  b_sigma_granularity=3.0,
                  show_fig=False)
