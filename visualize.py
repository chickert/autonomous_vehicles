import matplotlib.pyplot as plt
import numpy as np

class RingRoad:
    def __init__(self, car_positions, road_length):
        self.num_vehicles = len(car_positions)
        self.road_length = road_length
        self.car_positions = car_positions

    def visualize(self):

        # Set the axes projection as polar
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        # plt.axes(projection='polar')

        # Find the radius of the ring given the RingRoad length
        road_radius = self.road_length / (2 * np.pi)

        # Plot the road
        radians = np.arange(0, (2 * np.pi), 0.01)
        for radian in radians:
            plt.polar(radian, road_radius, color='k', marker='.', markersize=20)

        # Now plot the cars after transforming the 1-dimensional location of each to the polar coordinate system
        for car_position in self.car_positions:
            # Check that modulo operation in RingRoad has worked, such that all car positions are <= road length
            assert car_position <= self.road_length, "car position is greater than length of road"

            # Transform the 1-D coord to polar system
            normalized_pos = car_position / self.road_length
            car_radian = normalized_pos * (2 * np.pi)

            # plot (with color according to whether it's an AV or human driver)
            plt.polar(car_radian, road_radius, color='r', marker='s', markersize=8.)
            # if car_position.is_AV:
            #     plt.polar(car_radian, road_radius, color='g', marker='s', markersize=8.)
            # else:
            #     plt.polar(car_radian, road_radius, color='r', marker='s', markersize=8.)

        # Format and plot
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.grid(False)
        plt.show()



my_car_positions = [10, 50, 145]
env = RingRoad(car_positions=my_car_positions, road_length=200)


env.visualize()