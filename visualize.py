import matplotlib.pyplot as plt
import numpy as np

class RingRoad:
    def __init__(self, car_positions, road_length, car_length):
        self.num_vehicles = len(car_positions)
        self.road_length = road_length
        self.car_positions = car_positions
        self.car_length = car_length

        # Attributes for plotting
        self.road_width = 20.
        self.scaled_car_width = 10.
        self.point_car_size = 8.
        self.road_color = 'silver'
        self.hv_color = 'firebrick'
        self.av_color = 'seagreen'

    def visualize(self, draw_cars_to_scale=False):

        # Set the axes projection as polar
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        # plt.axes(projection='polar')

        # Find the radius of the ring given the RingRoad length
        road_radius = self.road_length / (2 * np.pi)

        # Plot the road
        radians = np.arange(0, (2 * np.pi), 0.01)
        for radian in radians:
            plt.polar(radian, road_radius, color=self.road_color, marker='.', markersize=self.road_width)

        # Now plot the cars after transforming the 1-dimensional location of each to the polar coordinate system
        for car_position in self.car_positions:
            # Check that modulo operation in RingRoad has worked, such that all car positions are <= road length
            assert car_position <= self.road_length, "car position is greater than length of road"

            # Transform the 1-D coord to polar system
            normalized_pos = car_position / self.road_length
            car_radian = normalized_pos * (2 * np.pi)

            # Now plot the cars, whether to scale or not, with color according to whether each is an AV or human driver
            # Note: for large ring roads, it is likely better to NOT draw to scale, for easier visualization
            if draw_cars_to_scale:
                normalized_car_length = self.car_length / self.road_length
                polar_car_length = normalized_car_length * (2 * np.pi)
                car_arc = np.arange(start=car_radian - polar_car_length/2,
                                    stop=car_radian + polar_car_length/2,
                                    step=0.005)
                for car_point in car_arc:
                    plt.polar(car_point, road_radius, color=self.hv_color, marker='.', markersize=self.scaled_car_width)
                # if car_position.is_AV:
                #     for car_point in car_arc:
                #         plt.polar(car_point, road_radius, color=self.av_color, marker='.', markersize=self.scaled_car_width)
                # else:
                #     for car_point in car_arc:
                #         plt.polar(car_point, road_radius, color=self.hv_color, marker='.', markersize=self.scaled_car_width)

            else:
                plt.polar(car_radian, road_radius, color=self.hv_color, marker='s', markersize=self.point_car_size)
                # if car_position.is_AV:
                #     plt.polar(car_radian, road_radius, color=self.av_color, marker='s', markersize=self.point_car_size)
                # else:
                #     plt.polar(car_radian, road_radius, color=self.hv_color, marker='s', markersize=self.point_car_size)

        # Format and plot
        if draw_cars_to_scale:
            print("Drawing cars to scale")
        elif not draw_cars_to_scale:
            print("Not drawing cars to scale")
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.spines['polar'].set_visible(False)
        plt.grid(False)
        plt.show()



my_car_positions = [10, 50, 245]
env = RingRoad(car_positions=my_car_positions, road_length=250, car_length=7)


env.visualize(draw_cars_to_scale=True)