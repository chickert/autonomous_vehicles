import numpy as np
import pandas as pd


class Controller:

    """
    Controller superclass.
    """

    def __init__(self, env):
        self.env = env
        self.type = None  # 'BandoFTL' or 'PID', etc

    def calculate(self, this_vehicle):
        """
        Calculate a velocity update for this vehicle
        given current state information that can be
        obtained from this vechicle and/or the environment.
        """
        return 0.0  # Null controller.


class BandoFTL(Controller):

    def __init__(self, env, a, b):
        super().__init__(env)
        self.type = 'BandoFTL'
        self.a = a
        self.b = b

    def calculate(self, this_vehicle):
        """
        Provide a contoller update for this vehicle.
        """
        assert self.env == this_vehicle.env, "Cannot mix environments."
        
        # TEST #
        if self.env.t <= 10.0:
            return 0.25
        else:
            return 0.0 + self.env.random.normal()


class PID(Controller):

    """
    PID Controller for the autonomous vehicle.
    The `update` method takes a Vehicle object and 
    returns a control (velocity delta) to apply to its velocity.
    The required information is this vehicle's recent velocity history 
    and the distance to and velocity of its lead vehicle (i.e. the one directly in front).
    """

    def __init__(self, env, safe_distance, gamma, m):
        """
        PID Controller from Delle Monarche et al. (2019).

        safe_distance : 
            Ccorresponds to delta{x}^s in eqn. 12.15.
        gamma : 
            Controls rate at which alpha transitions from 0 to 1; paper uses gamma = 2 meters.
        m : 
            number of velocity measurements for computing desired velocity v_d; paper uses m = 38.
        """
        super().__init__(env)
        self.type = 'PID'

        self.safe_distance = safe_distance
        self.gamma = gamma

        self.stored_command_velocity = None
        self.m = m
        self.stored_velocities = []

    def calc_alpha(self, delta_x):
        target = (delta_x - self.safe_distance) / self.gamma   # delta_x corresponds to current distance between AV and lead vehicle
        alpha_j = min(max(target, 0), 1)
        return alpha_j

    def calc_beta(self):
        """
        Currently unclear on how to calc this
        """
        print("TODO: Calculate beta.")
        return 1.0  # Arbitrary test value.

    def calc_desired_velocity(self, velocity_history):
        """
        Question here is in regards to paragraph just above eqn 12.13:
        It says to average over the "autonomous vehicle velocities over the last m measurements"
        I assume this is the true velocities; can we assume that the commanded velocity is the same as this?
        I make that assumption below, but we should discuss or just double-check
        """
        v_d = np.mean(velocity_history)

        """
        NEED TO UPDATE this to take self.m * (1/time_discretization) so for m=38 and time disc of 0.1s, it takes 380
        values instead of 38 values! 
        """
        return v_d

    def calc_target_velocity(self, delta_x, velocity_history):
        """
        The target here is very similar to eqn in self.calc_alpha(); same structure, but hard-coded values.
        So I'm wondering if they should be the same?
        If so, why don't they have alpha_j in eqn 12.13 where v_target is defined?
        If not, what do the 7 and 23 correspond to, and why are they hard-coded?
        """
        v_d = self.calc_desired_velocity(velocity_history)

        target = (delta_x - 7) / 23
        v_target = v_d + 1 * min(max(target, 0), 1)
        return v_target

    def update_command_velocity(self, delta_x, velocity_history, lead_vehicle_velocity):
        vj_target = self.calc_target_velocity(delta_x, velocity_history)
        prior_uj = self.stored_command_velocity
        vj_lead = lead_vehicle_velocity

        alpha_j = self.calc_alpha(delta_x)
        beta_j = self.calc_beta()

        new_uj = beta_j * (alpha_j * vj_target + (1 - alpha_j) * vj_lead) + (1 - beta_j) * prior_uj

        updated_command_velocity = new_uj
        self.stored_command_velocity = updated_command_velocity
        """
        NOTE [relevant to calc_desired_velocity() above]:
        Here I assume that the updated command velocity is equal to the actual vehicle velocity, but need to double-check
        that what is commanded is actually implemented, and there's no 'stickiness' or other factors that affect the actual
        implementation of the commanded velocity
        (such as if actual velocity was a moving average of commanded velocities, for example)
        """
        self.stored_velocities.append(self.stored_command_velocity)
        return updated_command_velocity

    def calculate(self, this_vehicle):
        """
        Provide a contoller update for this vehicle.
        """
        assert self.env == this_vehicle.env, "Cannot mix environments."

        # Get distance to lead vehicle and its velocity:
        lead_vehicle = self.env.get_lead_vehicle(this_vehicle)
        delta_x = this_vehicle.pos.distance_to(lead_vehicle.pos)
        lead_vehicle_velocity = lead_vehicle.vel

        # Get relevant velocity history:
        first_step = max(0, self.env.step - int(self.m / self.env.dt))  # Divide by temporal resolution.
        last_step = self.env.step
        state_history = this_vehicle.get_state_table(steps=range(first_step,last_step))
        velocity_history = np.array(state_history['pos'])

        # Calculate command velocity and return control:
        command_velocity = self.update_command_velocity(delta_x, velocity_history, lead_vehicle_velocity)
        current_velocity = this_vehicle.vel
        delta_velocity = command_velocity - current_velocity

        return delta_velocity
