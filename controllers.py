import numpy as numpy


class Controller:

    def __init__(self, env):
        self.env = env
        self.type = None  # 'BandoFTL' or 'PID', etc


class BandoFTL(Controller):

    def __init__(self, env, a, b):
        super().__init__(env)
        self.type = 'BandoFTL'
        self.a = a
        self.b = b


class PID(Controller):

    def __init__(self, env, safety_distance, gamma, m):
        super().__init__(env)
        self.type = 'PID'

        self.safety_distance = safety_distance    # corresponds to delta{x}^s in eqn. 12.15
        self.gamma = gamma                       # controls rate at which alpha transitions from 0 to 1

        self.stored_command_velocity = None
        self.m = m                          # number of velocity measurements for computing desired velocity v_d; paper uses m = 38
        self.stored_velocities = []

    def calc_alpha(self, delta_x):
        target = (delta_x - self.safety_distance) / self.gamma   # delta_x corresponds to current distance between AV and lead vehicle
        alpha_j = min(max(target, 0), 1)
        return alpha_j

    def calc_beta(self):
        """
        Currently unclear on how to calc this
        """
        pass

    def calc_desired_velocity(self):
        """
        Question here is in regards to paragraph just above eqn 12.13:
        It says to average over the "autonomous vehicle velocities over the last m measurements"
        I assume this is the true velocities; can we assume that the commanded velocity is the same as this?
        I make that assumption below, but we should discuss or just double-check
        """
        v_d = np.mean(self.stored_velocities[-self.m:])

        """
        NEED TO UPDATE this to take self.m * (1/time_discretization) so for m=38 and time disc of 0.1s, it takes 380
        values instead of 38 values! 
        """
        return v_d

    def calc_target_velocity(self, delta_x):
        """
        The target here is very similar to eqn in self.calc_alpha(); same structure, but hard-coded values.
        So I'm wondering if they should be the same?
        If so, why don't they have alpha_j in eqn 12.13 where v_target is defined?
        If not, what do the 7 and 23 correspond to, and why are they hard-coded?
        """
        v_d = self.calc_desired_velocity()

        target = (delta_x - 7) / 23
        v_target = v_d + 1 * min(max(target, 0), 1)
        return v_target

    def update_command_velocity(self, lead_vehicle_velocity, delta_x):
        vj_target = self.calc_target_velocity()
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