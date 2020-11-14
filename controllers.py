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

    def __init__(self, env):
        super().__init__(env)
        self.type = 'PID'
