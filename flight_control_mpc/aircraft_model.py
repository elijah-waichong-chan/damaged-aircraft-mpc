import numpy as np
from numpy import sin, cos

class AircraftModel:
    def __init__(self, pos_north, pos_east, altitude, vel, chi, gamma, dt = 0.05):
        # Initial Configuration
        self.pos_north = pos_north      # North Position (m)
        self.pos_east = pos_east        # East Position (m)
        self.altitude = altitude        # Altitude (m)
        self.vel = vel                  # Airspeed (m/s)
        self.chi = chi                  # Heading angle(rad)
        self.gamma = gamma              # Flight path (climb) angle (rad)
        self.dt = dt

    def get_state_vector(self):
        return np.array([
            self.pos_north,
            self.pos_east,
            self.altitude,
            self.vel,
            self.chi,
            self.gamma,
        ])
    
    def update_from_vector(self, x):
        self.pos_north, self.pos_east, self.altitude, \
        self.vel, self.chi, self.gamma = x

        self._update_linearized_kinematics()

    def _update_linearized_kinematics(self):
        """ 
        Update the dynamics matricies using linearized kinematics model 
        """
        V = self.vel
        gamma = self.gamma
        chi = self.chi
        dt = self.dt

        cgamma = cos(gamma)
        sgamma = sin(gamma)
        cchi = cos(chi)
        schi = sin(chi)

        # Continuous-time state-space matrices linearized about current state
        Ac = np.array([
            [0.0, 0.0, 0.0,  cgamma * cchi,  -V * cgamma * schi,  -V * sgamma * cchi],
            [0.0, 0.0, 0.0,  cgamma * schi,   V * cgamma * cchi,  -V * sgamma * schi],
            [0.0, 0.0, 0.0,  sgamma,          0.0,                 V * cgamma],
            [0.0, 0.0, 0.0,  0.0,             0.0,                 0.0],
            [0.0, 0.0, 0.0,  0.0,             0.0,                 0.0],
            [0.0, 0.0, 0.0,  0.0,             0.0,                 0.0],
        ])

        Bc = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        # Discretize using Euler forward method
        self.Ad = np.eye(6) + Ac * dt
        self.Bd = Bc * dt