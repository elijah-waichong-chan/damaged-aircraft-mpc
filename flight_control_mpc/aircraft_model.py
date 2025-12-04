import numpy as np
from numpy import sin, cos

# --------------------------------------------------------------
# Aircraft Performance Constants
# --------------------------------------------------------------
BEST_GLIDE_SPEED_KT = 65          # best glide speed (kt)
APPROACH_SPEED_KT = 80            # approach speed (kt)
STALL_SPEED_KT = 50.0             # stall speed (kt)
NEVER_EXCEED_SPEED_KT = 150.0     # never exceed speed (kt)

class AircraftModel:
    def __init__(self, pos_north, pos_east, altitude, vel_kt, heading_deg, climb_angle_deg, dt = 0.05):
        # Initial Configuration
        self.pos_north = pos_north                  # North Position (m)
        self.pos_east = pos_east                    # East Position (m)
        self.altitude = altitude                    # Altitude (m)
        self.vel_ms = vel_kt * 0.514444             # Airspeed (m/s) converted from knots
        self.chi = np.deg2rad(heading_deg)          # Heading angle(rad)
        self.gamma = np.deg2rad(climb_angle_deg)    # Flight path (climb) angle (rad)

        self.glide_speed_ms = BEST_GLIDE_SPEED_KT * 0.514444               # Best glide speed (m/s) converted from knots
        self.stall_speed_ms = STALL_SPEED_KT * 0.514444                    # Stall speed (m/s) converted from knots
        self.never_exceed_speed_ms = NEVER_EXCEED_SPEED_KT * 0.514444      # Never exceed speed (m/s) converted from knots
        self.approach_speed_ms = APPROACH_SPEED_KT * 0.514444              # Approach speed (m/s) converted from knots
        self.dt = dt

        self._update_linearized_kinematics()

    def get_state_vector(self):
        return np.array([
            self.pos_north,
            self.pos_east,
            self.altitude,
            self.vel_ms,
            self.chi,
            self.gamma,
        ])
    
    def update_from_vector(self, x):
        self.pos_north, self.pos_east, self.altitude, \
        self.vel_ms, self.chi, self.gamma = x

        self._update_linearized_kinematics()

    def _update_linearized_kinematics(self):
        """ 
        Update the dynamics matricies using linearized kinematics model 
        """
        V = self.vel_ms
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
            [0.0, 0.0, 0.0,  0.0,             0.0,                -9.81 * cgamma],
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