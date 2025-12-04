import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from aircraft_model import AircraftModel
from path_planner import GlidePathPlanner
from guidance_mpc import GuidanceMPC
from plot import plot_guidance_overview

# --------------------------------------------------------------
# Controller Settings
# --------------------------------------------------------------
MPC_DT = 1          # MPC time step (s)
MPC_N = 10          # MPC prediction horizon step
PLANNER_N = 50      # Number of waypoints in path planner

# --------------------------------------------------------------
# Environment and Airplane Initial Conditions
# --------------------------------------------------------------
RUNWAY_HEADING_DEG = 70                            # Runway heading in degrees
AIRPLANE_START_POS = (3000.0, -5000.0, 500.0)        # (x, y, h) in meters
AIRPLANE_START_VEL_KT = 100.0                       # initial speed (kt)
AIRPLANE_START_HEADING_DEG = 0.0                    # initial heading (deg)

# --------------------------------------------------------------
# End of settings
# --------------------------------------------------------------

aircraft = AircraftModel(pos_north=AIRPLANE_START_POS[0],
                            pos_east=AIRPLANE_START_POS[1],
                            altitude=AIRPLANE_START_POS[2],
                            vel_kt=AIRPLANE_START_VEL_KT,
                            heading_deg=AIRPLANE_START_HEADING_DEG,
                            climb_angle_deg=0.0,
                            dt=MPC_DT)
planner = GlidePathPlanner(RUNWAY_HEADING_DEG, PLANNER_N)
guidance_mpc = GuidanceMPC(planner, aircraft, MPC_N, MPC_DT)

x_planned = []
y_planned = []
h_planned = []

x_mpc = []
y_mpc = []
h_mpc = []

u_thrust_mpc_m_s2 = []
u_heading_rate_mpc_deg_s = []
u_climb_rate_mpc_deg_s = []

x_sim_m = []
y_sim_m = []
h_sim_m = []
vel_sim_ms = []
heading_sim_deg = []
climb_angle_sim_deg = []
x_sim_m.append(aircraft.pos_north)
y_sim_m.append(aircraft.pos_east)
h_sim_m.append(aircraft.altitude)
vel_sim_ms.append(aircraft.vel_ms)
heading_sim_deg.append(np.rad2deg(aircraft.chi))
climb_angle_sim_deg.append(np.rad2deg(aircraft.gamma))

sim_step = 0
while h_sim_m[-1] > 0.1:

    # 1) Solve MPC for control input and trajectory
    try:
        state_vector = aircraft.get_state_vector()
        u0, X_pred, U_pred, waypoints = guidance_mpc.solve_for_control_input(aircraft)
    except RuntimeError as e:
        print("MPC failed")
        break

    # 2) Store control inputs
    u_thrust_mpc_m_s2.append(u0[0])
    u_heading_rate_mpc_deg_s.append(np.rad2deg(u0[1]))
    u_climb_rate_mpc_deg_s.append(np.rad2deg(u0[2]))    

    # 3) Store planned waypoints
    x_planned.append(waypoints[:, 0])
    y_planned.append(waypoints[:, 1])
    h_planned.append(waypoints[:, 2])

    # 4) Store MPC predicted trajectory
    x_mpc.append(X_pred[0, :])   # N
    y_mpc.append(X_pred[1, :])   # E
    h_mpc.append(X_pred[2, :])   # h

    # 5) Store simulated state (next time step)
    x_sim_m.append(X_pred[0, 1])
    y_sim_m.append(X_pred[1, 1])
    h_sim_m.append(X_pred[2, 1])
    vel_sim_ms.append(X_pred[3, 1])
    heading_sim_deg.append(np.rad2deg(X_pred[4, 1]))
    climb_angle_sim_deg.append(np.rad2deg(X_pred[5, 1]))

    # 6) Update aircraft state
    aircraft.update_from_vector(X_pred[:,1])

    sim_step = sim_step + 1


# --------------------------------------------------------------
# Play Simulation Results
# --------------------------------------------------------------

plt.ion()
replay_step = 0

while True:

    plot_guidance_overview(
        x_planned[replay_step],
        y_planned[replay_step],
        h_planned[replay_step],
        RUNWAY_HEADING_DEG,
        x_mpc=x_mpc[replay_step],
        y_mpc=y_mpc[replay_step],
        h_mpc=h_mpc[replay_step],
        u_thrust=u_thrust_mpc_m_s2[replay_step],
        u_heading=u_heading_rate_mpc_deg_s[replay_step],
        u_climb_rate=u_climb_rate_mpc_deg_s[replay_step],
        vel_ms=vel_sim_ms[replay_step],
        heading_deg=heading_sim_deg[replay_step],
        climb_angle_deg=climb_angle_sim_deg[replay_step],
    )

    if replay_step == 0:
        plt.pause(3.0)

    replay_step = replay_step + 1
    if replay_step >= len(x_planned):
        plt.pause(3.0)
        replay_step = 0

    plt.pause(0.1)