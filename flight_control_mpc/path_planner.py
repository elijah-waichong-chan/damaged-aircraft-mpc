import numpy as np
import casadi as ca
from aircraft_model import AircraftModel

# --------------------------------------------------------------
# Path Planner Cost Weights
# --------------------------------------------------------------
WEIGHT_SMOOTH = 50.0                 # smoothness for x,y
WEIGHT_GS = 10.0                     # glideslope altitude error
WEIGHT_LAT = 1.0                     # lateral (cross-track) error
RW_ALIGN_WEIGHT = 1.0            # final runway heading alignment

# --------------------------------------------------------------
# Parameters
# --------------------------------------------------------------
GLIDE_ANGLE_DEG = 3.0                # target glide slope angle (deg)

ROLLOUT_DIST = 2000.0                # [m] along runway centerline after threshold
N_ROLLOUT    = 20                    # number of extra points along runway centerline
D_FINAL_ALIGN = 1000.0               # [m] desired straight-in segment length


class GlidePathPlanner:
    def __init__(self, runway_heading_deg, N):
        self.runway_heading_rad = np.deg2rad(runway_heading_deg)
        self.N = N

    def solve_for_waypoints(self, aircraft: AircraftModel, verbose=True):
        opti, z, n_wp = self._build_flight_path_qp(aircraft)
        sol = opti.solve()
        z_opt = np.array(sol.value(z)).flatten()
        base_waypoints = z_opt.reshape(n_wp, 3)

        # Append rollout points along runway centerline
        dx = np.cos(self.runway_heading_rad)
        dy = np.sin(self.runway_heading_rad)

        rollout_pts = []
        for i in range(1, N_ROLLOUT + 1):
            s = (ROLLOUT_DIST * i) / N_ROLLOUT
            rollout_pts.append([s * dx, s * dy, 0.0])

        rollout_pts = np.array(rollout_pts).reshape(-1, 3)
        self.waypoints = np.vstack([base_waypoints, rollout_pts])

        if verbose:
            start = self.waypoints[0]
            end   = self.waypoints[-1]
            print(
                "[FlightPathPlanner] ✅ Successfully solved flight path:\n"
                f"  - Waypoints: {self.waypoints.shape[0]}\n"
                f"  - Start:  N={start[0]:.1f}  E={start[1]:.1f}  h={start[2]:.1f}\n"
                f"  - End:    N={end[0]:.1f}  E={end[1]:.1f}  h={end[2]:.1f}"
            )

        return self.waypoints

    def _build_flight_path_qp(self, aircraft: AircraftModel):
        # Extract current position
        x0, y0, h0 = aircraft.pos_north, aircraft.pos_east, aircraft.altitude

        # Runway at origin
        x_end, y_end, h_end = 0.0, 0.0, 0.0

        N = self.N
        n_waypoint = N + 1
        n_var = 3 * n_waypoint

        def idx_x(i): return 3 * i
        def idx_y(i): return 3 * i + 1
        def idx_h(i): return 3 * i + 2

        opti = ca.Opti("conic")
        z = opti.variable(n_var)

        # Runway direction
        dx = np.cos(self.runway_heading_rad)
        dy = np.sin(self.runway_heading_rad)

        gamma = np.deg2rad(GLIDE_ANGLE_DEG)
        tan_gamma = np.tan(gamma)

        # --------------------------------------------------------------
        # Cost
        # --------------------------------------------------------------
        J = 0

        # Smoothness (second difference) for x,y,h
        for i in range(1, N):
            for d in range(3):
                w = WEIGHT_SMOOTH
                if w == 0.0:
                    continue
                id0 = 3 * (i - 1) + d
                id1 = 3 * i + d
                id2 = 3 * (i + 1) + d
                second_diff = z[id2] - 2 * z[id1] + z[id0]
                J += w * second_diff**2

        # Glideslope + centerline tracking (stronger near runway)
        for i in range(n_waypoint):
            xi = z[idx_x(i)]
            yi = z[idx_y(i)]
            hi = z[idx_h(i)]

            s_i = xi * dx + yi * dy        # signed along-runway
            s_dist = -s_i                  # distance-from-runway along approach direction

            # ✅ only enforce approach-side constraint at START and END (not every waypoint)
            if i == 0 or i == N:
                opti.subject_to(s_dist >= 0)

            c_i = -xi * dy + yi * dx       # cross-track distance
            h_ref = s_dist * tan_gamma     # glideslope altitude

            w_scale = (i / max(1, N))**2
            J += w_scale * WEIGHT_GS  * (hi - h_ref)**2
            J += w_scale * WEIGHT_LAT * (c_i)**2

        # --------------------------------------------------------------
        # Constraints (endpoints)
        # --------------------------------------------------------------
        opti.subject_to(z[idx_x(0)] == x0)
        opti.subject_to(z[idx_y(0)] == y0)
        opti.subject_to(z[idx_h(0)] == h0)

        opti.subject_to(z[idx_x(N)] == x_end)
        opti.subject_to(z[idx_y(N)] == y_end)
        opti.subject_to(z[idx_h(N)] == h_end)

        # --------------------------------------------------------------
        # Final alignment to runway heading (last segment window)
        # --------------------------------------------------------------
        D0 = float(np.hypot(x0, y0))
        if D0 > 1e-3:
            s_align_start = max(0.0, (D0 - D_FINAL_ALIGN) / D0)
            i_align_start = int(np.floor(s_align_start * N))

            for i in range(i_align_start, N):
                x_i   = z[idx_x(i)]
                x_ip1 = z[idx_x(i + 1)]
                y_i   = z[idx_y(i)]
                y_ip1 = z[idx_y(i + 1)]

                seg_dx = x_ip1 - x_i
                seg_dy = y_ip1 - y_i

                cross = seg_dx * dy - seg_dy * dx
                w_align = (i / max(1, N))**2
                J += w_align * RW_ALIGN_WEIGHT * cross**2

                # forward along runway direction
                opti.subject_to(seg_dx * dx + seg_dy * dy >= 0)

        # Minimize AFTER all costs added
        opti.minimize(J)

        # --------------------------------------------------------------
        # Solver setup
        # --------------------------------------------------------------
        opts = {
            "osqp": {
                "verbose": False,
                "polish": True,
                "max_iter": 4000,
                "eps_abs": 1e-4,
                "eps_rel": 1e-4,
            }
        }
        opti.solver("osqp", opts)

        return opti, z, n_waypoint




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from plot import (
        plot_3d_path,
        plot_ground_track,
        plot_altitude,
    )

    N = 50
    start_pos = (-5000.0, -15000.0, 1500.0)       # (x, y, h) in meters
    RUNWAY_HEADING_DEG = 90  # Runway heading in degrees

    aircraft = AircraftModel(pos_north=start_pos[0],
                                pos_east=start_pos[1],
                                altitude=start_pos[2],
                                vel_kt=100,
                                heading_deg=45.0,
                                climb_angle_deg=0.0,)
    
    planner = GlidePathPlanner(RUNWAY_HEADING_DEG, N)
    waypoints = planner.solve_for_waypoints(aircraft)

    x = waypoints[:, 0]
    y = waypoints[:, 1]
    h = waypoints[:, 2]

    # Plot 3D path
    plot_3d_path(x, y, h)

    # Plot 2D ground track
    plot_ground_track(
        x,
        y,
        RUNWAY_HEADING_DEG,
        heading_deg = 90.0,
    )

    # Plot altitude
    plot_altitude(
        x,
        y,
        h,
        RUNWAY_HEADING_DEG,
        glide_angle_deg=GLIDE_ANGLE_DEG,
    )

    plt.show()