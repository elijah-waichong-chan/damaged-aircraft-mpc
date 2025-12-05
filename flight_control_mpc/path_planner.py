import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from aircraft_model import AircraftModel


# --------------------------------------------------------------
# Path Planner Cost Weights
# --------------------------------------------------------------
WEIGHT_SMOOTH = 500.0                # weight for path smoothness 
WEIGHT_HEIGHT_SMOOTH = 10.0          # weight for altitude smoothness
WEIGHT_LENGTH = 1.0                  # weight for path length
WEIGHT_GS = 50.0                     # weight for glideslope altitude error
WEIGHT_LAT = 10.0                    # weight for lateral (cross-track) error

INIT_ALIGN_WEIGHT = 1.0              # weight for initial heading alignment
FINAL_ALIGN_WEIGHT = 10.0            # (not used in QP now, but kept for tuning if needed)

# --------------------------------------------------------------
# Parameters
# --------------------------------------------------------------
GLIDE_ANGLE_DEG = 3.0   # target glide slope angle (deg)

ROLLOUT_DIST = 2000.0   # [m] along runway centerline after threshold
N_ROLLOUT    = 20       # number of extra points along runway centerline

D_INIT_ALIGN  = 5.0       # [m] length of initial alignment segment (very small right now)
D_FINAL_ALIGN = 10000.0   # [m] length of desired straight-in segment BEFORE runway


class GlidePathPlanner:
    def __init__(self, runway_heading_deg, N):
        self.runway_heading_rad = np.deg2rad(runway_heading_deg)
        self.N = N

    def solve_for_waypoints(self, aircraft: AircraftModel, verbose=True):
        # 1) Build and solve the flight path QP (aircraft -> merge point)
        opti, z, n_wp = self._build_flight_path_qp(aircraft)
        sol = opti.solve()
        z_opt = np.array(sol.value(z)).flatten()
        base_waypoints = z_opt.reshape(n_wp, 3)     # (x, y, h) waypoints
        # base_waypoints[-1] is the merge point 10 km before runway

        dx = np.cos(self.runway_heading_rad)
        dy = np.sin(self.runway_heading_rad)
        gamma = np.deg2rad(GLIDE_ANGLE_DEG)
        tan_gamma = np.tan(gamma)

        # 2) Build explicit 10 km straight-in from merge point to runway
        x_merge, y_merge, h_merge = base_waypoints[-1]

        # Distance from merge to runway along centerline is D_FINAL_ALIGN by design,
        # but we'll parametrize along s from -D_FINAL_ALIGN -> 0.
        N_STRAIGHT = 80  # resolution of straight-in segment
        straight_pts = []
        for j in range(1, N_STRAIGHT + 1):
            tau = j / N_STRAIGHT  # 0 -> at merge, 1 -> at runway
            s = -D_FINAL_ALIGN * (1.0 - tau)   # from -D_FINAL_ALIGN up to 0

            x_s = s * dx          # North
            y_s = s * dy          # East
            s_dist = -s           # distance FROM runway along approach (>= 0)
            h_s = s_dist * tan_gamma

            straight_pts.append([x_s, y_s, h_s])

        straight_pts = np.array(straight_pts).reshape(-1, 3)

        # 3) Append rollout points along runway centerline (beyond threshold)
        rollout_pts = []
        for i in range(1, N_ROLLOUT + 1):
            s = (ROLLOUT_DIST * i) / N_ROLLOUT  # distance ALONG runway AFTER threshold
            x_ext = s * dx
            y_ext = s * dy
            h_ext = 0.0                         # on the runway surface
            rollout_pts.append([x_ext, y_ext, h_ext])

        rollout_pts = np.array(rollout_pts).reshape(-1, 3)

        # 4) Concatenate:
        #    aircraft -> merge (QP), then merge -> runway (straight), then rollout
        self.waypoints = np.vstack([base_waypoints, straight_pts, rollout_pts])

        start = self.waypoints[0]
        end   = self.waypoints[-1]

        if verbose:
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

        # Initial heading direction (unit vector in NE frame)
        chi0 = aircraft.chi                 # radians
        hx = np.cos(chi0)                   # heading x (north)
        hy = np.sin(chi0)                   # heading y (east)

        # Decision variable dimension
        N = self.N
        n_waypoint = N + 1
        n_var = 3 * n_waypoint

        # Index helpers from decision vector
        def idx_x(i): return 3 * i
        def idx_y(i): return 3 * i + 1
        def idx_h(i): return 3 * i + 2

        opti = ca.Opti('conic')

        # Decision variables (x0, y0, h0, ..., xN, yN, hN)
        z = opti.variable(n_var)

        # --------------------------------------------------------------
        # Cost function
        # --------------------------------------------------------------
        J = 0

        # Smoothness term
        for i in range(1, N):
            for d in range(3):
                w = WEIGHT_HEIGHT_SMOOTH if d == 2 else WEIGHT_SMOOTH
                if w == 0.0:
                    continue

                id0 = 3 * (i - 1) + d
                id1 = 3 * i + d
                id2 = 3 * (i + 1) + d

                second_diff = z[id2] - 2 * z[id1] + z[id0]
                J += w * second_diff**2

        # Path length term
        if WEIGHT_LENGTH > 0.0:
            for i in range(N):
                for d in range(3):
                    id0 = 3 * i + d
                    id1 = 3 * (i + 1) + d
                    diff = z[id1] - z[id0]
                    J += WEIGHT_LENGTH * diff**2

        # Glideslope + centerline tracking
        dx = np.cos(self.runway_heading_rad)
        dy = np.sin(self.runway_heading_rad)

        gamma = np.deg2rad(GLIDE_ANGLE_DEG)  # glide slope angle (global)
        tan_gamma = np.tan(gamma)

        # Merge point 10 km before runway along runway centerline
        s_merge = -D_FINAL_ALIGN                      # negative = before runway
        x_end = s_merge * dx                          # North
        y_end = s_merge * dy                          # East
        h_end = D_FINAL_ALIGN * tan_gamma             # altitude on 3 deg glideslope at that distance

        for i in range(n_waypoint):
            xi = z[idx_x(i)]
            yi = z[idx_y(i)]
            hi = z[idx_h(i)]

            # Signed along-runway coordinate: s_i < 0 is before threshold
            s_i = xi * dx + yi * dy

            # Distance from runway along approach (>= 0)
            s_dist = -s_i
            opti.subject_to(s_dist >= 0)  # keep all waypoints before runway for the QP

            # Cross-track error (lateral distance to runway centerline)
            c_i = xi * (-dy) + yi * dx

            # Reference altitude on glide slope from runway
            h_ref = s_dist * tan_gamma

            # Weight grows towards the merge point (approximate runway)
            w_scale = (i / N)**2

            J += w_scale * WEIGHT_GS  * (hi - h_ref)**2
            J += w_scale * WEIGHT_LAT * c_i**2

        # --------------------------------------------------------------
        # Constraints
        # --------------------------------------------------------------

        # Initial position = aircraft
        opti.subject_to(z[idx_x(0)] == x0)
        opti.subject_to(z[idx_y(0)] == y0)
        opti.subject_to(z[idx_h(0)] == h0)

        # End position = merge point 10 km before runway
        opti.subject_to(z[idx_x(N)] == x_end)
        opti.subject_to(z[idx_y(N)] == y_end)
        opti.subject_to(z[idx_h(N)] == h_end)

        # Initial-heading alignment (very short region right now because D_INIT_ALIGN is small)
        D0 = np.hypot(x0, y0)   # distance from start to runway

        if D0 > 1e-3:
            s_init_end = min(D_INIT_ALIGN, D0)
            frac_init = s_init_end / D0
            i_init_end = int(np.floor(frac_init * N))

            for i in range(0, max(1, i_init_end)):
                x_i   = z[idx_x(i)]
                x_ip1 = z[idx_x(i + 1)]
                y_i   = z[idx_y(i)]
                y_ip1 = z[idx_y(i + 1)]

                seg_dx = x_ip1 - x_i
                seg_dy = y_ip1 - y_i

                # Parallel to initial heading: cross = 0
                cross_h = seg_dx * hy - seg_dy * hx

                # Weight that decays with i
                w_init = (1.0 - i / max(1, i_init_end))**2

                J += w_init * INIT_ALIGN_WEIGHT * cross_h**2

                # Enforce "forward" along initial heading
                opti.subject_to(seg_dx * hx + seg_dy * hy >= 0)


        # --------------------------------------------------------------
        # Initial-heading alignment (unchanged)
        # --------------------------------------------------------------
        D0 = np.hypot(x0, y0)   # distance from start to runway

        if D0 > 1e-3:
            s_init_end = min(D_INIT_ALIGN, D0)
            frac_init = s_init_end / D0
            i_init_end = int(np.floor(frac_init * N))

            for i in range(0, max(1, i_init_end)):
                x_i   = z[idx_x(i)]
                x_ip1 = z[idx_x(i + 1)]
                y_i   = z[idx_y(i)]
                y_ip1 = z[idx_y(i + 1)]

                seg_dx = x_ip1 - x_i
                seg_dy = y_ip1 - y_i

                # Parallel to initial heading: cross = 0
                cross_h = seg_dx * hy - seg_dy * hx

                # Weight that decays with i
                w_init = (1.0 - i / max(1, i_init_end))**2

                J += w_init * INIT_ALIGN_WEIGHT * cross_h**2

                # Enforce "forward" along initial heading
                opti.subject_to(seg_dx * hx + seg_dy * hy >= 0)

        # --------------------------------------------------------------
        # NEW: make the *last* segment normal to the runway direction
        # --------------------------------------------------------------
        # Last segment: from waypoint N-1 to N (merge point)
        x_prev = z[idx_x(N - 1)]
        y_prev = z[idx_y(N - 1)]
        x_last = z[idx_x(N)]
        y_last = z[idx_y(N)]

        seg_dx_final = x_last - x_prev   # North component
        seg_dy_final = y_last - y_prev   # East component

        # Runway direction unit vector: (dx, dy)
        # Normal means: seg · (dx, dy) ≈ 0
        dot_along = seg_dx_final * dx + seg_dy_final * dy

        # Penalize any along-runway component of the final segment
        J += FINAL_ALIGN_WEIGHT * dot_along**2



        # --------------------------------------------------------------
        # Finalize cost
        # --------------------------------------------------------------
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
        opti.solver('osqp', opts)

        self.opti = opti
        self.z = z
        self.n_waypoint = n_waypoint

        return opti, z, n_waypoint


# Demo / test
if __name__ == "__main__":
    from plot import (
        plot_3d_path,
        plot_ground_track,
        plot_altitude,
    )

    N = 50
    RUNWAY_HEADING_DEG = 235  # Runway heading in degrees

    aircraft = AircraftModel(
        pos_north=5000.0,
        pos_east=5600.0,
        altitude=1000.0,
        vel_kt=100,
        heading_deg=90.0,
        climb_angle_deg=0.0,
    )

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
        heading_deg=90.0,
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
