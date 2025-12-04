import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from aircraft_model import AircraftModel


# --------------------------------------------------------------
# Path Planner Cost Weights
# --------------------------------------------------------------
WEIGHT_SMOOTH = 10.0                # weight for path smoothness 
WEIGHT_HEIGHT_SMOOTH = 10.0         # weight for altitude smoothness
WEIGHT_LENGTH = 1.0                 # weight for path length
WEIGHT_GS = 10.0                    # weight for glideslope altitude error
WEIGHT_LAT = 10.0                   # weight for lateral (cross-track) error

K_INIT = 10                         # number of segments to gently align with initial heading
INIT_ALIGN_WEIGHT = 100.0           # weight for initial heading alignment
FINAL_ALIGN_WEIGHT = 100.0          # weight for final runway heading alignment

# --------------------------------------------------------------
# Parameters
# --------------------------------------------------------------
GLIDE_ANGLE_DEG = 3.0   # target glide slope angle (deg)

ROLLOUT_DIST = 800.0   # [m] along runway centerline after threshold
N_ROLLOUT    = 20      # number of extra points along runway centerline

D_ALIGN = 2000.0  # [m] length of desired straight-in segment


class GlidePathPlanner:
    def __init__(self, runway_heading_deg, N):
        self.runway_heading_rad = np.deg2rad(runway_heading_deg)
        self.N = N

    def solve_for_waypoints(self, aircraft: AircraftModel, verbose=True):
        # 1) Build and solve the flight path QP
        opti, z, n_wp = self._build_flight_path_qp(aircraft)
        sol = opti.solve()
        z_opt = np.array(sol.value(z)).flatten()
        base_waypoints = z_opt.reshape(n_wp, 3)     # (x, y, h) waypoints

        # 2) Append rollout points along runway centerline
        dx = np.cos(self.runway_heading_rad)
        dy = np.sin(self.runway_heading_rad)

        rollout_pts = []
        for i in range(1, N_ROLLOUT + 1):
            s = (ROLLOUT_DIST * i) / N_ROLLOUT  # distance along runway
            x_ext = s * dx
            y_ext = s * dy
            h_ext = 0.0                         # on the runway surface
            rollout_pts.append([x_ext, y_ext, h_ext])

        rollout_pts = np.array(rollout_pts).reshape(-1, 3)
        self.waypoints = np.vstack([base_waypoints, rollout_pts])

        start = self.waypoints[0]
        end   = self.waypoints[-1]

        # 3) Verbose output
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
        # Runway assumed at origin without loss of generality
        x_end, y_end, h_end = 0.0, 0.0, 0.0

        # Decision variable dimension
        N = self.N
        n_waypoint = N + 1
        n_var = 3 * n_waypoint

        # Index helpers from decision vector
        def idx_x(i): return 3 * i
        def idx_y(i): return 3 * i + 1
        def idx_h(i): return 3 * i + 2

        # Build optimization problem using CasADi's Opti stack
        opti = ca.Opti()

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

        # Length-like term
        if WEIGHT_LENGTH > 0.0:
            for i in range(N):
                for d in range(3):
                    id0 = 3 * i + d
                    id1 = 3 * (i + 1) + d
                    diff = z[id1] - z[id0]
                    J += WEIGHT_LENGTH * diff**2

        # Align first initial segments with initial heading
        for i in range(0, min(K_INIT, N)):
            x_i   = z[idx_x(i)]
            x_ip1 = z[idx_x(i + 1)]
            y_i   = z[idx_y(i)]
            y_ip1 = z[idx_y(i + 1)]

            seg_dx = x_ip1 - x_i
            seg_dy = y_ip1 - y_i

            # Cross product with initial heading: zero means parallel
            cross_h = seg_dx * hy - seg_dy * hx

            # Weight that can decay over these first K segments (optional)
            w_init = (1.0 - i / max(1, K_INIT))**2

            # Soft penalty to make early segments parallel to initial heading
            J += w_init * INIT_ALIGN_WEIGHT * cross_h**2

            # Optional: enforce "forward" along initial heading so it doesn't go backwards
            opti.subject_to(seg_dx * hx + seg_dy * hy >= 0)


        # Glideslope + centerline tracking
        dx = np.cos(self.runway_heading_rad)
        dy = np.sin(self.runway_heading_rad)

        gamma = np.deg2rad(GLIDE_ANGLE_DEG)  # glide slope angle (global)
        tan_gamma = np.tan(gamma)

        for i in range(n_waypoint):
            xi = z[idx_x(i)]
            yi = z[idx_y(i)]
            hi = z[idx_h(i)]

            # Signed along-runway coordinate (can be negative)
            s_i = xi * dx + yi * dy  # runway at origin

            # Use absolute distance from runway along the runway axis
            s_dist = ca.fabs(s_i)    # |s_i| so altitude is always >= 0 for positive distance

            # Cross-track error (lateral distance to runway centerline)
            c_i = xi * (-dy) + yi * dx

            # Reference altitude on glide slope from runway
            # h_ref = distance_from_runway * tan(gamma)
            h_ref = s_dist * tan_gamma   # h_end = 0 at origin

            # Weight grows towards the runway
            w_scale = (i / N)**2

            J += w_scale * WEIGHT_GS  * (hi - h_ref)**2
            J += w_scale * WEIGHT_LAT * c_i**2

        opti.minimize(J)

        # --------------------------------------------------------------
        # Constraints
        # --------------------------------------------------------------

        # Initial position
        opti.subject_to(z[idx_x(0)] == x0)
        opti.subject_to(z[idx_y(0)] == y0)
        opti.subject_to(z[idx_h(0)] == h0)

        # End position (runway at origin)
        opti.subject_to(z[idx_x(N)] == x_end)
        opti.subject_to(z[idx_y(N)] == y_end)
        opti.subject_to(z[idx_h(N)] == h_end)

        # Final-alignment with runway heading
        # Horizontal distance from start to runway at origin
        D0 = np.hypot(x0, y0)

        if D0 > 1e-3:
            # Fraction of the path where we start alignment
            s_align_start = max(0.0, (D0 - D_ALIGN) / D0)
            i_align_start = int(np.floor(s_align_start * N))

            for i in range(i_align_start, N):
                x_i   = z[idx_x(i)]
                x_ip1 = z[idx_x(i + 1)]
                y_i   = z[idx_y(i)]
                y_ip1 = z[idx_y(i + 1)]

                seg_dx = x_ip1 - x_i
                seg_dy = y_ip1 - y_i

                # Parallel to runway: cross = 0
                cross = seg_dx * dy - seg_dy * dx
                w_align = (i / N)**2
                J += w_align * FINAL_ALIGN_WEIGHT * cross**2

                # Forward along runway direction (relative to heading convention)
                opti.subject_to(seg_dx * dx + seg_dy * dy >= 0)

        # --------------------------------------------------------------
        # Solver setup
        # --------------------------------------------------------------

        ## Even though we are using IPOPT here, the problem is a QP.
        ## In a more complete implementation, we could use a dedicated QP solver.
        opti.solver('ipopt', {
            "print_time": False,
            "ipopt.print_level": 0,
        })

        self.opti = opti
        self.z = z
        self.n_waypoint = n_waypoint

        return opti, z, n_waypoint



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from plot import (
        plot_3d_path,
        plot_ground_track,
        plot_altitude,
    )

    N = 50
    start_pos = (3000.0, -5000.0, 500.0)    # (x, y, h) in meters
    RUNWAY_HEADING_DEG = 235  # Runway heading in degrees

    aircraft = AircraftModel(pos_north=5000.0,
                                pos_east=5600,
                                altitude=1000.0,
                                vel=50,
                                heading_deg=30.0,
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
        heading_ang = 0.0,
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
