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
        self._n_base = base_waypoints.shape[0]
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
    def polyline_length_xy(self, waypoints):
        d = np.diff(waypoints[:, :2], axis=0)
        return float(np.sum(np.linalg.norm(d, axis=1)))
    
    def polyline_length_to_threshold_xy(self, waypoints):
        wp = np.asarray(waypoints, dtype=float)
        n_base = getattr(self, "_n_base", wp.shape[0])
        wp = wp[:n_base, :]          # only up to runway threshold
        d = np.diff(wp[:, :2], axis=0)
        return float(np.sum(np.linalg.norm(d, axis=1)))
    def remaining_length_to_threshold_xy(self, waypoints, s0):
        wp = np.asarray(waypoints, dtype=float)
        n_base = getattr(self, "_n_base", wp.shape[0])
        wp_base = wp[:n_base, :]

        # arc-length along the base polyline
        d = np.diff(wp_base[:, :2], axis=0)
        seg = np.linalg.norm(d, axis=1)
        s_wp = np.zeros(wp_base.shape[0])
        s_wp[1:] = np.cumsum(seg)

        s_end = float(s_wp[-1])
        return max(0.0, s_end - float(s0))


    def max_horizontal_range(self, altitude_m, gamma_max_rad):
        """
        Max range before hitting ground if gamma is constrained to be <= gamma_max_rad < 0.
        Shallowest descent is gamma_max_rad (least negative). Range ≈ h / tan(|gamma_max|).
        """
        g = abs(float(gamma_max_rad))
        g = max(g, np.deg2rad(1.0))  # avoid divide-by-zero
        return float(altitude_m / np.tan(g))
    def solve_for_crash_waypoints(self, aircraft, gamma_max_rad, no_land_zones, verbose=False):
        # 1) compute allowed horizontal range (same geometry)
        R_max = self.max_horizontal_range(aircraft.altitude, gamma_max_rad)
        def ellipse_value(p, z):
            x, y = float(p[0]), float(p[1])
            cx, cy = float(z["cx"]), float(z["cy"])
            a, b   = float(z["a"]),  float(z["b"])
            return ((x - cx)**2)/(a**2) + ((y - cy)**2)/(b**2)

        def find_containing_zone(p, zones):
            """Return a zone that contains p (v<=1). If multiple, return the most 'inside' (smallest v)."""
            best = None
            best_v = np.inf
            for z in zones:
                v = ellipse_value(p, z)
                if v <= 1.0 and v < best_v:
                    best_v = v
                    best = z
            return best, best_v

        def exit_point_from_zone(p, z, inflate=1.10):
            """
            Compute a point just outside ellipse along radial direction from center through p.
            If p is at center (rare), push along +x.
            """
            x, y = float(p[0]), float(p[1])
            cx, cy = float(z["cx"]), float(z["cy"])
            a, b   = float(z["a"]),  float(z["b"])

            dx = x - cx
            dy = y - cy
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                dx = 1.0
                dy = 0.0

            # Solve for scale t such that ((t*dx)/a)^2 + ((t*dy)/b)^2 = 1
            denom = (dx*dx)/(a*a) + (dy*dy)/(b*b)
            t = 1.0 / np.sqrt(max(denom, 1e-12))

            # Inflate slightly so we are strictly outside (v>1)
            xe = cx + inflate * t * dx
            ye = cy + inflate * t * dy
            return np.array([xe, ye], dtype=float)

        # 2) choose a touchdown target at ~80% of range, scan candidate bearings
        p0 = np.array([aircraft.pos_north, aircraft.pos_east], dtype=float)
        escape_start = p0.copy()
        containing_zone, v0 = find_containing_zone(p0, no_land_zones)

        if containing_zone is not None:
            escape_start = exit_point_from_zone(p0, containing_zone, inflate=1.15)
        r  = 0.8 * R_max

        def inside_any_ellipse(p):
            x, y = float(p[0]), float(p[1])
            for z in no_land_zones:
                v = ((x - z["cx"])**2)/(z["a"]**2) + ((y - z["cy"])**2)/(z["b"]**2)
                if v <= 1.0:
                    return True
            return False
        def segment_hits_any_ellipse(p0, p1, zones, n_samples=60):
            if zones is None or len(zones) == 0:
                return False
            ts = np.linspace(0.0, 1.0, n_samples)
            pts = (1 - ts)[:, None] * p0[None, :] + ts[:, None] * p1[None, :]
            for z in zones:
                cx, cy, a, b = float(z["cx"]), float(z["cy"]), float(z["a"]), float(z["b"])
                # ellipse equation <= 1 means inside
                v = ((pts[:, 0] - cx) ** 2) / (a ** 2) + ((pts[:, 1] - cy) ** 2) / (b ** 2)
                if np.any(v <= 1.0):
                    return True
            return False
        def min_ellipse_value(p, zones):
            """
            Returns min over zones of v = ((x-cx)^2/a^2 + (y-cy)^2/b^2).
            v < 1  => inside ellipse
            v = 1  => on boundary
            v > 1  => outside; larger is farther (in normalized units)
            """
            if zones is None or len(zones) == 0:
                return np.inf
            x, y = float(p[0]), float(p[1])
            vmin = np.inf
            for z in zones:
                cx, cy = float(z["cx"]), float(z["cy"])
                a, b   = float(z["a"]),  float(z["b"])
                v = ((x - cx)**2)/(a**2) + ((y - cy)**2)/(b**2)
                vmin = min(vmin, v)
            return vmin



        def angle_wrap(a):
            return (a + np.pi) % (2*np.pi) - np.pi
        

        chi = float(aircraft.chi)
        best_q = None
        best_score = -np.inf

        # tune: how hard to penalize turning away from current heading
        TURN_PENALTY = 0.25  # smaller => prioritize zone clearance more

        for dpsi in np.deg2rad(np.linspace(-180, 180, 73)):   # wider sweep helps a lot
            q = escape_start + r * np.array([np.cos(chi + dpsi), np.sin(chi + dpsi)])

            if inside_any_ellipse(q):
                continue
            if segment_hits_any_ellipse(escape_start, q, no_land_zones):
                continue

            vmin = min_ellipse_value(q, no_land_zones)
            turn_cost = abs(angle_wrap(dpsi))
            score = vmin - TURN_PENALTY * turn_cost

            if score > best_score:
                best_score = score
                best_q = q

        if best_q is None:
            best_q = escape_start + r * np.array([np.cos(chi + np.pi/2), np.sin(chi + np.pi/2)])

        target = best_q



        # 3) build a simple 3D waypoint polyline from current position to target, altitude down to 0
        # n = max(20, self.N + 1)
        # xs = np.linspace(aircraft.pos_north, target[0], n)
        # ys = np.linspace(aircraft.pos_east,  target[1], n)
        # hs = np.linspace(aircraft.altitude,  0.0,       n)
        # return np.vstack([xs, ys, hs]).T
        n = max(20, self.N + 1)

        if np.linalg.norm(escape_start - p0) < 1e-6:
            # not inside any ellipse: single segment
            xs = np.linspace(p0[0], target[0], n)
            ys = np.linspace(p0[1], target[1], n)
        else:
            # inside ellipse: two segments with ~25% points for escape
            n1 = max(5, int(0.25 * n))
            n2 = n - n1

            xs1 = np.linspace(p0[0], escape_start[0], n1, endpoint=False)
            ys1 = np.linspace(p0[1], escape_start[1], n1, endpoint=False)

            xs2 = np.linspace(escape_start[0], target[0], n2)
            ys2 = np.linspace(escape_start[1], target[1], n2)

            xs = np.concatenate([xs1, xs2])
            ys = np.concatenate([ys1, ys2])

        hs = np.linspace(aircraft.altitude, 0.0, n)
        return np.vstack([xs, ys, hs]).T


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