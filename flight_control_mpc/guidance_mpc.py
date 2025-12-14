from aircraft_model import AircraftModel
from path_planner import GlidePathPlanner
import numpy as np
import casadi as ca

# --------------------------------------------------------------
# COST WEIGHTS
# --------------------------------------------------------------
Q_POS_N   = 10.0    # north tracking
Q_POS_E   = 10.0    # east tracking
Q_ALT     = 50.0    # altitude tracking
Q_V       = 10.0    # airspeed tracking
Q_CHI     = 1       # heading tracking
Q_GAMMA   = 1       # flight path angle tracking

R_THRUST  = 0.1     # thrust/braking effort
R_CHI_DOT = 0.1     # heading rate effort
R_GAM_DOT = 0.1     # flight path angle rate effort

W_DU = ca.DM(np.diag([1.0, 1.0, 1.0]))

# --------------------------------------------------------------
# STATE SMOOTHNESS (Δx) PENALTIES
# --------------------------------------------------------------
W_DCHI   = 10.0   # penalize (chi_{k+1}-chi_k)^2
W_DGAMMA = 10.0   # penalize (gamma_{k+1}-gamma_k)^2

# --------------------------------------------------------------
# INPUT CONSTRAINTS
# --------------------------------------------------------------
u_accel_max  = 0                    # acceleration command upper-bound (m/s^2)
u_accel_min  = 0                    # acceleration command lower-bound (m/s^2)
u_chi_max    = np.deg2rad(5.0)      # heading rate command upper-bound (rad/s)
u_chi_min    = np.deg2rad(-5.0)     # heading rate command lower-bound (rad/s)
u_gamma_max  = np.deg2rad(3.0)      # climb-angle rate command upper-bound (rad/s)
u_gamma_min  = np.deg2rad(-3.0)     # climb-angle rate command lower-bound (rad/s)

# --------------------------------------------------------------
# INPUT RATE (Δu) CONSTRAINTS
# --------------------------------------------------------------
du_accel_max = 0.25                # m/s^2 per step
du_chi_max   = np.deg2rad(2.0)     # rad/s per step
du_gamma_max = np.deg2rad(1.0)     # rad/s per step

# --------------------------------------------------------------
# STATE CONSTRAINTS
# --------------------------------------------------------------
gamma_max = np.deg2rad(30.0)    # max climb angle (rad)
gamma_min = np.deg2rad(-30.0)   # min climb angle (rad)


class GuidanceMPC:
    def __init__(self ,path_planner: GlidePathPlanner, aircraft: AircraftModel , mpc_N, mpc_dt):
        self.N = mpc_N
        self.dt = mpc_dt
        self.path_planner = path_planner
        self.glide_speed_ms = aircraft.glide_speed_ms
        self.approach_speed_ms = aircraft.approach_speed_ms
        self._build_mpc_problem(aircraft)
        self.u_prev_value = np.zeros(3, dtype=float)

        # --- Replan gate memory ---
        self._s0_prev = None
        self._eperp_count = 0
        self._stall_count = 0
        self._sat_count = 0

        # --- Thresholds ---
        self.EPERP_MAX = 100.0      # meters
        self.S_PROGRESS_MIN = 1.0   # meters per step
        self.M_STEPS = 3            # consecutive steps
        self.SAT_FRAC = 0.9         # fraction of input bound


    def solve_for_control_input(self, aircraft: AircraftModel, waypoints, Xref_abs):
        """ 
        This method updates the linearization and solves the MPC problem.
        """
        # 1) Update linearization
        Ad = aircraft.Ad.astype(float)
        Bd = aircraft.Bd.astype(float)

        # 2) Current state and drift from the nonlinear model
        x_bar = aircraft.get_state_vector().astype(float)
        V = aircraft.vel_ms
        chi = aircraft.chi
        gamma = aircraft.gamma
        dt = aircraft.dt

        drift = np.array([
            V * np.cos(gamma) * np.cos(chi) * dt,
            V * np.cos(gamma) * np.sin(chi) * dt,
            V * np.sin(gamma) * dt,
            0.0,
            0.0,
            0.0,
        ])

        opti = self.opti

        # 4) Set parameter values
        opti.set_value(self.x0,    x_bar)
        opti.set_value(self.Ad,    Ad)
        opti.set_value(self.Bd,    Bd)
        opti.set_value(self.drift, drift)

        Xref_rel = Xref_abs - x_bar.reshape(-1, 1)
        opti.set_value(self.Xref, Xref_rel)
        opti.set_value(self.u_prev, self.u_prev_value)

        # 5) Solve MPC
        try:
            sol = opti.solve()
        except RuntimeError as e:
            print("[GuidanceMPC] ❌ MPC solve failed:", str(e))
            raise

        # Optimized trajectories
        U_opt = np.array(sol.value(self.U))
        delta_X_opt = np.array(sol.value(self.X))
        X_abs_opt = delta_X_opt + x_bar.reshape(-1, 1)

        u0 = U_opt[:, 0]
        x0 = X_abs_opt[:, 0]
        
        # Decide whether to replan
        replan = self._should_replan(aircraft, waypoints, u0)
        self.u_prev_value = u0.copy()

        print(
            f"[GuidanceMPC] ✅ MPC solved. "
            f"Current Position: x={x0[0]:.3f}, y={x0[1]:.3f}, h={x0[2]:.3f} | "
            f"replan={replan}"
        )

        return u0, X_abs_opt, U_opt, waypoints, replan


    # --------------------------------------------------------------
    # Build Opti problem once
    # --------------------------------------------------------------
    def _build_mpc_problem(self, aircraft: AircraftModel):
        N  = self.N
        nx = 6
        nu = 3

        opti = ca.Opti("conic")
        # opti = ca.Opti()

        # Extract Aircraft Model parameters
        V_MIN = aircraft.stall_speed_ms
        V_MAX = aircraft.never_exceed_speed_ms
        self.glide_speed_ms = aircraft.glide_speed_ms

        # Decision variables
        X = opti.variable(nx, N + 1)   # δx_0..δx_N
        U = opti.variable(nu, N)       # u_0..u_{N-1}

        # Parameters (set at each solve)
        x0    = opti.parameter(nx)               # current absolute state
        Ad    = opti.parameter(nx, nx)           # linearized A_d
        Bd    = opti.parameter(nx, nu)           # linearized B_d
        drift = opti.parameter(nx)               # affine drift
        Xref_rel = opti.parameter(nx, N + 1)     # reference in delta coords
        u_prev = opti.parameter(nu)

        # State cost matrix
        Q = np.diag([
            Q_POS_N,   # pos_north
            Q_POS_E,   # pos_east
            Q_ALT,     # altitude
            Q_V,       # v
            Q_CHI,     # chi
            Q_GAMMA,   # gamma
        ])
        R = np.diag([
            R_THRUST,
            R_CHI_DOT,
            R_GAM_DOT,
        ])

        Q_CA = ca.DM(Q)
        R_CA = ca.DM(R)

        # --------------------------------------------------------------
        # Constraints
        # --------------------------------------------------------------
        # Initial condition (delta coords)
        opti.subject_to(X[:, 0] == 0)  # δx_0 = 0

        # Dynamics
        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]
            x_next = ca.mtimes(Ad, xk) + ca.mtimes(Bd, uk) + drift
            opti.subject_to(X[:, k + 1] == x_next)

        # State constraints (absolute airspeed)
        v_abs = X[3, :] + x0[3]
        opti.subject_to(opti.bounded(V_MIN, v_abs, V_MAX))
        # State constraints (absolute flight path angle gamma)
        gamma_abs = X[5, :] + x0[5]
        opti.subject_to(opti.bounded(gamma_min, gamma_abs, gamma_max))

        # Input constraints (absolute)
        opti.subject_to(opti.bounded(u_accel_min,  U[0, :], u_accel_max))
        opti.subject_to(opti.bounded(u_chi_min,    U[1, :], u_chi_max))
        opti.subject_to(opti.bounded(u_gamma_min,  U[2, :], u_gamma_max))

        # Δu (input-rate) constraints
        # First step relative to last applied input
        opti.subject_to(opti.bounded(-du_accel_max, U[0, 0] - u_prev[0], du_accel_max))
        opti.subject_to(opti.bounded(-du_chi_max,   U[1, 0] - u_prev[1], du_chi_max))
        opti.subject_to(opti.bounded(-du_gamma_max, U[2, 0] - u_prev[2], du_gamma_max))
        # Remaining steps relative to previous step
        for k in range(1, N):
            opti.subject_to(opti.bounded(-du_accel_max, U[0, k] - U[0, k-1], du_accel_max))
            opti.subject_to(opti.bounded(-du_chi_max,   U[1, k] - U[1, k-1], du_chi_max))
            opti.subject_to(opti.bounded(-du_gamma_max, U[2, k] - U[2, k-1], du_gamma_max))

        # --------------------------------------------------------------
        # Cost Function
        # --------------------------------------------------------------
        J = 0
        for k in range(N):
            e  = X[:, k] - Xref_rel[:, k]
            uk = U[:, k]

            J += ca.mtimes([e.T, Q_CA, e]) + ca.mtimes([uk.T, R_CA, uk])

            # Δu
            du = uk - u_prev if k == 0 else uk - U[:, k-1]
            J += ca.mtimes([du.T, W_DU, du])

            # Δchi, Δgamma
            dchi   = X[4, k+1] - X[4, k]
            dgamma = X[5, k+1] - X[5, k]
            J += W_DCHI * dchi**2 + W_DGAMMA * dgamma**2

        # terminal
        eN = X[:, N] - Xref_rel[:, N]
        J += ca.mtimes([eN.T, Q_CA, eN])

        opti.minimize(J)

        # --------------------------------------------------------------
        # Solver options
        # --------------------------------------------------------------
        opts = {
            "printLevel": "none",
            "terminationTolerance": 1e-4,
            "boundTolerance": 1e-4,
        }
        opti.solver("qpoases", opts)

        # Store handles
        self.opti   = opti
        self.X      = X
        self.U      = U
        self.x0     = x0
        self.Ad     = Ad
        self.Bd     = Bd
        self.drift  = drift
        self.Xref   = Xref_rel
        self.u_prev = u_prev


    def build_local_reference(self, aircraft: AircraftModel, waypoints=None):
        """
        Build an absolute reference Xref_abs over the MPC horizon by interpolating
        along the (possibly cached) waypoint polyline, starting from the aircraft's
        current projected along-path location s0 (not necessarily waypoint 0).
        """
        # 1) Get waypoints (use cached if provided)
        if waypoints is None:
            waypoints = self.path_planner.solve_for_waypoints(aircraft, verbose=False)

        waypoints = np.asarray(waypoints)
        north_wp = waypoints[:, 0]
        east_wp  = waypoints[:, 1]
        alt_wp   = waypoints[:, 2]
        M = waypoints.shape[0]

        # 2) Compute s0 by projecting current aircraft position onto the polyline
        p = np.array([aircraft.pos_north, aircraft.pos_east, aircraft.altitude], dtype=float)
        s0, s_wp, _ = self._project_to_polyline_s(p, waypoints, use_altitude=False)

        # 3) Build desired along-path distances for each MPC step (start at s0)
        T = self.N + 1
        s_ref = np.zeros(T)

        D_margin = 50.0  # meters before runway threshold
        s_end = max(s_wp[-1] - D_margin, s0)  # never end behind current projection

        for k in range(T):
            s_k = s0 + self.approach_speed_ms * self.dt * k

            s_ref[k] = np.clip(s_k, s0, s_end)

        # 4) Interpolate references
        north_ref = np.interp(s_ref, s_wp, north_wp)
        east_ref  = np.interp(s_ref, s_wp, east_wp)
        alt_ref   = np.interp(s_ref, s_wp, alt_wp)
        v_ref     = np.full(T, self.approach_speed_ms)

        # 5) Angular references: keep current (your original choice)
        # --- Derivatives wrt along-path distance s (not index) ---
        dN_ds = np.gradient(north_ref, s_ref, edge_order=1)
        dE_ds = np.gradient(east_ref,  s_ref, edge_order=1)
        dh_ds = np.gradient(alt_ref,   s_ref, edge_order=1)

        # Heading from horizontal tangent
        chi_ref = np.unwrap(np.arctan2(dE_ds, dN_ds))
        for k in range(1, len(chi_ref)):
            chi_ref[k] = 0.2*chi_ref[k] + 0.8*chi_ref[k-1]   # tune 0.2

        # Flight-path angle from vertical slope vs horizontal distance
        gamma_ref = np.arctan(dh_ds)   # equivalent to atan2(dh_ds, 1)

        # 6) Assemble Xref_abs (nx x T)
        Xref_abs = np.vstack([
            north_ref,
            east_ref,
            alt_ref,
            v_ref,
            chi_ref,
            gamma_ref,
        ])

        return Xref_abs, waypoints
    
    def _project_to_polyline_s(self, p, waypoints, use_altitude=False):
        wp = np.asarray(waypoints)

        if not use_altitude:
            p2 = p[:2]
            segs0 = wp[:-1, :2]
            segs1 = wp[1:,  :2]
        else:
            p2 = p
            segs0 = wp[:-1, :]
            segs1 = wp[1:,  :]

        d = segs1 - segs0
        seg_len = np.linalg.norm(d, axis=1)
        seg_len_safe = np.maximum(seg_len, 1e-9)

        s_wp = np.zeros(wp.shape[0])
        s_wp[1:] = np.cumsum(seg_len)

        best_dist2 = np.inf
        best_s0 = 0.0

        for i in range(len(seg_len)):
            v = d[i]
            w = p2 - segs0[i]
            t = float(np.dot(w, v) / (seg_len_safe[i]**2))
            t = np.clip(t, 0.0, 1.0)
            proj = segs0[i] + t * v
            dist2 = float(np.sum((p2 - proj)**2))
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_s0 = float(s_wp[i] + t * seg_len[i])

        e_perp = float(np.sqrt(best_dist2))
        return best_s0, s_wp, e_perp
    
    def _should_replan(self, aircraft: AircraftModel, waypoints: np.ndarray, u0: np.ndarray):
        # Project current position onto path
        p = np.array([aircraft.pos_north, aircraft.pos_east, aircraft.altitude], dtype=float)
        s0, _, e_perp = self._project_to_polyline_s(p, waypoints, use_altitude=False)

        # --- Cross-track error ---
        if e_perp > self.EPERP_MAX:
            self._eperp_count += 1
        else:
            self._eperp_count = 0

        # --- Progress stall ---
        if self._s0_prev is None:
            ds = np.inf
            self._stall_count = 0
        else:
            ds = s0 - self._s0_prev
            if ds < self.S_PROGRESS_MIN:
                self._stall_count += 1
            else:
                self._stall_count = 0

        self._s0_prev = s0

        # --- Final decision ---
        return (
            self._eperp_count >= self.M_STEPS or
            self._stall_count >= self.M_STEPS
        )
    
    def reset_replan_memory(self):
        self._s0_prev = None
        self._eperp_count = 0
        self._stall_count = 0
