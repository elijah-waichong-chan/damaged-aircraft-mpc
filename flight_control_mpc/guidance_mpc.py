from aircraft_model import AircraftModel
from path_planner import GlidePathPlanner
import numpy as np
import casadi as ca


# --------------------------------------------------------------
# Parameters
# --------------------------------------------------------------

# Cost weights
Q_POS_N   = 10.0   # north tracking
Q_POS_E   = 10.0   # east tracking
Q_ALT     = 100.0   # altitude tracking

Q_V       = 0.0    # speed tracking (0 -> free)
Q_CHI     = 0.0    # heading tracking (0 -> free)
Q_GAMMA   = 0.0    # flight path angle tracking (0 -> free)

R_THRUST  = 0.1
R_CHI_DOT = 0.1
R_GAM_DOT = 0.1

V_MIN, V_MAX = 28.0, 62.0

U_THRUST_MAX = 5.0
U_CHI_MAX    = np.deg2rad(5.0)
U_GAMMA_MAX  = np.deg2rad(3.0)

class GuidanceMPC:
    def __init__(self ,path_planner: GlidePathPlanner, mpc_N, mpc_dt):
        self.N = mpc_N
        self.dt = mpc_dt
        self.path_planner = path_planner

        self._build_mpc_problem()


    # ------------------------------------------------------------------
    # Build Opti problem once
    # ------------------------------------------------------------------
    def _build_mpc_problem(self):
        N  = self.N
        nx = 6
        nu = 3

        opti = ca.Opti()

        # Decision variables
        X = opti.variable(nx, N + 1)   # x_0..x_N
        U = opti.variable(nu, N)       # u_0..u_{N-1}

        # Parameters (set at each solve)
        x0   = opti.parameter(nx)          # current absolute state
        Ad   = opti.parameter(nx, nx)      # linearized A_d
        Bd   = opti.parameter(nx, nu)      # linearized B_d
        drift = opti.parameter(nx)         # affine drift f(x̄)*dt
        Xref_rel = opti.parameter(nx, N + 1)   # reference in delta coords

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

        # Initial condition
        opti.subject_to(X[:, 0] == 0)  # δx_0 = 0

        J = 0

        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]
            Xref_rel_k = Xref_rel[:, k]

            e = xk - Xref_rel_k
            J += ca.mtimes([e.T, Q_CA, e]) + ca.mtimes([uk.T, R_CA, uk])

            # Dynamics: δx_{k+1} = Ad δx_k + Bd δu_k + f̄*dt
            x_next = ca.mtimes(Ad, xk) + ca.mtimes(Bd, uk) + drift
            opti.subject_to(X[:, k+1] == x_next)

        # Terminal cost
        eN = X[:, N] - Xref_rel[:, N]
        J += ca.mtimes([eN.T, Q_CA, eN])

        opti.minimize(J)

        # State constraints (example: speed bounds)
        v_abs = X[3, :] + x0[3]   # v̄ + δv
        opti.subject_to(opti.bounded(V_MIN, v_abs, V_MAX))

        # Input constraints
        opti.subject_to(opti.bounded(-U_THRUST_MAX, U[0, :], U_THRUST_MAX))
        opti.subject_to(opti.bounded(-U_CHI_MAX,    U[1, :], U_CHI_MAX))
        opti.subject_to(opti.bounded(-U_GAMMA_MAX,  U[2, :], U_GAMMA_MAX))

        # Solver options
        opti.solver("ipopt", {"print_time": False}, {"print_level": 0})

        # Store handles
        self.opti   = opti
        self.X      = X
        self.U      = U
        self.x0     = x0
        self.Ad     = Ad
        self.Bd     = Bd
        self.drift  = drift
        self.Xref   = Xref_rel

    def _build_local_reference(self, aircraft: AircraftModel):
        """
        Build a dense reference Xref[:, 0..N] by interpolating along
        the waypoint path according to a desired along-track speed.

        Assumptions:
          - path_planner.solve_for_waypoints(aircraft) returns
            waypoints with row 0 equal to current aircraft position.
        """
        # 1) Get waypoints from planner
        waypoints = self.path_planner.solve_for_waypoints(aircraft, verbose=False)
        north_wp = waypoints[:, 0]
        east_wp  = waypoints[:, 1]
        alt_wp   = waypoints[:, 2]

        M = waypoints.shape[0]
        assert M >= 2, "Need at least 2 waypoints"

        # 2) Compute cumulative arc length along the waypoint polyline
        dNEH = np.diff(waypoints, axis=0)  # shape (M-1, 3)
        seg_len = np.linalg.norm(dNEH, axis=1)  # length of each segment
        s_wp = np.zeros(M)
        s_wp[1:] = np.cumsum(seg_len)  # s=0 at first waypoint

        # 3) Design along-path speed (m/s)
        #    You can make this a parameter; for now use current speed.
        v_des = aircraft.vel

        # 4) Build desired along-path distances for each MPC step
        T = self.N + 1
        s_ref = np.zeros(T)
        for k in range(T):
            s_k = v_des * self.dt * k  # distance along path from current
            # Clamp to end of path
            s_ref[k] = np.clip(s_k, 0.0, s_wp[-1])

        # 5) Interpolate north/east/alt along s_wp
        north_ref = np.interp(s_ref, s_wp, north_wp)
        east_ref  = np.interp(s_ref, s_wp, east_wp)
        alt_ref   = np.interp(s_ref, s_wp, alt_wp)

        # 6) v, chi, gamma references (we may not track them strongly)
        #    For now just hold current values.
        v_ref     = np.full(T, aircraft.vel)
        chi_ref   = np.full(T, aircraft.chi)
        gamma_ref = np.full(T, aircraft.gamma)

        Xref_abs = np.vstack([
            north_ref,
            east_ref,
            alt_ref,
            v_ref,
            chi_ref,
            gamma_ref,
        ])  # shape (6, T)

        return Xref_abs, waypoints

    # ------------------------------------------------------------------
    # Solve MPC for current control input
    # ------------------------------------------------------------------
    def solve_for_control_input(self, aircraft: AircraftModel):
        """
        Update A_d, B_d from current aircraft state,
        build local reference from the path planner,
        and solve MPC for the first control input u_0.
        """
        # 1) Update linearization
        Ad = aircraft.Ad.astype(float)
        Bd = aircraft.Bd.astype(float)

        # 2) Current state and drift (f̄*dt) from the nonlinear model
        x_bar = aircraft.get_state_vector().astype(float)
        V = aircraft.vel
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

        # 4) Set parameter values (THIS is the important bit)
        opti.set_value(self.x0,   x_bar)    # shape (6,)
        opti.set_value(self.Ad,   Ad)    # shape (6,6)
        opti.set_value(self.Bd,   Bd)    # shape (6,3)
        opti.set_value(self.drift, drift)

        Xref_abs, waypoints = self._build_local_reference(aircraft)
        Xref_rel = Xref_abs - x_bar.reshape(-1, 1)      # (6, N+1)
        opti.set_value(self.Xref, Xref_rel)


        # 5) (optional) warm start: keep previous solution as initial guess
        # opti.set_initial(self.U, opti.value(self.U))
        # opti.set_initial(self.X, opti.value(self.X))

        # 6) Solve MPC
        try:
            sol = opti.solve()
        except RuntimeError as e:
            print("[GuidanceMPC] ❌ MPC solve failed:", str(e))
            raise

        # Optimized trajectories
        U_opt = np.array(sol.value(self.U))  # shape (3, N)

        delta_X_opt = np.array(sol.value(self.X))  # (6, N+1)
        X_abs_opt = delta_X_opt + x_bar.reshape(-1, 1)


        u0 = U_opt[:, 0]
        x0 = X_abs_opt[:, 0]

        print(
            "[GuidanceMPC] ✅ MPC solved. u0 = "
            f"[thrust={u0[0]:.3f}, chi_dot={u0[1]:.3f}, gamma_dot={u0[2]:.3f}]"
            f"[x={x0[0]:.3f}, y={x0[1]:.3f}, h={x0[2]:.3f}]"
        )

        return u0, X_abs_opt, U_opt, waypoints
