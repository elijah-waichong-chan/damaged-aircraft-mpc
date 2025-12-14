import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec

def _runway_axis(runway_heading_deg: float):
    """
    Runway assumed at origin. Return:
      - heading_rad: runway heading in radians
      - dx, dy: unit vector along runway heading in (x, y) plane
    """
    heading_rad = np.deg2rad(runway_heading_deg)
    dx = np.cos(heading_rad)
    dy = np.sin(heading_rad)
    return heading_rad, dx, dy


def plot_3d_path(x, y, h, title="3D Glide Path"):
    """
    Plot 3D path (x, y, h) with runway threshold at the origin,
    and equal scaling on all axes.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, h, "-")
    ax.scatter(
        0.0,
        0.0,
        0.0,
        marker="x",
        s=80,
        label="Runway threshold",
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("h [m]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    # Equal scaling on all axes
    x_range = np.ptp(x)
    y_range = np.ptp(y)
    h_range = np.ptp(h)
    max_range = max(x_range, y_range, h_range)
    if max_range == 0:
        max_range = 1.0

    x_mid = 0.5 * (x.max() + x.min())
    y_mid = 0.5 * (y.max() + y.min())
    h_mid = 0.5 * (h.max() + h.min())

    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(h_mid - max_range / 2, h_mid + max_range / 2)

    ax.set_box_aspect((1, 1, 1))

    return fig, ax



def plot_ground_track(
    x,
    y,
    runway_heading_deg,
    x_mpc=None,
    y_mpc=None,
    heading_deg=None,
    runway_length=2000.0,
    runway_width=200.0,
    title="Ground Track (Top-down View)",
):
    if x_mpc is None: x_mpc = []
    if y_mpc is None: y_mpc = []

    x = np.asarray(x)
    y = np.asarray(y)
    x_mpc = np.asarray(x_mpc)
    y_mpc = np.asarray(y_mpc)

    # --- 1) Create or reuse figure/axes ---
    if not hasattr(plot_ground_track, "_fig"):
        fig, ax = plt.subplots(num="ground_track")
        plot_ground_track._fig = fig
        plot_ground_track._ax  = ax

        # Lines
        (line_plan,) = ax.plot([], [], "--", color="blue", label="Long Horizon Planned Flight Path", linewidth=1)
        (line_mpc,)  = ax.plot([], [], "-",  color="orange",   label="Short Horizon MPC Flight Path", linewidth=4)

        plot_ground_track._line_plan = line_plan
        plot_ground_track._line_mpc  = line_mpc

        # Runway (static)
        heading_rad, dx, dy = _runway_axis(runway_heading_deg)
        L = runway_length
        W = runway_width
        corners_local = np.array(
            [
                [0.0, -W / 2],
                [L,   -W / 2],
                [L,    W / 2],
                [0.0,  W / 2],
            ]
        )
        u = corners_local[:, 0]
        v = corners_local[:, 1]
        x_rw = u * dx - v * dy   # North
        y_rw = u * dy + v * dx   # East

        ax.fill(
            y_rw,
            x_rw,
            facecolor="orange",
            edgecolor="black",
            alpha=0.8,
            label="Runway",
        )

        ax.set_xlabel("y (East) [m]")
        ax.set_ylabel("x (North) [m]")
        ax.legend()
        ax.set_aspect("equal", adjustable="box")

        # ---- Axis limits once (include runway + initial paths if present) ----
        xs = [x_rw]
        ys = [y_rw]
        if x.size > 0 and y.size > 0:
            xs.append(x)
            ys.append(y)
        if x_mpc.size > 0 and y_mpc.size > 0:
            xs.append(x_mpc)
            ys.append(y_mpc)

        xs_all = np.concatenate(xs)
        ys_all = np.concatenate(ys)
        x_min, x_max = xs_all.min(), xs_all.max()
        y_min, y_max = ys_all.min(), ys_all.max()
        dx_range = x_max - x_min
        dy_range = y_max - y_min

        span = max(dx_range, dy_range, 1.0)
        pad_factor = 0.2
        pad = pad_factor * span
        min_pad = 0.2 * runway_length
        pad = max(pad, min_pad)

        ax.set_xlim(y_min - pad, y_max + pad)  # x-axis = East
        ax.set_ylim(x_min - pad, x_max + pad)  # y-axis = North

        # ---- Plane shape (created once, updated every call) ----
        # Define a tiny plane "in body frame":
        # nose at (0, 0), wings/tail behind.
        plane_body = np.array([
            [0.0,   0.0],   # nose (this will sit exactly at the MPC point)
            [-1.0, -0.4],   # left wing
            [-0.7,  0.0],   # fuselage center
            [-1.0,  0.4],   # right wing
        ])
        plot_ground_track._plane_body = plane_body

        # Placeholder polygon; coordinates will be updated every call
        plane_patch = Polygon([[0, 0], [0, 0], [0, 0]],
                              closed=True,
                              facecolor="red",
                              edgecolor="black")
        ax.add_patch(plane_patch)
        plot_ground_track._plane_patch = plane_patch

    else:
        fig = plot_ground_track._fig
        ax  = plot_ground_track._ax
        line_plan = plot_ground_track._line_plan
        line_mpc  = plot_ground_track._line_mpc

    # --- 2) Update line data ---
    line_plan.set_data(y,     x)      # (East, North)
    line_mpc.set_data(y_mpc, x_mpc)   # (East, North)

    ax.set_title(title)

    # --- 3) Update plane polygon position + heading ---
    plane_patch = plot_ground_track._plane_patch
    plane_body  = plot_ground_track._plane_body

    # Choose path to place plane on: MPC if available, else planned
    x_path = x
    y_path = y

    if x_path.size >= 1 and y_path.size >= 1:
        # Nose position = first point of the path
        x0 = x_path[0]  # North
        y0 = y_path[0]  # East

        # Heading in deg: 0 = North, 90 = East
        h_deg = float(np.asarray(heading_deg).ravel()[0])

        # Convert heading (0=N, 90=E) to angle from East axis (matplotlib x-axis)
        theta = np.deg2rad(90.0 - h_deg)

        # Scale plane size based on current axes span
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        span = max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]))
        scale = 0.05 * span  # tweak 0.03 if you want larger/smaller plane

        # Rotate + scale body-frame plane
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
        ])
        plane_pts = (plane_body @ R.T) * scale

        # Translate so nose (0,0) is at (y0, x0) in (East, North)
        plane_pts[:, 0] += y0   # East -> x-axis
        plane_pts[:, 1] += x0   # North -> y-axis

        plane_patch.set_xy(plane_pts)

    # --- 4) Redraw for animation ---
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    return fig, ax

def plot_altitude(
    x,
    y,
    h,
    runway_heading_deg,
    x_sim=None,
    y_sim=None,
    h_sim=None,
    climb_angle_deg=None,   # kept for future use
    glide_angle_deg=3.0,
    title="Altitude vs Distance From Runway",
):
    """
    Plot altitude h versus absolute along-runway distance |s|, where:
      - s is the coordinate along the runway axis through the origin
      - |s| = 0 at the runway threshold
      - |s| increases with distance away from the runway (either side)
    """
    if x_sim is None: x_sim = []
    if y_sim is None: y_sim = []
    if h_sim is None: h_sim = []

    # --- 0) Convert to arrays ---
    x      = np.asarray(x)
    y      = np.asarray(y)
    h      = np.asarray(h)
    x_sim  = np.asarray(x_sim)
    y_sim  = np.asarray(y_sim)
    h_sim  = np.asarray(h_sim)

    # --- 1) Project onto runway axis ---
    heading_rad, dx, dy = _runway_axis(runway_heading_deg)

    # signed along-runway coordinate
    s_raw = x * dx + y * dy
    s_abs = np.abs(s_raw)

    if x_sim.size > 0 and y_sim.size > 0:
        s_raw_sim = x_sim * dx + y_sim * dy
        s_abs_sim = np.abs(s_raw_sim)
    else:
        s_abs_sim = np.array([])

    # --- 2) Create or reuse figure/axes ---
    if not hasattr(plot_altitude, "_fig"):
        fig, ax = plt.subplots(num="altitude_profile")
        plot_altitude._fig = fig
        plot_altitude._ax  = ax

        # Line handles for later updates
        (line_gs,)   = ax.plot([], [], "--", label=f"{glide_angle_deg:.1f}° glideslope")
        (line_plan,) = ax.plot([], [], "--", color="blue", label="Long Horizon Planner Flight Path", linewidth=2)
        (line_sim,)  = ax.plot([], [], "-",  color="orange",   label="Short Horizon MPC Predicted Path", linewidth=4)

        plot_altitude._line_plan = line_plan
        plot_altitude._line_sim  = line_sim
        plot_altitude._line_gs   = line_gs

        # Marker for the first point (sim if available, else planner)
        (first_pt_marker,) = ax.plot(
            [], [], "X", color="red", markersize=8, label="Current Position"
        )
        plot_altitude._first_pt_marker = first_pt_marker

        ax.set_xlabel("Absolute distance from runway |s| [m]")
        ax.set_ylabel("Altitude h [m]")
        ax.set_title(title)
        ax.grid(True)
        ax.legend()

        # --- 3) Set glideslope line & axis limits ONCE ---
        gamma = np.deg2rad(glide_angle_deg)
        tan_gamma = np.tan(gamma)

        # Max distance from initial data
        s_candidates = []
        if s_abs.size > 0:
            s_candidates.append(s_abs)
        if s_abs_sim.size > 0:
            s_candidates.append(s_abs_sim)

        if s_candidates:
            s_all = np.concatenate(s_candidates)
            s_max_data = max(s_all.max(), 0.0)
        else:
            s_max_data = 1.0

        # more forgiving padding
        pad_s = 0.3 * max(s_max_data, 1.0)

        # extend glideslope all the way to visible max (data + padding)
        s_max_gs = s_max_data + pad_s
        s_gs = np.linspace(0.0, s_max_gs, 200)
        h_gs = s_gs * tan_gamma

        line_gs.set_data(s_gs, h_gs)

        # y-limit: include planned/sim/glideslope
        h_candidates = [h_gs]
        if h.size > 0:
            h_candidates.append(h)
        if h_sim.size > 0:
            h_candidates.append(h_sim)
        h_all = np.concatenate(h_candidates)
        h_max = max(h_all.max(), 0.0)

        pad_h = 0.3 * max(h_max, 1.0)

        # x-axis from a bit before runway (negative) out to end of glideslope
        ax.set_xlim(0.0 - pad_s, s_max_gs)
        ax.set_ylim(0.0 - pad_h, h_max + pad_h)

        # Runway at the right side of the plot
        ax.invert_xaxis()

    else:
        fig = plot_altitude._fig
        ax  = plot_altitude._ax
        line_plan       = plot_altitude._line_plan
        line_sim        = plot_altitude._line_sim
        line_gs         = plot_altitude._line_gs
        first_pt_marker = plot_altitude._first_pt_marker
        ax.set_title(title)

    # --- 4) Update line data each frame ---
    line_plan.set_data(s_abs,    h)
    line_sim.set_data(s_abs_sim, h_sim)

    # --- 5) Update first-point marker ---
    # Prefer sim/MPC path if available
    if s_abs_sim.size > 0 and h_sim.size > 0:
        s_path = s_abs_sim
        h_path = h_sim
    else:
        s_path = s_abs
        h_path = h

    if s_path.size > 0 and h_path.size > 0:
        s0 = s_path[0]
        h0 = h_path[0]
        first_pt_marker = plot_altitude._first_pt_marker
        first_pt_marker.set_data([s0], [h0])
    else:
        first_pt_marker = plot_altitude._first_pt_marker
        first_pt_marker.set_data([], [])

    # --- 6) Redraw for animation ---
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    return fig, ax

def wrap_angle_deg(a):
    """Wrap degrees to [-180, 180) for nicer plots."""
    a = (np.asarray(a) + 180.0) % 360.0 - 180.0
    return a

def compute_runway_frame(x, y, runway_heading_deg):
    """
    Convert world (north=x, east=y) into runway-aligned coordinates:
      s = along-runway axis (positive in runway heading direction)
      d = cross-track (positive left of runway axis, right-hand rule)
    """
    psi = np.deg2rad(runway_heading_deg)
    c, s = np.cos(psi), np.sin(psi)
    x = np.asarray(x); y = np.asarray(y)
    s_along =  c * x + s * y
    d_cross = -s * x + c * y
    return s_along, d_cross

def plot_report_figures(
    x_sim_m, y_sim_m, h_sim_m,
    vel_sim_ms, heading_sim_deg, climb_angle_sim_deg,
    u_thrust_mpc_m_s2, u_heading_rate_mpc_deg_s, u_climb_rate_mpc_deg_s,
    x_planned, y_planned, h_planned,
    runway_heading_deg,
    out_dir="report_figs",
    glide_angle_deg=3.0,
    runway_length_m=2000.0,
    input_limits=None,
    mpc_dt=1.0,
    replan_flags=None,        # NEW
    replan_times=None,        # NEW (optional)
):
    import os
    os.makedirs(out_dir, exist_ok=True)

    # ==========================================================
    # Global font / style for report-quality figures (bigger)
    # ==========================================================
    FS_BASE  = 16   # default text
    FS_LABEL = 18   # axis labels
    FS_TITLE = 20   # titles
    FS_LEG   = 14   # legend
    FS_TICK  = 14   # tick labels

    plt.rcParams.update({
        "font.size": FS_BASE,
        "axes.titlesize": FS_TICK,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEG,
        "legend.title_fontsize": FS_LEG,
        "figure.titlesize": FS_TITLE,
        "lines.linewidth": 2.5,
    })

    # Convert to numpy
    x_sim = np.asarray(x_sim_m, dtype=float)
    y_sim = np.asarray(y_sim_m, dtype=float)
    h_sim = np.asarray(h_sim_m, dtype=float)
    V_sim = np.asarray(vel_sim_ms, dtype=float)
    chi_sim = wrap_angle_deg(heading_sim_deg)
    gam_sim = np.asarray(climb_angle_sim_deg, dtype=float)

    uT = np.asarray(u_thrust_mpc_m_s2, dtype=float)
    uChi = np.asarray(u_heading_rate_mpc_deg_s, dtype=float)
    uGam = np.asarray(u_climb_rate_mpc_deg_s, dtype=float)

    t = np.arange(len(uT), dtype=float) * float(mpc_dt)  # inputs exist for each MPC step

    # --------------------------
    # NEW: replanning timestamps
    # --------------------------
    t_replan = np.array([], dtype=float)
    if replan_flags is not None:
        rf = np.asarray(replan_flags, dtype=bool)
        if replan_times is not None:
            tt = np.asarray(replan_times, dtype=float)
            t_replan = tt[rf]
        else:
            # assume one flag per MPC step aligned with inputs
            idx = np.where(rf)[0]
            t_replan = idx.astype(float) * float(mpc_dt)

    # Use the last planned path for a clean reference overlay
    x_ref = np.asarray(x_planned[-1], dtype=float)
    y_ref = np.asarray(y_planned[-1], dtype=float)
    h_ref = np.asarray(h_planned[-1], dtype=float)

    # Runway axis line for plotting
    psi = np.deg2rad(runway_heading_deg)
    c, s = np.cos(psi), np.sin(psi)
    L = float(runway_length_m)
    runway_x = np.array([0.0, L * c])
    runway_y = np.array([0.0, L * s])

    # ---- Figure 1: Ground track (top-down) ----
    plt.figure(figsize=(9, 4))
    plt.plot(x_sim, y_sim, label="Simulated trajectory")
    plt.plot(x_ref, y_ref, "--", label="Planned reference (final solve)")
    plt.plot(runway_x, runway_y, label="Runway axis")
    plt.scatter([0.0], [0.0], marker="x", s=80, label="Runway threshold")
    plt.axis("equal")
    plt.xlabel("North x (m)")
    plt.ylabel("East y (m)")
    plt.title("Ground Track (Top-Down)")
    plt.grid(True)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_ground_track.png"), dpi=300)

    # ---- Figure 2: Altitude vs along-runway distance ----
    s_sim, d_sim = compute_runway_frame(x_sim, y_sim, runway_heading_deg)
    s_ref, d_ref = compute_runway_frame(x_ref, y_ref, runway_heading_deg)

    # Use absolute along-runway distance like you used in your plot function (optional)
    s_abs = np.abs(np.asarray(s_sim, dtype=float))

    # Glideslope reference line: h = tan(gamma_gs) * |s|
    gs = np.tan(np.deg2rad(glide_angle_deg))
    h_gs = gs * s_abs

    plt.figure(figsize=(9, 4))
    plt.plot(s_abs, h_sim, label="Simulated altitude")
    plt.plot(s_abs, h_gs, "--", label=f"{glide_angle_deg:.1f} deg glideslope")
    plt.xlabel("Along-runway distance (m)")
    plt.ylabel("Altitude (m)")
    plt.title("Altitude vs Distance from Runway")
    plt.grid(True)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_altitude_vs_distance.png"), dpi=300)

    # ---- Figure 3: Cross-track error vs time ----
    t_xtk = np.arange(len(s_sim), dtype=float) * float(mpc_dt)

    plt.figure(figsize=(9, 4))
    plt.plot(t_xtk, d_sim, label="Cross-track Distance (m)")
    plt.xlabel("Time (s)")
    plt.ylabel("Cross-track distance (m)")
    plt.title("Cross-Track Error vs Time")
    plt.grid(True)

    # NEW: star markers at replanning instants (on the curve)
    if t_replan.size > 0:
        # For each replan time, sample d(t) by interpolation so stars sit on the line
        d_replan = np.interp(t_replan, t_xtk, d_sim)

        plt.plot(
            t_replan, d_replan,
            linestyle="None",
            marker="*",
            markersize=14,
            label=f"Replan events (N={len(t_replan)})"
        )

    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "03_cross_track_vs_time.png"), dpi=300)

    # ---- Figure 4: States vs time ----
    t_state = np.arange(len(x_sim), dtype=float) * float(mpc_dt)
    plt.figure(figsize=(9, 4))
    plt.plot(t_state, V_sim, label="Airspeed V (m/s)")
    plt.plot(t_state, chi_sim, label="Heading χ (deg)")
    plt.plot(t_state, gam_sim, label="Flight-path γ (deg)")
    plt.xlabel("Time (s)")
    plt.title("States vs Time")
    plt.grid(True)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "04_states_vs_time.png"), dpi=300)

    # ---- Figure 5: Inputs vs time ----
    plt.figure(figsize=(9, 4))
    plt.plot(t, uT, label="Acceleration Command (m/s²)")
    plt.plot(t, uChi, label="Heading rate command (deg/s)")
    plt.plot(t, uGam, label="Climb rate command (deg/s)")
    plt.xlabel("Time (s)")
    plt.title("MPC Guidance Inputs vs Time")
    plt.grid(True)

    # Optional constraint bands if you provide them
    if input_limits is not None:
        if "uT" in input_limits:
            lo, hi = input_limits["uT"]
            plt.axhline(lo, linestyle="--")
            plt.axhline(hi, linestyle="--")
        if "chi_dot" in input_limits:
            lo, hi = input_limits["chi_dot"]
            plt.axhline(lo, linestyle="--")
            plt.axhline(hi, linestyle="--")
        if "gamma_dot" in input_limits:
            lo, hi = input_limits["gamma_dot"]
            plt.axhline(lo, linestyle="--")
            plt.axhline(hi, linestyle="--")

    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "05_inputs_vs_time.png"), dpi=300)

    print(f"[Saved] Report figures to: {out_dir}/")