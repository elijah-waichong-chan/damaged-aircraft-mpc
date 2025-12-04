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

def plot_guidance_overview(
    x, y, h,
    runway_heading_deg,
    x_mpc, y_mpc, h_mpc,
    u_thrust, u_heading, u_climb_rate, 
    vel_ms, heading_deg, climb_angle_deg,
    glide_angle_deg=3.0,
    runway_length=2000.0,
    runway_width=200.0,
    title_ground="Ground Track (Top-down View)",
    title_alt="Altitude vs Distance From Runway",
):
    # --- 0) Convert and prep inputs ---
    vel_kt = vel_ms * 1.94384
    u = np.array([u_thrust, u_heading, u_climb_rate], dtype=float)

    x      = np.asarray(x)
    y      = np.asarray(y)
    h      = np.asarray(h)
    x_mpc  = np.asarray(x_mpc)
    y_mpc  = np.asarray(y_mpc)
    h_mpc  = np.asarray(h_mpc)

    # --- 1) Project onto runway axis for altitude subplot ---
    heading_rad, dx, dy = _runway_axis(runway_heading_deg)

    s_raw = x * dx + y * dy
    s_abs = np.abs(s_raw)

    if x_mpc.size > 0 and y_mpc.size > 0:
        s_raw_mpc = x_mpc * dx + y_mpc * dy
        s_abs_mpc = np.abs(s_raw_mpc)
    else:
        s_abs_mpc = np.array([])

    # --- 2) Create or reuse figure/subplots ---
    if not hasattr(plot_guidance_overview, "_fig"):
        # ============================================================
        # FIGURE + GRID: 4 columns
        #   [state gauges] | [ground] | [altitude] | [MPC inputs]
        # Left column is further split into 3 rows for gauges
        # ============================================================
        fig = plt.figure(figsize=(20, 6), num="guidance_overview")
        gs = fig.add_gridspec(
            1, 4,
            width_ratios=[1.2, 2.2, 2.0, 1.0]
        )

        # Left column: 3 gauges (airspeed, heading, climb/pos)
        gs_state = gs[0].subgridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.5)
        ax_v     = fig.add_subplot(gs_state[0])                   # airspeed
        ax_hdg   = fig.add_subplot(gs_state[1], projection="polar")  # heading
        ax_climb = fig.add_subplot(gs_state[2])                   # climb + x/y/h

        # Main three panels
        ax_g = fig.add_subplot(gs[1])
        ax_h = fig.add_subplot(gs[2])
        ax_u = fig.add_subplot(gs[3])

        # Store axes
        plot_guidance_overview._fig      = fig
        plot_guidance_overview._ax_v     = ax_v
        plot_guidance_overview._ax_hdg   = ax_hdg
        plot_guidance_overview._ax_climb = ax_climb
        plot_guidance_overview._ax_g     = ax_g
        plot_guidance_overview._ax_h     = ax_h
        plot_guidance_overview._ax_u     = ax_u

        # =========================
        # LEFT COLUMN: GAUGES
        # =========================

        # --- Airspeed gauge (horizontal bar) ---
        ax_v.set_title("Airspeed [kt]")
        vmax_kt = max(vel_kt * 1.5, 60.0)   # simple dynamic range
        ax_v.set_xlim(0, vmax_kt)
        ax_v.set_yticks([])
        ax_v.set_xlabel("kt")

        airspeed_bar = ax_v.barh([0], [vel_kt], height=0.6)[0]
        airspeed_text = ax_v.text(
            0.98, 0.5, f"{vel_kt:.1f}",
            transform=ax_v.transAxes,
            ha="right",
            va="center",
            fontsize=10,
        )

        plot_guidance_overview._airspeed_bar  = airspeed_bar
        plot_guidance_overview._airspeed_text = airspeed_text
        plot_guidance_overview._vmax_kt       = vmax_kt

        # --- Heading gauge (simple compass) ---
        ax_hdg.set_title("Heading")
        ax_hdg.set_theta_zero_location("N")  # 0° at North
        ax_hdg.set_theta_direction(-1)       # CW positive
        ax_hdg.set_ylim(0, 1.0)
        ax_hdg.set_yticks([])
        ax_hdg.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax_hdg.set_xticklabels(["N", "E", "S", "W"])

        hdg_angle = np.deg2rad(heading_deg if heading_deg is not None else 0.0)
        heading_arrow, = ax_hdg.plot(
            [hdg_angle, hdg_angle], [0, 1.0], linewidth=2
        )
        heading_text = ax_hdg.text(
            0.5, -0.15,
            f"{heading_deg:.1f}°" if heading_deg is not None else "—",
            transform=ax_hdg.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )

        plot_guidance_overview._heading_arrow = heading_arrow
        plot_guidance_overview._heading_text  = heading_text

        # --- Climb angle gauge + x/y/h text ---
        ax_climb.set_title("Climb Angle [deg]")
        climb_span = 10.0
        ax_climb.set_ylim(-climb_span, climb_span)
        ax_climb.set_xlim(0, 1)
        ax_climb.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax_climb.set_xticks([])
        ax_climb.set_ylabel("deg")

        climb_line, = ax_climb.plot(
            [0.0, 1.0],
            [climb_angle_deg, climb_angle_deg],
            linewidth=3,
        )

        # current position text (x,y,h)
        pos_text = ax_climb.text(
            0.5, -0.25,
            "",
            transform=ax_climb.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )

        plot_guidance_overview._climb_line = climb_line
        plot_guidance_overview._pos_text   = pos_text
        plot_guidance_overview._climb_span = climb_span

        # =========================
        # MIDDLE-LEFT: Ground track
        # =========================
        (line_plan_g,) = ax_g.plot(
            [], [], "--", color="blue",
            label="Long Horizon Planned Flight Path", linewidth=1
        )
        (line_mpc_g,)  = ax_g.plot(
            [], [], "-",  color="orange",
            label="Short Horizon MPC Flight Path", linewidth=4
        )

        plot_guidance_overview._line_plan_g = line_plan_g
        plot_guidance_overview._line_mpc_g  = line_mpc_g

        # Runway
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
        u_loc = corners_local[:, 0]
        v_loc = corners_local[:, 1]
        x_rw = u_loc * dx - v_loc * dy   # North
        y_rw = u_loc * dy + v_loc * dx   # East

        ax_g.fill(
            y_rw,
            x_rw,
            facecolor="orange",
            edgecolor="black",
            alpha=0.8,
            label="Runway",
        )

        ax_g.set_xlabel("y (East) [m]")
        ax_g.set_ylabel("x (North) [m]")
        ax_g.set_aspect("equal", adjustable="box")
        ax_g.legend(loc="best")

        # Ground track axis limits once
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

        span_xy = max(dx_range, dy_range, 1.0)
        pad_factor = 0.2
        pad = pad_factor * span_xy
        min_pad = 0.2 * runway_length
        pad = max(pad, min_pad)

        ax_g.set_xlim(y_min - pad, y_max + pad)  # x-axis = East
        ax_g.set_ylim(x_min - pad, x_max + pad)  # y-axis = North

        # Plane shape
        plane_body = np.array([
            [0.0,   0.0],   # nose
            [-1.0, -0.4],   # left wing
            [-0.7,  0.0],   # fuselage center
            [-1.0,  0.4],   # right wing
        ])
        plot_guidance_overview._plane_body = plane_body

        plane_patch = Polygon(
            [[0, 0], [0, 0], [0, 0]],
            closed=True,
            facecolor="red",
            edgecolor="black"
        )
        ax_g.add_patch(plane_patch)
        plot_guidance_overview._plane_patch = plane_patch

        # =========================
        # MIDDLE-RIGHT: Altitude profile
        # =========================
        (line_gs_h,)   = ax_h.plot([], [], "--", label=f"{glide_angle_deg:.1f}° glideslope")
        (line_plan_h,) = ax_h.plot(
            [], [], "--", color="blue",
            label="Long Horizon Planner Flight Path", linewidth=1
        )
        (line_mpc_h,)  = ax_h.plot(
            [], [], "-",  color="orange",
            label="Short Horizon MPC Predicted Path", linewidth=4
        )

        plot_guidance_overview._line_plan_h = line_plan_h
        plot_guidance_overview._line_mpc_h  = line_mpc_h
        plot_guidance_overview._line_gs_h   = line_gs_h

        (first_pt_marker,) = ax_h.plot(
            [], [], "X", color="red", markersize=8, label="Current Position"
        )
        plot_guidance_overview._first_pt_marker = first_pt_marker

        ax_h.set_xlabel("Absolute distance from runway |s| [m]")
        ax_h.set_ylabel("Altitude h [m]")
        ax_h.grid(True)
        ax_h.legend(loc="best")

        gamma = np.deg2rad(glide_angle_deg)
        tan_gamma = np.tan(gamma)

        s_candidates = []
        if s_abs.size > 0:
            s_candidates.append(s_abs)
        if s_abs_mpc.size > 0:
            s_candidates.append(s_abs_mpc)

        if s_candidates:
            s_all = np.concatenate(s_candidates)
            s_max_data = max(s_all.max(), 0.0)
        else:
            s_max_data = 1.0

        pad_s = 0.3 * max(s_max_data, 1.0)
        s_max_gs = s_max_data + pad_s
        s_gs = np.linspace(0.0, s_max_gs, 200)
        h_gs = s_gs * tan_gamma
        line_gs_h.set_data(s_gs, h_gs)

        h_candidates = [h_gs]
        if h.size > 0:
            h_candidates.append(h)
        if h_mpc.size > 0:
            h_candidates.append(h_mpc)
        h_all = np.concatenate(h_candidates)
        h_max = max(h_all.max(), 0.0)

        pad_h = 0.3 * max(h_max, 1.0)

        ax_h.set_xlim(0.0 - pad_s, s_max_gs)
        ax_h.set_ylim(0.0 - pad_h, h_max + pad_h)
        ax_h.invert_xaxis()

        # =========================
        # RIGHT: control inputs (bar chart)
        # =========================
        ax_u.set_title("MPC Inputs")
        u_labels = ["Speed Brake (m/s²)", "Heading Rate (deg/s)", "Climb Rate (deg/s)"]
        y_pos = np.arange(len(u_labels))

        bars = ax_u.barh(y_pos, u, color="gray")
        plot_guidance_overview._bars_u   = bars
        plot_guidance_overview._u_labels = u_labels

        ax_u.set_yticks(y_pos)
        ax_u.set_yticklabels(u_labels)
        ax_u.set_xlabel("Magnitude")
        ax_u.grid(axis="x", linestyle="--", alpha=0.5)

        u_span = max(np.max(np.abs(u)), 1.0)
        pad_u  = 0.2 * u_span
        xlim_min = -u_span - pad_u
        xlim_max =  u_span + pad_u
        ax_u.set_xlim(xlim_min, xlim_max)
        plot_guidance_overview._u_xlim = (xlim_min, xlim_max)

        fig.tight_layout()

    else:
        fig        = plot_guidance_overview._fig
        ax_v       = plot_guidance_overview._ax_v
        ax_hdg     = plot_guidance_overview._ax_hdg
        ax_climb   = plot_guidance_overview._ax_climb
        ax_g       = plot_guidance_overview._ax_g
        ax_h       = plot_guidance_overview._ax_h
        ax_u       = plot_guidance_overview._ax_u

        line_plan_g = plot_guidance_overview._line_plan_g
        line_mpc_g  = plot_guidance_overview._line_mpc_g
        plane_patch = plot_guidance_overview._plane_patch
        plane_body  = plot_guidance_overview._plane_body

        line_plan_h       = plot_guidance_overview._line_plan_h
        line_mpc_h        = plot_guidance_overview._line_mpc_h
        line_gs_h         = plot_guidance_overview._line_gs_h
        first_pt_marker   = plot_guidance_overview._first_pt_marker

        airspeed_bar  = plot_guidance_overview._airspeed_bar
        airspeed_text = plot_guidance_overview._airspeed_text
        vmax_kt       = plot_guidance_overview._vmax_kt

        heading_arrow = plot_guidance_overview._heading_arrow
        heading_text  = plot_guidance_overview._heading_text

        climb_line    = plot_guidance_overview._climb_line
        pos_text      = plot_guidance_overview._pos_text
        climb_span    = plot_guidance_overview._climb_span


    # Get shared objects for both first and subsequent calls
    bars_u = plot_guidance_overview._bars_u
    xlim_min, xlim_max = plot_guidance_overview._u_xlim

    airspeed_bar  = plot_guidance_overview._airspeed_bar
    airspeed_text = plot_guidance_overview._airspeed_text
    vmax_kt       = plot_guidance_overview._vmax_kt

    heading_arrow = plot_guidance_overview._heading_arrow
    heading_text  = plot_guidance_overview._heading_text

    climb_line    = plot_guidance_overview._climb_line
    pos_text      = plot_guidance_overview._pos_text
    climb_span    = plot_guidance_overview._climb_span

    # =========================
    # UPDATE LEFT: ground track
    # =========================
    plot_guidance_overview._line_plan_g.set_data(y,     x)
    plot_guidance_overview._line_mpc_g.set_data(y_mpc, x_mpc)
    ax_g.set_title(title_ground)

    # Plane icon on ground track
    x_path = x
    y_path = y
    plane_patch = plot_guidance_overview._plane_patch
    plane_body  = plot_guidance_overview._plane_body

    if x_path.size >= 1 and y_path.size >= 1 and heading_deg is not None:
        x0 = x_path[0]  # North
        y0 = y_path[0]  # East

        h_deg = float(np.asarray(heading_deg).ravel()[0])  # 0 = North, 90 = East
        theta = np.deg2rad(90.0 - h_deg)  # convert to matplotlib frame

        xlim = ax_g.get_xlim()
        ylim = ax_g.get_ylim()
        span_xy = max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]))
        scale = 0.05 * span_xy

        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
        ])
        plane_pts = (plane_body @ R.T) * scale

        plane_pts[:, 0] += y0
        plane_pts[:, 1] += x0

        plane_patch.set_xy(plane_pts)

    # =========================
    # UPDATE MIDDLE: altitude
    # =========================
    plot_guidance_overview._line_plan_h.set_data(s_abs,    h)
    plot_guidance_overview._line_mpc_h.set_data(s_abs_mpc, h_mpc)
    ax_h.set_title(title_alt)

    if s_abs_mpc.size > 0 and h_mpc.size > 0:
        s_path = s_abs_mpc
        h_path = h_mpc
    else:
        s_path = s_abs
        h_path = h

    first_pt_marker = plot_guidance_overview._first_pt_marker
    if s_path.size > 0 and h_path.size > 0:
        s0 = s_path[0]
        h0 = h_path[0]
        first_pt_marker.set_data([s0], [h0])
    else:
        first_pt_marker.set_data([], [])

    # =========================
    # UPDATE RIGHT: bar chart
    # =========================
    for i, bar in enumerate(bars_u):
        if i < u.size:
            bar.set_width(u[i])
        else:
            bar.set_width(0.0)

    ax_u.set_xlim(xlim_min, xlim_max)

    # =========================
    # UPDATE GAUGES (LEFT COLUMN)
    # =========================

    # --- Airspeed gauge ---
    new_v = np.clip(vel_kt, 0.0, vmax_kt)
    airspeed_bar.set_width(new_v)
    airspeed_text.set_text(f"{vel_kt:.1f}")

    # --- Heading gauge ---
    if heading_deg is not None:
        hdg_angle = np.deg2rad(float(heading_deg))
        heading_arrow.set_data([hdg_angle, hdg_angle], [0.0, 1.0])
        heading_text.set_text(f"{heading_deg:.1f}°")
    else:
        heading_text.set_text("—")

    # --- Climb angle gauge + position text ---
    climb_val = float(climb_angle_deg if climb_angle_deg is not None else 0.0)
    climb_val = np.clip(climb_val, -climb_span, climb_span)
    climb_line.set_ydata([climb_val, climb_val])

    if x.size > 0 and y.size > 0 and h.size > 0:
        x0 = x[0]
        y0 = y[0]
        h0 = h[0]
        pos_text.set_text(
            f"x = {x0:.0f} m\n"
            f"y = {y0:.0f} m\n"
            f"h = {h0:.0f} m"
        )
    else:
        pos_text.set_text("")

    # =========================
    # Redraw for animation
    # =========================
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    return fig, (ax_g, ax_h, ax_u)