import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse

def animate_results(
    current_step,
    x, y, h,
    runway_heading_deg,
    x_mpc, y_mpc, h_mpc,
    u_thrust, u_heading, u_climb_rate,
    vel_ms, heading_deg, climb_angle_deg,
    no_land_zones=None,
    glide_angle_deg=3.0,
    runway_length=2000.0,
    runway_width=200.0,
    title_ground="Ground Track (Top-down View)",
    title_alt="Altitude vs Distance From Runway",

):
    """
    2x2 overview:
      [0,0] Ground track      [0,1] Altitude vs |s|
      [1,0] Current state     [1,1] MPC inputs

    Adds:
      - Actual flown path (solid line) accumulated over time
      - Keeps ALL old planned paths (grayed out) after replans
      - Replan counter
      - Real-time timer (elapsed wall time since first call)
    """
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    # --- 0) Convert and prep inputs ---
    vel_kt = float(vel_ms) * 1.94384
    u = np.array([u_thrust, u_heading, u_climb_rate], dtype=float)

    x      = np.asarray(x, dtype=float)
    y      = np.asarray(y, dtype=float)
    h      = np.asarray(h, dtype=float)
    x_mpc  = np.asarray(x_mpc, dtype=float)
    y_mpc  = np.asarray(y_mpc, dtype=float)
    h_mpc  = np.asarray(h_mpc, dtype=float)

    # --- helper: runway axis ---
    def _runway_axis(runway_heading_deg):
        heading_rad = np.deg2rad(runway_heading_deg)
        dx = np.cos(heading_rad)
        dy = np.sin(heading_rad)
        return heading_rad, dx, dy

    # --- 1) Project onto runway axis for altitude subplot ---
    heading_rad, dx, dy = _runway_axis(runway_heading_deg)
    
    s_raw = x * dx + y * dy
    s_abs = np.abs(s_raw)

    if x_mpc.size > 0 and y_mpc.size > 0:
        s_raw_mpc = x_mpc * dx + y_mpc * dy
        s_abs_mpc = np.abs(s_raw_mpc)
    else:
        s_abs_mpc = np.array([])

    # --- choose "current" aircraft position for drawing actual path ---
    if x_mpc.size >= 1 and y_mpc.size >= 1 and h_mpc.size >= 1:
        x_now = float(x_mpc[0]); y_now = float(y_mpc[0]); h_now = float(h_mpc[0])
    elif x.size >= 1 and y.size >= 1 and h.size >= 1:
        x_now = float(x[0]); y_now = float(y[0]); h_now = float(h[0])
    else:
        x_now = y_now = h_now = None

    # --- 2) Create or reuse figure/subplots (2x2) ---
    if not hasattr(animate_results, "_fig"):
        fig = plt.figure(figsize=(15, 8), num="guidance_overview")

        gs = fig.add_gridspec(
            2, 2,
            height_ratios=[3.0, 1.0],
            hspace=0.4,
            wspace=0.3,
        )

        ax_g     = fig.add_subplot(gs[0, 0])
        ax_h     = fig.add_subplot(gs[0, 1])
        ax_state = fig.add_subplot(gs[1, 0])
        ax_u     = fig.add_subplot(gs[1, 1])

        animate_results._no_land_patches = []
        # ==========================================================
        # Helper: update / draw no-land zones (ellipses)
        # ==========================================================
        def _update_no_land_zones(zones):
            # Remove old patches
            if hasattr(animate_results, "_no_land_patches"):
                for p in animate_results._no_land_patches:
                    try:
                        p.remove()
                    except Exception:
                        pass
            animate_results._no_land_patches = []

            if zones is None:
                return

            from matplotlib.patches import Ellipse
            for i, z in enumerate(zones):
                # NOTE: ax_g uses x = East (y), y = North (x)
                e = Ellipse(
                    (z["cy"], z["cx"]),
                    width=2.0 * z["b"],     # East radius
                    height=2.0 * z["a"],    # North radius
                    facecolor="red",
                    edgecolor="black",
                    alpha=0.20,
                    linewidth=1.0,
                    zorder=1,
                    label="No-land zone" if i == 0 else "_nolegend_",
                )
                ax_g.add_patch(e)
                animate_results._no_land_patches.append(e)
        animate_results._update_no_land_zones = _update_no_land_zones



        # Store axes
        animate_results._fig      = fig
        animate_results._ax_g     = ax_g
        animate_results._ax_h     = ax_h
        animate_results._ax_u     = ax_u
        animate_results._ax_state = ax_state

        # --- timer / counters ---
        animate_results._t0_wall = time.time()
        animate_results._replan_count = 0
        animate_results._last_plan_sig = None

        # --- actual path history ---
        animate_results._actual_x = []  # North
        animate_results._actual_y = []  # East
        animate_results._actual_h = []  # Alt

        # =====================================================
        # TOP-LEFT: Ground track + runway + plane icon
        # =====================================================
        (line_plan_g,) = ax_g.plot(
            [], [], "--", color="blue",
            label="Long-horizon planned path", linewidth=2,
            zorder=2,
        )
        (line_mpc_g,) = ax_g.plot(
            [], [], "-", color="orange",
            label="Short-horizon MPC path", linewidth=6, alpha=0.6,
            zorder=4,
        )
        # NEW: actual flown path (solid)
        (line_actual_g,) = ax_g.plot(
            [], [], "-", color="black",
            label="Actual path (flown)", linewidth=2.5, alpha=0.9,
            zorder=3,
        )

        animate_results._line_plan_g   = line_plan_g
        animate_results._line_mpc_g    = line_mpc_g
        animate_results._line_actual_g = line_actual_g

        # keep ALL old plans (gray)
        animate_results._old_plan_lines_g = []
        animate_results._old_plan_lines_h = []
        # legend: only label the first archived gray line (avoid duplicates)
        animate_results._past_plan_legend_added_g = False
        animate_results._past_plan_legend_added_h = False

        # Runway polygon in (North x, East y)
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
            y_rw, x_rw,
            facecolor="orange",
            edgecolor="black",
            alpha=0.7,
            label="Runway",
            zorder=0,
        )

        ax_g.set_xlabel("y (East) [m]")
        ax_g.set_ylabel("x (North) [m]")
        ax_g.set_aspect("auto")
        ax_g.grid(True, linestyle="--", alpha=0.3)
        ax_g.legend(loc="best", fontsize=8)

        # Axis limits (based on current data + runway)
        xs = [x_rw]
        ys = [y_rw]
        if x.size > 0 and y.size > 0:
            xs.append(x); ys.append(y)
        if x_mpc.size > 0 and y_mpc.size > 0:
            xs.append(x_mpc); ys.append(y_mpc)

        xs_all = np.concatenate(xs)
        ys_all = np.concatenate(ys)
        x_min, x_max = xs_all.min(), xs_all.max()
        y_min, y_max = ys_all.min(), ys_all.max()
        span_xy = max(x_max - x_min, y_max - y_min, 1.0)
        pad = max(0.2 * span_xy, 0.2 * runway_length)

        ax_g.set_xlim(y_min - pad, y_max + pad)
        ax_g.set_ylim(x_min - pad, x_max + pad)

        # Plane shape
        plane_body = np.array([
            [0.0,   0.0],
            [-1.0, -0.4],
            [-0.7,  0.0],
            [-1.0,  0.4],
        ])
        plane_patch = Polygon(
            [[0, 0], [0, 0], [0, 0]],
            closed=True,
            facecolor="red",
            edgecolor="black",
            zorder=10
        )
        ax_g.add_patch(plane_patch)
        
        animate_results._plane_body  = plane_body
        animate_results._plane_patch = plane_patch

        # =====================================================
        # TOP-RIGHT: Altitude vs distance |s|
        # =====================================================
        (line_gs_h,) = ax_h.plot(
            [], [], "--", color="black",
            label=f"{glide_angle_deg:.1f}° glideslope", linewidth=1
        )
        (line_plan_h,) = ax_h.plot(
            [], [], "--", color="blue",
            label="Long-horizon planned path", linewidth=2
        )
        (line_mpc_h,) = ax_h.plot(
            [], [], "-", color="orange",
            label="Short-horizon MPC prediction", linewidth=6, alpha=0.6
        )
        # NEW: actual flown altitude profile
        (line_actual_h,) = ax_h.plot(
            [], [], "-", color="black",
            label="Actual altitude (flown)", linewidth=2.5, alpha=0.9
        )
        (first_pt_marker,) = ax_h.plot(
            [], [], "X", color="red", markersize=8, label="Current position"
        )

        animate_results._line_gs_h       = line_gs_h
        animate_results._line_plan_h     = line_plan_h
        animate_results._line_mpc_h      = line_mpc_h
        animate_results._line_actual_h   = line_actual_h
        animate_results._first_pt_marker = first_pt_marker

        ax_h.set_xlabel("Absolute distance from runway |s| [m]")
        ax_h.set_ylabel("Altitude h [m]")
        ax_h.grid(True, linestyle="--", alpha=0.3)
        ax_h.legend(loc="best", fontsize=8)

        # glide-slope envelope
        gamma_gs = np.deg2rad(glide_angle_deg)
        tan_gamma = np.tan(gamma_gs)

        s_candidates = []
        if s_abs.size > 0: s_candidates.append(s_abs)
        if s_abs_mpc.size > 0: s_candidates.append(s_abs_mpc)

        s_max_data = max(np.concatenate(s_candidates).max(), 1.0) if s_candidates else 1.0
        pad_s = 0.3 * max(s_max_data, 1.0)
        s_max_gs = s_max_data + pad_s

        s_gs = np.linspace(0.0, s_max_gs, 200)
        h_gs = s_gs * tan_gamma
        line_gs_h.set_data(s_gs, h_gs)

        h_candidates = [h_gs]
        if h.size > 0: h_candidates.append(h)
        if h_mpc.size > 0: h_candidates.append(h_mpc)
        h_all = np.concatenate(h_candidates)
        h_max = max(h_all.max(), 1.0)
        pad_h = 0.3 * max(h_max, 1.0)

        ax_h.set_xlim(0.0 - pad_s, s_max_gs)
        ax_h.set_ylim(0.0 - pad_h, h_max + pad_h)
        ax_h.invert_xaxis()

        # =====================================================
        # BOTTOM-RIGHT: control inputs
        # =====================================================
        ax_u.set_title("MPC Inputs")
        u_labels = ["Accel (m/s²)", "Heading rate (deg/s)", "Climb rate (deg/s)"]
        y_pos = np.arange(len(u_labels))

        bars = ax_u.barh(y_pos, u, color="gray")
        animate_results._bars_u   = bars
        animate_results._u_labels = u_labels

        ax_u.set_yticks(y_pos)
        ax_u.set_yticklabels(u_labels)
        ax_u.set_xlabel("Magnitude")
        ax_u.grid(axis="x", linestyle="--", alpha=0.5)

        u_span = max(np.max(np.abs(u)), 1.0)
        pad_u  = 0.2 * u_span
        xlim_min = -u_span - pad_u
        xlim_max =  u_span + pad_u
        ax_u.set_xlim(xlim_min, xlim_max)
        animate_results._u_xlim = (xlim_min, xlim_max)

        # =====================================================
        # BOTTOM-LEFT: current state summary
        # =====================================================
        ax_state.set_title("Current State", loc="left", fontsize=11, fontweight="bold")
        ax_state.axis("off")

        bbox_style = dict(
            boxstyle="round,pad=0.6",
            fc="#f7f7f7",
            ec="#b0b0b0",
            lw=1.0,
        )

        state_box = ax_state.text(
            0.02, 0.98,
            "",
            transform=ax_state.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            family="monospace",
            bbox=bbox_style,
        )
        animate_results._state_box = state_box

    else:
        fig      = animate_results._fig
        ax_g     = animate_results._ax_g
        ax_h     = animate_results._ax_h
        ax_u     = animate_results._ax_u
        ax_state = animate_results._ax_state
    # Ensure no-land zones are updated every frame
    if hasattr(animate_results, "_update_no_land_zones"):
        animate_results._update_no_land_zones(no_land_zones)


    # =========================
    # RESET on replay
    # =========================
    if int(round(float(current_step))) == 0 and hasattr(animate_results, "_fig"):
        # clear actual flown history
        animate_results._actual_x = []
        animate_results._actual_y = []
        animate_results._actual_h = []
        animate_results._line_actual_g.set_data([], [])
        animate_results._line_actual_h.set_data([], [])

        # remove all archived gray plan lines from axes
        for ln in animate_results._old_plan_lines_g:
            try: ln.remove()
            except Exception: pass
        for ln in animate_results._old_plan_lines_h:
            try: ln.remove()
            except Exception: pass
        animate_results._old_plan_lines_g = []
        animate_results._old_plan_lines_h = []

        # reset replan bookkeeping (recommended for a clean replay)
        animate_results._replan_count = 0
        animate_results._last_plan_sig = None
        animate_results._past_plan_legend_added_g = False
        animate_results._past_plan_legend_added_h = False

        # refresh legends (gray entry removed since lines are gone)
        animate_results._ax_g.legend(loc="best", fontsize=8)
        animate_results._ax_h.legend(loc="best", fontsize=8)

    

    # ----------------------------
    # helper: detect replan + archive old plan as gray
    # ----------------------------
    def _maybe_archive_old_plan(x_plan, y_plan, s_abs_plan, h_plan):
        if x_plan.size < 2 or y_plan.size < 2:
            return False

        # signature: length + endpoints (good enough)
        sig = (
            int(x_plan.size),
            float(x_plan[0]), float(y_plan[0]),
            float(x_plan[-1]), float(y_plan[-1]),
        )

        if animate_results._last_plan_sig is None:
            animate_results._last_plan_sig = sig
            return False

        if sig == animate_results._last_plan_sig:
            return False

        # archive the CURRENT blue plan lines by turning them gray (and dashed)
        old_g = animate_results._line_plan_g
        old_h = animate_results._line_plan_h

        # add legend label only once
        g_label = "Past planned paths" if not animate_results._past_plan_legend_added_g else "_nolegend_"
        h_label = "Past planned paths" if not animate_results._past_plan_legend_added_h else "_nolegend_"

        old_g.set_color("0.6"); old_g.set_alpha(0.60); old_g.set_linewidth(1.8)
        old_g.set_linestyle("--"); old_g.set_label(g_label)

        old_h.set_color("0.6"); old_h.set_alpha(0.60); old_h.set_linewidth(1.8)
        old_h.set_linestyle("--"); old_h.set_label(h_label)

        if g_label != "_nolegend_":
            animate_results._past_plan_legend_added_g = True
        if h_label != "_nolegend_":
            animate_results._past_plan_legend_added_h = True

        animate_results._old_plan_lines_g.append(old_g)
        animate_results._old_plan_lines_h.append(old_h)

        # make fresh blue plan lines
        ax_g = animate_results._ax_g
        ax_h = animate_results._ax_h
        (new_plan_g,) = ax_g.plot([], [], "--", color="blue", linewidth=2,
                                  label="Long-horizon planned path")
        (new_plan_h,) = ax_h.plot([], [], "--", color="blue", linewidth=2,
                                  label="Long-horizon planned path")

        animate_results._line_plan_g = new_plan_g
        animate_results._line_plan_h = new_plan_h

        animate_results._last_plan_sig = sig
        animate_results._replan_count += 1

        # refresh legends once (gray lines excluded)
        ax_g.legend(loc="best", fontsize=8)
        ax_h.legend(loc="best", fontsize=8)

        return True

    # append actual position history
    if x_now is not None:
        animate_results._actual_x.append(x_now)
        animate_results._actual_y.append(y_now)
        animate_results._actual_h.append(h_now)

    actual_x = np.asarray(animate_results._actual_x, dtype=float)
    actual_y = np.asarray(animate_results._actual_y, dtype=float)
    actual_h = np.asarray(animate_results._actual_h, dtype=float)

    # current handles (may get replaced by archive)
    line_mpc_g    = animate_results._line_mpc_g
    line_actual_g = animate_results._line_actual_g
    plane_body    = animate_results._plane_body
    plane_patch   = animate_results._plane_patch

    line_mpc_h      = animate_results._line_mpc_h
    line_actual_h   = animate_results._line_actual_h
    first_pt_marker = animate_results._first_pt_marker

    bars_u = animate_results._bars_u
    xlim_min, xlim_max = animate_results._u_xlim
    state_box = animate_results._state_box

    # =========================
    # UPDATE: detect replan + archive previous plan (keep ALL grays)
    # =========================
    _maybe_archive_old_plan(x, y, s_abs, h)

    # refresh plan handles (might have changed)
    line_plan_g = animate_results._line_plan_g
    line_plan_h = animate_results._line_plan_h

    # =========================
    # UPDATE: ground track
    # =========================
    line_plan_g.set_data(y, x)         # (East, North)
    line_mpc_g.set_data(y_mpc, x_mpc)
    line_actual_g.set_data(actual_y, actual_x)
    animate_results._ax_g.set_title(title_ground)

    # Plane icon at CURRENT position
    if heading_deg is not None and x_now is not None:
        h_deg = float(np.asarray(heading_deg).ravel()[0])  # 0=N, 90=E
        theta = np.deg2rad(90.0 - h_deg)

        xlim = animate_results._ax_g.get_xlim()
        ylim = animate_results._ax_g.get_ylim()
        span_xy = max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]))
        scale = 0.05 * span_xy

        Rm = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
        ])
        plane_pts = (plane_body @ Rm.T) * scale
        plane_pts[:, 0] += y_now
        plane_pts[:, 1] += x_now
        plane_patch.set_xy(plane_pts)

    # =========================
    # UPDATE: altitude plot
    # =========================
    line_plan_h.set_data(s_abs, h)
    line_mpc_h.set_data(s_abs_mpc, h_mpc)

    # actual altitude vs runway distance |s|
    if actual_x.size > 0 and actual_y.size > 0:
        s_act = np.abs(actual_x * dx + actual_y * dy)
        line_actual_h.set_data(s_act, actual_h)
    else:
        line_actual_h.set_data([], [])

    animate_results._ax_h.set_title(title_alt)

    # marker at current
    if x_now is not None:
        s0_now = abs(x_now * dx + y_now * dy)
        first_pt_marker.set_data([s0_now], [h_now])
    else:
        first_pt_marker.set_data([], [])

    # =========================
    # UPDATE: bar chart
    # =========================
    for i, bar in enumerate(bars_u):
        bar.set_width(u[i] if i < u.size else 0.0)
    animate_results._ax_u.set_xlim(xlim_min, xlim_max)

    # --- Sim clock from current_step (1 step = 1 s) ---
    sim_sec = int(round(float(current_step)))  # robust to numpy scalars / floats
    hh, rem = divmod(sim_sec, 3600)
    mm, ss  = divmod(rem, 60)
    clock_str = f"{hh:02d}:{mm:02d}:{ss:02d}"

    replan_count = animate_results._replan_count

    hdg_str = f"{float(heading_deg):5.1f}°" if heading_deg is not None else "   —  "
    climb_str = f"{float(climb_angle_deg):5.1f}°" if climb_angle_deg is not None else "   —  "

    if x_now is not None:
        pos_str = f"x = {x_now:7.1f} m   y = {y_now:7.1f} m   h = {h_now:7.1f} m"
        s0 = x_now * dx + y_now * dy
        c0 = -x_now * dy + y_now * dx
        dist_km = abs(s0) / 1000.0
        dist_str = f"Distance to Runway: {dist_km:5.2f} km"
        xtk_str  = f"Cross-track:      {c0:7.1f} m"
        gamma_gs = np.deg2rad(glide_angle_deg)
        h_gs0 = abs(s0) * np.tan(gamma_gs)
        gs_err = float(h_now - h_gs0)
        gs_err_str = f"Glideslope error: {gs_err:7.1f} m"
    else:
        pos_str = "Position:   —"
        dist_str = "Distance to Runway:   —"
        xtk_str  = "Cross-track:      —"
        gs_err_str = "Glideslope error:   —"

    lines = [
        f"Clock:       {clock_str}   (t = {sim_sec:d} s)",
        f"Replans:     {replan_count:d}",
        "",
        f"Airspeed:    {vel_kt:6.1f} kt",
        f"Heading:     {hdg_str}",
        f"Climb angle: {climb_str}",
        "",
        pos_str,
        "",
        dist_str,
        xtk_str,
        gs_err_str,
    ]
    state_box.set_text("\n".join(lines))

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    return fig, (animate_results._ax_g, animate_results._ax_h, animate_results._ax_u)

