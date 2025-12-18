import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button

from aircraft_model import AircraftModel
from path_planner import GlidePathPlanner
from guidance_mpc import GuidanceMPC
from plot import plot_guidance_overview

# -----------------------------
# Controller Settings
# -----------------------------
MPC_DT = 1.0
MPC_N = 20
PLANNER_N = 50

# -----------------------------
# Interactive UI to launch sim with user-defined initial conditions
# -----------------------------
def launch_ui():
    plt.ioff()
    fig = plt.figure(figsize=(15, 7))
    fig.canvas.manager.set_window_title("Glide MPC - Initial Conditions")

    # Status text at top center
    status_text = fig.text(0.5, 0.97, "", ha="center", va="top", fontsize=11)

    # Right-hand preview axes (East vs North)
    ax_preview = fig.add_subplot(1, 2, 2)
    ax_preview.set_title("Setup Preview")
    ax_preview.set_xlabel("East [m]")
    ax_preview.set_ylabel("North [m]")
    ax_preview.set_aspect("equal", adjustable="box")

    # Dummy left axes (we draw widgets over this using figure coords)
    ax_dummy = fig.add_subplot(1, 2, 1)
    ax_dummy.axis("off")

    # Text boxes positions [left, bottom, width, height] in figure coords
    ax_posN = plt.axes([0.12, 0.75, 0.30, 0.07])
    ax_posE = plt.axes([0.12, 0.65, 0.30, 0.07])
    ax_alt  = plt.axes([0.12, 0.55, 0.30, 0.07])
    ax_hd   = plt.axes([0.12, 0.45, 0.30, 0.07])
    ax_ca   = plt.axes([0.12, 0.35, 0.30, 0.07])
    ax_v    = plt.axes([0.12, 0.25, 0.30, 0.07])
    ax_rw   = plt.axes([0.12, 0.15, 0.30, 0.07])

    tb_posN = TextBox(ax_posN, "North [m]:",        initial="5000")
    tb_posE = TextBox(ax_posE, "East [m]:",         initial="5000")
    tb_alt  = TextBox(ax_alt,  "Alt [m]:",          initial="500")
    tb_hd   = TextBox(ax_hd,   "Heading [deg]:",    initial="30")
    tb_ca   = TextBox(ax_ca,   "Climb [deg]:",      initial="0")
    tb_v    = TextBox(ax_v,    "Speed [kt]:",       initial="80")
    tb_rw   = TextBox(ax_rw,   "Runway hdg [deg]:", initial="235")

    ax_btn = plt.axes([0.08, 0.03, 0.30, 0.08])
    btn_run = Button(ax_btn, "Run Simulation")

    # --- live preview function ---
    def draw_preview():
        ax_preview.cla()
        ax_preview.set_title("Setup Preview")
        ax_preview.set_xlabel("East [m]")
        ax_preview.set_ylabel("North [m]")
        ax_preview.set_aspect("equal", adjustable="box")
        try:
            posN = float(tb_posN.text)
            posE = float(tb_posE.text)
            hd   = float(tb_hd.text)
            rw   = float(tb_rw.text)
        except ValueError:
            # If invalid numbers, just skip drawing
            fig.canvas.draw_idle()
            return

        # Draw runway (threshold at origin, extending behind)
        E_rw, N_rw = _runway_polygon(rw)
        ax_preview.fill(
            E_rw, N_rw,
            facecolor="orange",
            edgecolor="black",
            alpha=0.7,
            label="Runway"
        )

        # Draw plane emoji at aircraft position
        angle_disp = 90.0 - hd  # 0 deg text rotation = +x (East)
        ax_preview.text(
            posE,
            posN,
            "✈",
            fontsize=28,
            ha="center",
            va="center",
            rotation=angle_disp,
            rotation_mode="anchor",
        )

        # Set view limits to include runway + plane, with padding
        xs = np.concatenate([E_rw, np.array([posE])])
        ys = np.concatenate([N_rw, np.array([posN])])
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        span_x = max(x_max - x_min, 1.0)
        span_y = max(y_max - y_min, 1.0)
        span = max(span_x, span_y)
        pad = 0.2 * span

        ax_preview.set_xlim(x_min - pad, x_max + pad)
        ax_preview.set_ylim(y_min - pad, y_max + pad)
        ax_preview.legend(loc="best")

        fig.canvas.draw_idle()

    # Call preview once at start
    draw_preview()

    # Hook preview to text box "enter" events
    def on_any_submit(_):
        draw_preview()

    for tb in [tb_posN, tb_posE, tb_alt, tb_hd, tb_ca, tb_v, tb_rw]:
        tb.on_submit(on_any_submit)

    def on_run_clicked(event):
        try:
            posN = float(tb_posN.text)
            posE = float(tb_posE.text)
            alt  = float(tb_alt.text)
            hd   = float(tb_hd.text)
            ca   = float(tb_ca.text)
            vel  = float(tb_v.text)
            rw   = float(tb_rw.text)
        except ValueError:
            status_text.set_text("Invalid input: please enter numbers.")
            fig.canvas.draw_idle()
            return

        msg = (f"Running simulation...\n"
               f"N={posN:.1f}, E={posE:.1f}, h={alt:.1f}, "
               f"hd={hd:.1f}°, ca={ca:.1f}°, V={vel:.1f} kt, RW={rw:.1f}°")
        print("[UI]", msg.replace("\n", " "))
        status_text.set_text(msg)
        fig.canvas.draw_idle()

        plt.pause(0.5)  # brief pause to show status update

        _run_simulation(posN, posE, alt, vel, hd, ca, rw)

        msg = (f"Simulation Ended")
        print("[UI]", msg.replace("\n", " "))
        status_text.set_text(msg)
        fig.canvas.draw_idle()


    btn_run.on_clicked(on_run_clicked)
    plt.show()

# Simple helper to build runway geometry
def _runway_polygon(runway_heading_deg, runway_length=2000.0, runway_width=200.0):
    """
    Return (E_coords, N_coords) for a rectangular runway whose THRESHOLD
    is at the origin, extending 'behind' the origin along the runway heading.

    - Runway heading = direction you land TOWARD the origin.
    - So the runway rectangle extends from 0 to -L in the heading direction.
    """
    h_rad = np.deg2rad(runway_heading_deg)
    dx = np.cos(h_rad)   # along-runway (North component)
    dy = np.sin(h_rad)   # along-runway (East component)

    L = runway_length
    W = runway_width

    # Local coordinates:
    # u = 0 at threshold (origin), negative behind the threshold
    corners_local = np.array(
        [
            [ L, -W / 2],   # threshold left
            [0,  -W / 2],   # far end left
            [0,   W / 2],   # far end right
            [ L,  W / 2],   # threshold right
        ]
    )
    u = corners_local[:, 0]
    v = corners_local[:, 1]

    # Transform to world: x = North, y = East
    N = u * dx - v * dy
    E = u * dy + v * dx
    return E, N


# -----------------------------
# A helper function that runs ONE full sim for given initial state
# -----------------------------
def _run_simulation(
    pos_north, pos_east, altitude,
    vel_kt, heading_deg, climb_angle_deg,
    runway_heading_deg
):
    # --- 1) Setup models ---
    aircraft = AircraftModel(
        pos_north=pos_north,
        pos_east=pos_east,
        altitude=altitude,
        vel_kt=vel_kt,
        heading_deg=heading_deg,
        climb_angle_deg=climb_angle_deg,
        dt=MPC_DT,
    )
    planner = GlidePathPlanner(runway_heading_deg, PLANNER_N)
    mpc = GuidanceMPC(planner, aircraft, MPC_N, MPC_DT)

    # --- 2) Logging containers ---
    x_planned = []
    y_planned = []
    h_planned = []

    x_mpc = []
    y_mpc = []
    h_mpc = []

    u_thrust_mpc = []
    u_heading_rate_mpc_deg_s = []
    u_climb_rate_mpc_deg_s = []

    x_sim_m = []
    y_sim_m = []
    h_sim_m = []
    vel_sim_ms = []
    heading_sim_deg = []
    climb_angle_sim_deg = []

    # initial state log
    x_sim_m.append(aircraft.pos_north)
    y_sim_m.append(aircraft.pos_east)
    h_sim_m.append(aircraft.altitude)
    vel_sim_ms.append(aircraft.vel_ms)
    heading_sim_deg.append(np.rad2deg(aircraft.chi))
    climb_angle_sim_deg.append(np.rad2deg(aircraft.gamma))

    sim_step = 0

    # --- 3) Forward simulate until we "reach" the runway (alt ~ 0) or MPC fails ---
    while h_sim_m[-1] > 0.1:
        try:
            u0, X_pred, U_pred, waypoints = mpc.solve_for_control_input(aircraft)
        except RuntimeError as e:
            print(f"[SIM] MPC failed at sim_step={sim_step}: {e}")
            break

        # control inputs
        u_thrust_mpc.append(u0[0])
        u_heading_rate_mpc_deg_s.append(np.rad2deg(u0[1]))
        u_climb_rate_mpc_deg_s.append(np.rad2deg(u0[2]))

        # planned waypoints
        x_planned.append(waypoints[:, 0])
        y_planned.append(waypoints[:, 1])
        h_planned.append(waypoints[:, 2])

        # MPC predicted trajectory
        x_mpc.append(X_pred[0, :])
        y_mpc.append(X_pred[1, :])
        h_mpc.append(X_pred[2, :])

        # "apply" first step of prediction
        x_sim_m.append(X_pred[0, 1])
        y_sim_m.append(X_pred[1, 1])
        h_sim_m.append(X_pred[2, 1])
        vel_sim_ms.append(X_pred[3, 1])
        heading_sim_deg.append(np.rad2deg(X_pred[4, 1]))
        climb_angle_sim_deg.append(np.rad2deg(X_pred[5, 1]))

        aircraft.update_from_vector(X_pred[:, 1])
        sim_step += 1

        if sim_step > 500:
            print("[SIM] Reached max sim steps, stopping.")
            break

    # --- 4) Replay the sim with your guidance overview plot ---
    plt.ion()
    replay_step = 0
    n_steps = len(x_planned)

    if n_steps == 0:
        print("[SIM] Nothing to replay (no MPC steps).")
        return

    while True:
        plot_guidance_overview(
            x_planned[replay_step],
            y_planned[replay_step],
            h_planned[replay_step],
            runway_heading_deg,
            x_mpc=x_mpc[replay_step],
            y_mpc=y_mpc[replay_step],
            h_mpc=h_mpc[replay_step],
            u_thrust=u_thrust_mpc[replay_step],
            u_heading=u_heading_rate_mpc_deg_s[replay_step],
            u_climb_rate=u_climb_rate_mpc_deg_s[replay_step],
            vel_ms=vel_sim_ms[replay_step],
            heading_deg=heading_sim_deg[replay_step],
            climb_angle_deg=climb_angle_sim_deg[replay_step],
        )

        if replay_step == 0:
            plt.pause(3.0)

        replay_step += 1
        if replay_step >= n_steps:
            plt.pause(3.0)
            break

        plt.pause(0.1)



if __name__ == "__main__":
    launch_ui()
