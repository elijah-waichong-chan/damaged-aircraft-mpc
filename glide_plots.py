# glide_plots.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection


def _runway_axis(runway_pos, runway_heading_deg):
    """
    Helper: return (x_rwy, y_rwy, heading_rad, dx, dy)
    where (dx, dy) is the unit vector along runway heading.
    """
    x_rwy, y_rwy, _ = runway_pos
    heading_rad = np.deg2rad(runway_heading_deg)
    dx = np.cos(heading_rad)
    dy = np.sin(heading_rad)
    return x_rwy, y_rwy, heading_rad, dx, dy


def plot_3d_path(x, y, h, runway_pos, title="3D Glide Path"):
    """
    Plot 3D path (x, y, h) with runway threshold marked,
    and equal scaling on all axes.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, h, "-")
    ax.scatter(
        [runway_pos[0]],
        [runway_pos[1]],
        [runway_pos[2]],
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


def plot_ground_track_with_runway(
    x,
    y,
    runway_pos,
    runway_heading_deg,
    runway_length=2000.0,
    runway_width=150.0,
    title="Ground Track",
):
    """
    Plot 2D ground track (x, y), runway centerline, and a
    rectangular runway area extending 'runway_length' behind
    the threshold along runway heading.
    """
    fig, ax = plt.subplots()

    # Aircraft path
    ax.plot(x, y, "-o", label="Flight Path")

    # Runway threshold
    x_rwy, y_rwy, heading_rad, dx, dy = _runway_axis(
        runway_pos, runway_heading_deg
    )
    # ax.scatter([x_rwy], [y_rwy], c="r", label="Runway threshold")

    # # Centerline: length based on path extent
    # s_vals = (x - x_rwy) * dx + (y - y_rwy) * dy
    # max_abs_s = np.max(np.abs(s_vals))
    # if max_abs_s == 0:
    #     max_abs_s = 1.0

    # s_line = np.linspace(0.0, 1.2 * max_abs_s, 2)
    # x_cl = x_rwy + s_line * dx
    # y_cl = y_rwy + s_line * dy
    # ax.plot(x_cl, y_cl, "--", label="Runway centerline")

    # Runway rectangle
    L = runway_length
    W = runway_width

    # Local runway coords: u along runway, v left
    # Here we define runway from u = 0 (start) back to u = -L
    corners_local = np.array(
        [
            [0.0,  -W / 2],
            [-L,   -W / 2],
            [-L,    W / 2],
            [0.0,   W / 2],
        ]
    )
    u = corners_local[:, 0]
    v = corners_local[:, 1]

    # world = origin + u*[dx,dy] + v*[-dy, dx]
    x_rw = x_rwy + u * dx - v * dy
    y_rw = y_rwy + u * dy + v * dx

    # Fill rectangle with orange
    ax.fill(
        x_rw,
        y_rw,
        facecolor="orange",
        edgecolor="black",
        alpha=0.8,
        label="Runway"
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    return fig, ax


def plot_altitude_vs_runway_distance(
    x,
    y,
    h,
    runway_pos,
    runway_heading_deg,
    title="Altitude vs Runway Distance",
):
    """
    Plot altitude h versus along-runway distance s, where
    s = 0 at the runway threshold and positive along runway heading.
    """
    x_rwy, y_rwy, heading_rad, dx, dy = _runway_axis(
        runway_pos, runway_heading_deg
    )

    # Along-runway distance from threshold
    s = (x - x_rwy) * dx + (y - y_rwy) * dy

    fig, ax = plt.subplots()
    ax.plot(s, h, "-o")
    ax.set_xlabel("Runway axis distance s [m]")
    ax.set_ylabel("Altitude h [m]")
    ax.set_title(title)
    ax.grid(True)

    # Optional: put runway at s=0 on the right-hand side
    # ax.invert_xaxis()

    return fig, ax
