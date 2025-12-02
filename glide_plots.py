import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    runway_length=2000.0,
    runway_width=150.0,
    title="Ground Track (Top-down View)",
):
    """
    Plot 2D ground track.

    Convention:
      - x: North
      - y: East
      - Plot shows East on x-axis, North on y-axis (standard map-like view).
      - Runway starts at origin and extends along runway_heading_deg.
    """
    fig, ax = plt.subplots()

    # Aircraft path (East on x-axis, North on y-axis)
    ax.plot(y, x, "-o", label="Flight Path")

    # Runway axis
    heading_rad, dx, dy = _runway_axis(runway_heading_deg)

    # Runway rectangle
    L = runway_length
    W = runway_width

    # Local runway coords: u along runway, v left
    # Runway from u = 0 (threshold at origin) to u = +L
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

    # world coords (runway at origin):
    # x = u*dx - v*dy
    # y = u*dy + v*dx
    x_rw = u * dx - v * dy
    y_rw = u * dy + v * dx

    # Plot runway in same (East, North) frame as path
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
    ax.axis("equal")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    return fig, ax


def plot_altitude(
    x,
    y,
    h,
    runway_heading_deg,
    glide_angle_deg=3.0,
    title="Altitude vs Distance From Runway",
):
    """
    Plot altitude h versus absolute along-runway distance |s|, where:
      - s is the coordinate along the runway axis through the origin
      - |s| = 0 at the runway threshold
      - |s| increases with distance away from the runway (either side)

    Also plots a reference glide slope line with given glide_angle_deg
    starting at the runway (h = 0 at |s| = 0).
    """
    # Runway axis (runway at origin)
    heading_rad, dx, dy = _runway_axis(runway_heading_deg)

    # Signed along-runway coordinate
    s_raw = x * dx + y * dy

    # Absolute distance from runway along the runway axis
    s_abs = np.abs(s_raw)

    fig, ax = plt.subplots()

    # Aircraft altitude profile vs absolute distance
    ax.plot(s_abs, h, "-o", label="Path")

    # Reference glideslope from runway across full distance range
    gamma = np.deg2rad(glide_angle_deg)
    tan_gamma = np.tan(gamma)

    s_max = max(s_abs.max(), 0.0)
    s_gs = np.linspace(0.0, s_max, 200)
    h_gs = s_gs * tan_gamma  # h = |s| * tan(gamma), starting at (0,0)

    ax.plot(s_gs, h_gs, "--", label=f"{glide_angle_deg:.1f}° glideslope")

    ax.set_xlabel("Absolute distance from runway |s| [m]")
    ax.set_ylabel("Altitude h [m]")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    # Optional: runway at the right side of the plot
    ax.invert_xaxis()

    return fig, ax
