import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib.constants import ACC_COLS, GYRO_COLS, MAG_COLS

def show_plot(title=""):
    """Displays a plot with grid lines, a legend, and a title."""

    plt.title(title)
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.legend()

    if title: plt.savefig(f"out/{title}.png")
    plt.show()

def draw_mag_sphere(x_data, y_data, z_data):
    """Graphs a 3D scatter plot of tri-axis magnetometer data."""

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(x_data, y_data, z_data)

    # fit graph to data bounds and keep aspect at 1:1
    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis=1))

    ax.set_xlabel("Mag X")
    ax.set_ylabel("Mag Y")
    ax.set_zlabel("Mag Z")

    draw_sphere(ax=ax)

def draw_sphere(r=1, c=(0,0,0), ax=None):
    """
    Graphs a wireframe sphere in a 3D plot.

    By default, this function draws a unit sphere
    (a sphere with a radius of 1 and centered at the origin).
    """

    # if no plot is given, create a new one
    if not ax: ax = plt.figure().add_subplot(projection="3d")

    # calculate sphere using parametric equations
    u = np.linspace(0, np.pi, 30)
    v = np.linspace(0, 2 * np.pi, 30)
    x = np.outer(np.sin(u), np.sin(v))
    y = np.outer(np.sin(u), np.cos(v))
    z = np.outer(np.cos(u), np.ones_like(v))

    ax.plot_wireframe(c[0]+r*x, c[1]+r*y, c[2]+r*z, color="r", alpha=0.25)
    plt.savefig("out/kga_mag_cal.png")
    plt.show()

def draw_sensor(time, data: pd.DataFrame, graph_name: str=""):
    """
    Graphs data vs time for each axis of a given sensor.

    `time` should be an array-like object,
    and `data` should be a subset of a DataFrame containing the necessary axes.
    """

    for name, series in data.iteritems(): plt.plot(time, series, label=name)
    show_plot(graph_name)

def draw_all(data: pd.DataFrame):
    """Graphs all sensor data for a given DataFrame."""

    # define a 2D list so that each sensor can be iterated over
    SENSORS = [ACC_COLS, GYRO_COLS, MAG_COLS]

    for s_axes in SENSORS: draw_sensor(data["Time"], data[s_axes], (s_axes[0])[:-1])