import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt

import lib.csv_reader as reader
import lib.mag_calibration as mag_cal
import lib.plotter as plotter
from lib.constants import *

from pyquaternion import Quaternion

#=========================================
TEST_NAME = "euler_angles_2"
DRAW_GRAPHS = False
#=========================================

print("KGA algorithm started.")
print(f"Reading test '{TEST_NAME}'...")

# read test data at 96 samples/second
data, params = reader.read(TEST_NAME, same_sps=True)
data[ACC_COLS] = data[ACC_COLS].rolling(window=100).mean().fillna(data[ACC_COLS].iloc[49])

print("Test read.")
print("Calibrating magnetometer data...")

# calibrate magnetometer data
data[MAG_COLS] = mag_cal.calibrate(data[MAG_COLS])

print("Magnetometer calibration complete.")

# DEBUG: plot data
if DRAW_GRAPHS:
    plotter.draw_mag_sphere(data["MagX"], data["MagY"], data["MagZ"])
    plotter.draw_all(data)

def calc_lg_q(row):
    """
    Calculates the quaternion representing the orientation
    of the global frame relative to the local frame.

    Should be applied to the DataFrame containing trial data.
    """

    # create normalized acceleration vector
    acc = np.array(row[ACC_COLS])
    acc = acc / np.linalg.norm(acc)

    # create magnetometer vector
    mag = np.array(row[MAG_COLS])

    # calculate auxiliary quats
    q_acc = calc_q_acc(*acc)
    q_mag = calc_q_mag(*mag)

    # combine quats (Equation 13)
    lg_q = q_acc * q_mag

    return lg_q

def calc_q_acc(ax, ay, az):
    """
    Calculates the quaternion representing acceleration data, `q_acc` (Equation 25).
    
    The acceleration vector should be normalized before being passed into this function.
    """

    if az >= 0:
        q0 = sqrt((az + 1) / 2)
        q1 = -ay / sqrt(2*(az + 1))
        q2 = ax / sqrt(az + 1)
        q3 = 0

        return Quaternion(q0, q1, q2, q3)

    elif az < 0:
        q0 = -ay / sqrt(2*(1 - az))
        q1 = sqrt((1 - az) / 2)
        q2 = 0
        q3 = ax / sqrt(2 * (az + 1))

        return Quaternion(q0, q1, q2, q3)

def calc_q_mag(mx, my, mz):
    """
    Calculates the quaternion representing magnetometer data, `q_mag` (Equation 35).
    
    The magnetometer vector should be normalized and calibrated before being passed into this function.
    """

    # L represents gamma
    L = mx**2 + my**2

    if mx >= 0:
        q0 = sqrt(L + mx*sqrt(L)) / sqrt(2*L)
        q3 = my / (sqrt(2) * sqrt(L + mx*sqrt(L)))

        return Quaternion(q0, 0, 0, q3)

    elif mx < 0:
        q0 = my / (sqrt(2) * sqrt(L - mx*sqrt(L)))
        q3 = sqrt(L - mx*sqrt(L)) / sqrt(2*L)
        
        return Quaternion(q0, 0, 0, q3)

print("Calculating local frame quats...")

lg_q = data.apply(calc_lg_q, axis=1)

print("Local frame quats calculated.")
print("Converting to euler angle representation...")

ANGLES = ["Yaw", "Pitch", "Roll"]

lg_angles = lg_q.map(lambda x: x.yaw_pitch_roll).to_list()
lg_angles = pd.DataFrame(lg_angles, columns=ANGLES)
lg_angles["Time"] = data["Time"].to_list()

print("Euler angles calculated.")
print(lg_angles.head(20))

# plot roll/pitch/yaw independently for debugging
if DRAW_GRAPHS:
    plt.plot(lg_angles["Time"], lg_angles["Roll"] * RAD_TO_DEG)
    plotter.show_plot()

    plt.plot(lg_angles["Time"], lg_angles["Pitch"] * RAD_TO_DEG)
    plotter.show_plot()

    plt.plot(lg_angles["Time"], lg_angles["Yaw"] * RAD_TO_DEG)
    plotter.show_plot()

plotter.draw_sensor(lg_angles["Time"], lg_angles[ANGLES] * RAD_TO_DEG)