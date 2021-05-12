import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
import scipy.signal
import os
import subprocess
import time

import lib.csv_reader as reader
import lib.mag_calibration as mag_cal
import lib.plotter as plotter
from lib.constants import *

from pyquaternion import Quaternion

#=========================================
TEST_NAME = "euler_angles_2"
DEBUG_LEVEL = 0

# "BUTTER", "SMA", None to disable
APPLY_SMOOTHING = "SMA"
CALIBRATE_MAG = True
USE_PRECALC_MAG = False
CORRECT_MAG_AXES = True
ONLY_Q_MAG = False
NORM_HEADING = True
#=========================================

M = np.array([[ 0.56144721, -0.01910871, -0.01292889],
              [-0.01910871,  0.6276801,  -0.00568568],
              [-0.01292889, -0.00568568,  0.53873008]])

n = np.array([[-13.60233683],
              [ -1.60856291],
              [ 11.10481335]])

d = -390.59292573690266

#=========================================

print("KGA algorithm started.")
print(f"Reading test '{TEST_NAME}'...")

# read test data at 96 samples/second
data, params = reader.read(TEST_NAME, same_sps=True, correct_axes=CORRECT_MAG_AXES)

# normalized cutoff frequency = cutoff frequency / (2 * sample rate)
ORDER = 10
CUTOFF_FREQ = 50
NORM_CUTOFF_FREQ = CUTOFF_FREQ / (2 * 960)

# TODO: filtering mag causes inaccurate data
if APPLY_SMOOTHING == "BUTTER":
    # Butterworth filter
    num_coeffs, denom_coeffs = scipy.signal.butter(ORDER, NORM_CUTOFF_FREQ)
    for axis in ACC_COLS: data[axis] = scipy.signal.lfilter(num_coeffs, denom_coeffs, data[axis])
elif APPLY_SMOOTHING == "SMA":
    # simple moving average
    data[ACC_COLS] = data[ACC_COLS].rolling(window=50).mean().fillna(data[ACC_COLS].iloc[24])
    data[MAG_COLS] = data[MAG_COLS].rolling(window=50).mean().fillna(data[MAG_COLS].iloc[24])

print("Test read.")

if CALIBRATE_MAG:
    print("Calibrating magnetometer data...")

    if USE_PRECALC_MAG:
        data[MAG_COLS] = mag_cal.calibrate(data[MAG_COLS], M, n, d)
    else:
        data[MAG_COLS] = mag_cal.calibrate(data[MAG_COLS])

    print("Magnetometer calibration complete.")

# DEBUG: plot data
if DEBUG_LEVEL == 1:
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
    mag = mag / np.linalg.norm(mag)

    # calculate acceleration quat
    q_acc = calc_q_acc(*acc)

    # rotate mag vector into intermediate frame
    l_mag = q_acc.inverse.rotate(mag)

    # calculate mag quat
    q_mag = calc_q_mag(*l_mag)

    # combine quats (Equation 13)
    lg_q = q_acc * q_mag

    # only return q_mag if selected
    return lg_q if not ONLY_Q_MAG else q_mag

def calc_q_acc(ax, ay, az):
    """
    Calculates the quaternion representing acceleration data, `q_acc` (Equation 25).
    
    The acceleration vector should be normalized before being passed into this function.
    """

    if az >= 0:
        q0 = sqrt((az + 1) / 2)
        q1 = -ay / sqrt(2*(az + 1))
        q2 = ax / sqrt(2*(az + 1))
        q3 = 0

        return Quaternion(q0, q1, q2, q3)

    elif az < 0:
        q0 = -ay / sqrt(2*(1 - az))
        q1 = sqrt((1 - az) / 2)
        q2 = 0
        q3 = ax / sqrt(2 * (1 - az))

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

def calc_q_w(row):
    w = row[GYRO_COLS]
    lg_q = row["lg_q"]
    qdot_w = calc_qdot_w(*w, lg_q)
    return lg_q + qdot_w * (1/96)

def calc_qdot_w(wx, wy, wz, lg_q):
    w_quat = Quaternion(0, wx, wy, wz)
    return (-1/2) * w_quat * lg_q

print("Calculating local frame quats...")

lg_q = data.apply(calc_lg_q, axis=1)

print("Local frame quats calculated.")
print("Calculating gyro quats...")

# process all necessary data into DF
gyro_df = data[GYRO_COLS]
gyro_df["lg_q"] = lg_q.shift(1)
gyro_df = gyro_df.iloc[1:]

# calculate gyro quaternions
lg_q_w = gyro_df.apply(calc_q_w, axis=1)

print("Gyro quats calculated.")
print("Converting to Euler angles...")

ANGLES = ["Yaw", "Pitch", "Roll"]

lg_angles = lg_q.map(lambda x: x.yaw_pitch_roll).to_list()
lg_angles = pd.DataFrame(lg_angles, columns=ANGLES)
lg_angles = lg_angles.applymap(lambda x: x * RAD_TO_DEG)
lg_angles["Time"] = data["Time"].to_list()

# Euler angles for gyro quaternions
#==================================
# lg_angles = lg_q_w.map(lambda x: x.yaw_pitch_roll).to_list()
# lg_angles = pd.DataFrame(lg_angles, columns=ANGLES)
# lg_angles = lg_angles.applymap(lambda x: x * RAD_TO_DEG)
# lg_angles["Time"] = data["Time"].to_list()[1:]

if NORM_HEADING:
    heading_offset = lg_angles["Yaw"].head(48).mean()
    lg_angles["Yaw"] = lg_angles["Yaw"].map(lambda x: x - heading_offset)

print("Euler angles calculated.")

# plot roll/pitch/yaw independently for debugging
if DEBUG_LEVEL == 2:
    plt.plot(lg_angles["Time"], lg_angles["Roll"])
    plotter.show_plot()

    plt.plot(lg_angles["Time"], lg_angles["Pitch"])
    plotter.show_plot()

    plt.plot(lg_angles["Time"], lg_angles["Yaw"])
    plotter.show_plot()

plotter.draw_sensor(lg_angles["Time"], lg_angles[ANGLES], "ea_kga")

print("Saving Euler angles to 'out/ea_kga.csv'...")
lg_angles[["Roll", "Pitch", "Yaw"]].to_csv("out/ea_kga.csv", index=False, header=False)
print("Done.")

print("Saving lg quats to 'out/accelmag_quat_kga.csv'...")
lg_quat_arr = lg_q.map(lambda x: x.elements).to_list()
lg_quat_arr = pd.DataFrame(lg_quat_arr, columns=["w","x","y","z"])
lg_quat_arr.to_csv("out/accelmag_quat_kga.csv", index=False, header=False)
print("Done.")

print("Saving gyro quats to 'out/gyro_quat_kga.csv'...")
lg_w_quat_arr = lg_q_w.map(lambda x: x.elements).to_list()
lg_w_quat_arr = pd.DataFrame(lg_w_quat_arr, columns=["w","x","y","z"])
lg_w_quat_arr.to_csv("out/gyro_quat_kga.csv", index=False, header=False)
print("Done.")

print("Loading orientation view...")

root = os.path.dirname(os.path.abspath(__file__))
bin_path = os.path.join(root, "../bin")
file_path = os.path.join(root, "../out/quat_kga.csv")

subprocess.run(["orientation_view", "-sps", "96", "-file", file_path], cwd=bin_path, shell=True)