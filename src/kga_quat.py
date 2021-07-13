import numpy as np
import pandas as pd
from math import sqrt, acos, sin, fabs
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
# CHANGE THIS LINE TO USE A DIFFERENT TEST
TEST_NAME = "euler_angles_2"

#=========================================
# KGA magnetometer calibration data range config parameters

MAG_CAL_START = None
MAG_CAL_END = None

#=========================================
# KGA algorithm config parameters

APPLY_SMOOTHING = None      # "BUTTER", "SMA", None to disable
CALIBRATE_MAG = True        # should be disabled if mag data is already calibrated
USE_PRECALC_MAG = False     # uses hard-coded mag calibration parameters
CORRECT_MAG_AXES = True     # re-aligns mag axes to match accel/gyro axes (needed for MPU-9250 data)
NORM_HEADING = True         # normalizes yaw in euler angles graph (cosmetic, does not affect calculations)

#=========================================
# KGA debugging parameters (not commonly used)

DEBUG_LEVEL = 1             # displays more detailed data graphs when set to 1
ONLY_Q_MAG = False          # only returns the mag quat from `calc_lg_q`
ONLY_CALC_ACCELMAG = False  # excludes gyro from orientation calculations
ONLY_CALC_GYRO = False      # only calculates gyro quat for orientation
HIDE_ROLL = True            # if selected, hides roll from graph

#=========================================
# KGA complementary filter parameters
GAIN = 0.01
BIAS_ALPHA = 0.01
GYRO_THRESHOLD = 0.2
ACC_THRESHOLD = 0.1
DELTA_GYRO_THRESHOLD = 0.1

USE_ADAPTIVE_GAIN = True
UPDATE_GYRO_BIAS = True
#=========================================
# Hard-coded mag parameters for "euler_angles_2"
# (intended to be used with `USE_PRECALC_MAG`)

M = np.array([[ 0.56144721, -0.01910871, -0.01292889],
              [-0.01910871,  0.6276801,  -0.00568568],
              [-0.01292889, -0.00568568,  0.53873008]])

n = np.array([[-13.60233683],
              [ -1.60856291],
              [ 11.10481335]])

d = -390.59292573690266

#=========================================
# SEE ALSO: KGA C++ implementation, published by the authors of the original paper
# https://github.com/ccny-ros-pkg/imu_tools/blob/indigo/imu_complementary_filter/src/complementary_filter.cpp

print("KGA algorithm started.")

if not os.path.isdir('out'):
    print("output folder does not exist, creating new.")
    os.makedirs("out")

print(f"Reading test '{TEST_NAME}'...")

# read test data at 96 samples/second and convert gyro data to rads
data, params = reader.read(TEST_NAME, same_sps=True, correct_axes=CORRECT_MAG_AXES, convert_to_rads=True, apply_gyro_bias=True)

if APPLY_SMOOTHING == "BUTTER":
    # Butterworth filter parameters (somewhat arbitrary, but not being used)
    ORDER = 10
    CUTOFF_FREQ = 50

    # normalized cutoff frequency = cutoff frequency / (2 * sample rate)
    NORM_CUTOFF_FREQ = CUTOFF_FREQ / (2 * 960)

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
        data[MAG_COLS] = mag_cal.calibrate(data[MAG_COLS], first=MAG_CAL_START, last=MAG_CAL_END)

    print("Magnetometer calibration complete.")

# DEBUG: plot data
if DEBUG_LEVEL == 1:
    if not (MAG_CAL_START is None or MAG_CAL_END is None):
        print(data["MagY"].loc[MAG_CAL_START:MAG_CAL_END].head)
        plotter.draw_mag_sphere(data["MagX"].loc[MAG_CAL_START:MAG_CAL_END], data["MagY"].loc[MAG_CAL_START:MAG_CAL_END], data["MagZ"].loc[MAG_CAL_START:MAG_CAL_END])
    else:
        plotter.draw_mag_sphere(data["MagX"], data["MagY"], data["MagZ"])

    plotter.draw_all(data)

##############################################################################
#                       BEGIN KGA ALGORITHM FUNCTIONS
##############################################################################

# previous orientation quat
lg_q_prev: Quaternion = None

# previous gyro vector
w_prev: np.array = np.array([0,0,0])

# current gyro bias
w_bias: np.array = np.array([0,0,0])

def calc_lg_q_accelmag(row):
    """
    Calculates the quaternion representing the orientation
    of the global frame relative to the local frame,
    using only accel and mag calculations.

    Should be used to calculate the initial orientation.
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
    return lg_q

def calc_lg_q(row):
    """
    Calculates the quaternion representing the orientation
    of the global frame relative to the local frame.

    Should be applied to the DataFrame containing trial data.
    """

    global lg_q_prev

    # create normalized acceleration vector
    acc = np.array(row[ACC_COLS])
    acc_mag = np.linalg.norm(acc)
    acc = acc / acc_mag

    # create normalized magnetometer vector
    mag = np.array(row[MAG_COLS])
    mag = mag / np.linalg.norm(mag)

    # create gyro vector and remove current bias
    gyro = np.array(row[GYRO_COLS])

    # update gyro bias calculation
    if UPDATE_GYRO_BIAS: update_gyro_bias(acc_mag, gyro)

    # correct for gyro bias
    gyro -= w_bias

    # calculate adaptive gain from acc if selected
    alpha = calc_gain(GAIN, acc_mag) if USE_ADAPTIVE_GAIN else GAIN

    # calculate gyro quaternion
    lg_q_w = calc_q_w(*gyro)

    if ONLY_CALC_GYRO:
        lg_q_t = lg_q_w
        return lg_q_w

    # rotate acc vector into frame
    g_pred = lg_q_w.inverse.rotate(acc)

    # calculate acceleration quat
    q_acc = calc_q_acc(*g_pred)

    # TODO: LERP/SLERP q_acc
    q_acc_adj = scale_quat(alpha, q_acc)

    # calculate intermediate quat
    lg_q_prime = lg_q_w * q_acc_adj

    # rotate mag vector into intermediate frame
    l_mag = lg_q_prime.inverse.rotate(mag)

    # calculate mag quat
    q_mag = calc_q_mag(*l_mag)

    # TODO: LERP/SLERP q_mag
    q_mag_adj = scale_quat(alpha, q_mag)

    # combine quats (Equation 13)
    lg_q_prev = lg_q_prime * q_mag_adj

    # only return q_mag if selected
    return lg_q_prev if not ONLY_Q_MAG else q_mag

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

def calc_q_w(wx, wy, wz):
    """
    Calculates the quaternion representing gyroscope data, `q_w` (Equation 42).
    """

    # calculate delta gyro quaternion
    w_quat = Quaternion(0, wx, wy, wz)
    dq_w = (-1/2) * w_quat * lg_q_prev

    # add delta gyro quat to previous orientation
    return lg_q_prev + dq_w * (1/96)

def calc_gain(alpha, a_mag):
    """
    Calculates the adaptive gain for scaling correction quaternions.
    Will return a floating point number between 0 and 1.
    """
    error = abs(a_mag - GRAVITY)/GRAVITY

    bound1, bound2 = 0.1, 0.2
    m = 1.0/(bound1 - bound2)
    b = 1.0 - m * bound1

    factor = 0.0

    if error < bound1:
        factor = 1.0
    elif error < bound2:
        factor = m*error + b
    else:
        factor = 0.0

    return factor*alpha

def is_steady_state(acc_mag, wx, wy, wz):
    """
    Checks if the module is in a steady state with no external dynamic motion or rotation.
    """

    # check if module is in nongravitational dynamic motion
    if abs(acc_mag - GRAVITY) > ACC_THRESHOLD: return False

    # check if module has changed angular acceleration
    if (abs(wx - w_prev[0]) > DELTA_GYRO_THRESHOLD
    or  abs(wy - w_prev[1]) > DELTA_GYRO_THRESHOLD
    or  abs(wz - w_prev[2]) > DELTA_GYRO_THRESHOLD): return False

    # check if module is currently rotating
    if (abs(wx - w_bias[0]) > GYRO_THRESHOLD
    or  abs(wy - w_bias[1]) > GYRO_THRESHOLD
    or  abs(wz - w_bias[2]) > GYRO_THRESHOLD): return False

    # if none of those conditions are true, the module is in a steady state
    return True

def update_gyro_bias(acc_mag, w):
    """
    Calculates new gyro bias if the module is in a steady state.
    """

    global w_bias
    global w_prev

    if is_steady_state(acc_mag, *w):
        w_bias = BIAS_ALPHA * (w - w_bias)
        if DEBUG_LEVEL:
            print(f"Module at rest, updating gyro bias: {w_bias}")

    # update previous gyro calculation
    w_prev = w

def scale_quat(gain, quat):
    """
    Scales the given quaternion by an interpolation with the identity quaternion.
    Uses LERP or SLERP depending on the angle between the quaternion and identity quaternion.
    """

    # LERP (to be more efficient):
    if quat[0] > 0.9:
        q0 = (1 - gain) + gain * quat[0]
        q1 = gain * quat[1]
        q2 = gain * quat[2]
        q3 = gain * quat[3]

        return Quaternion(q0, q1, q2, q3).normalised

    # SLERP
    else:
        angle = acos(quat[0])
        A = sin(angle * (1 - gain)) / sin(angle)
        B = sin(angle * gain) / sin(angle)

        q0 = A + B * quat[0]
        q1 = B * quat[1]
        q2 = B * quat[2]
        q3 = B * quat[3]

        return Quaternion(q0, q1, q2, q3).normalised

##############################################################################
#                       END KGA ALGORITHM FUNCTIONS
##############################################################################

print("Calculating initial orientation...")

# calculate initial orientation 
lg_q_prev = calc_lg_q_accelmag(data.iloc[0])

print("Initial orientation calculated.")
print("Calculating orientations w/ gyro data...")

if not DEBUG_LEVEL:
    print("Updating Gyro Biases")

# choose selected orientation calculation function
calc_func = calc_lg_q_accelmag if ONLY_CALC_ACCELMAG else calc_lg_q

lg_q = data.apply(calc_func, axis=1)

print("Orientations calculated.")
print("Converting to Euler angles...")

ANGLES = ["Yaw", "Pitch", "Roll"]

lg_angles = lg_q.map(lambda x: x.yaw_pitch_roll).to_list()
lg_angles = pd.DataFrame(lg_angles, columns=ANGLES)
lg_angles = lg_angles.applymap(lambda x: x * RAD_TO_DEG)
lg_angles["Time"] = data["Time"].to_list()

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

# if selected, don't graph roll
GRAPHED_ANGLES = ANGLES if not HIDE_ROLL else ["Yaw", "Pitch"]

plotter.draw_sensor(lg_angles["Time"], lg_angles[GRAPHED_ANGLES], "ea_kga")

print("Saving Euler angles to 'out/ea_kga.csv'...")
lg_angles[["Roll", "Pitch", "Yaw"]].to_csv("out/ea_kga.csv", index=False, header=False)
print("Done.")

print("Saving quats to 'out/quat_kga.csv'...")
lg_quat_arr = lg_q.map(lambda x: x.elements).to_list()
lg_quat_arr = pd.DataFrame(lg_quat_arr, columns=["w","x","y","z"])
lg_quat_arr.to_csv("out/quat_kga.csv", index=False, header=False)
print("Done.")

print("Loading orientation view...")

root = os.path.dirname(os.path.abspath(__file__))
bin_path = os.path.join(root, "../bin")
file_path = os.path.join(root, "../out/quat_kga.csv")

subprocess.run(["orientation_view", "-sps", "96", "-file", file_path], cwd=bin_path, shell=True)