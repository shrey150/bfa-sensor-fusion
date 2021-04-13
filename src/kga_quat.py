import numpy as np
import pandas as pd
from math import sqrt

import lib.csv_reader as reader
import lib.mag_calibration as mag_cal
import lib.plotter as plotter
from lib.constants import *

from pyquaternion import Quaternion

#=========================================
TEST_NAME = "mag_test_3"
DRAW_GRAPHS = False
#=========================================

# read test data at 96 samples/second
data = reader.read(TEST_NAME, same_sps=True)

# calibrate magnetometer data
data[MAG_COLS] = mag_cal.calibrate(data[MAG_COLS])

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

    q_acc = calc_q_acc(*acc)

    return q_acc


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

lg_q = data.apply(calc_lg_q, axis=1, result_type='expand')

# TODO: method to run for each time sample repeatedly over entire DF
# (might be bad practice but it's best to follow along w/ the algorithm closely)
# MAKE SURE TO NORMALIZE acc_x -> acc_z
def calc_row(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z): return None