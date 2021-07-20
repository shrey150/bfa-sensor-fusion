import numpy as np
import pandas as pd
import csv

import errno
from os import path

from lib.constants import *

def read(test_name: str, same_sps=False, correct_axes=False, convert_to_rads=False, apply_gyro_bias=False) -> pd.DataFrame:
    """
    Reads raw test data from a given CSV and returns a Pandas DataFrame.

    Parameters
    ----------
    test_name: either a path to the CSV file or the name of the test in the `data` folder.
    same_sps: indicates whether accel/gyro data should be "downsampled" to match the mag's sample rate of 96sps.
    correct_axes: indicates whether magnetometer axes should be manipulated to align with accel/gyro axes (necessary for MPU-9250 data).
    convert_to_rads: indicates whether gyro readings should be converted from deg/s to rad/s (necessary for KGA algorithm).
    apply_gyro_bias: indicates whether gyro readings should be "normalized" to 0 (vertically shifting the data to rest on the x-axis).

    Assumes data is 960sps for accel/gyro readings and 96sps for mag.
    This method does NOT filter data, only returning the readings for each sensor.
    """

    # convert test_name to a path if it isn't already one
    file_path = test_name if path.exists(test_name) else f"data/{test_name}.csv"

    # read test params from CSVP
    csvp = open(file_path + "p")

    # create params array
    params = np.array([eval(line) for line in csvp])

    # read data from CSV
    data = pd.read_csv(file_path, names=AXES, index_col=False)

    sample_rate = params[7]

    # add time axis to data set
    time = np.linspace(0, len(data)/sample_rate, num=len(data), endpoint=False)
    data.insert(0, "Time", time)

    # sign data
    data = data.applymap(lambda x: x-65535 if x > 32767 else x)

    # apply accel sensitivity
    acc_sens = params[9]
    data[ACC_COLS] = data[ACC_COLS].applymap(lambda x: x * acc_sens * GRAVITY / 32768)

    # calculate conversion factor if selected
    GYRO_UNITS = DEG_TO_RAD if convert_to_rads else 1

    # apply gyro sensitivity
    gyro_sens = params[10]
    data[GYRO_COLS] = data[GYRO_COLS].applymap(lambda x: x * gyro_sens * GYRO_UNITS / 32768)

    # apply mag sensitivity
    mag_sens = 4800
    data[MAG_COLS] = data[MAG_COLS].applymap(lambda x: x * mag_sens / 32768)

    if apply_gyro_bias:
        # if selected, calculate gyro bias using first 0.5s of data
        gyro_offsets = data[GYRO_COLS].head(480).mean()

        # apply offsets to gyroscope (remove sensor bias)
        for i, axis in enumerate(GYRO_COLS):
            data[axis] = data[axis].map(lambda x: x - gyro_offsets[i])

    # if selected, manipulate axes to align mag with accel/gyro axes
    if correct_axes:
        data[["MagX","MagY"]] = data[["MagY","MagX"]]
        data["MagZ"] = -data["MagZ"]

    # reorder axes so that mag columns are in X-Y-Z order
    data = data[["Time"] + AXES]
    
    #fill null mag values with previous value
    data = data.fillna(method='ffill')

    # if enabled, only keep every 10th row to create 96sps data
    if same_sps:
        data = data.iloc[::10]
        params[7] /= 10

    # for some reason, the first mag data point is always erroneous, so remove its row
    data = data.iloc[1:]

    return data, params
