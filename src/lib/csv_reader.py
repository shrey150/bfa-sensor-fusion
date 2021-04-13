import numpy as np
import pandas as pd

from lib.constants import *

def read(test_name: str, same_sps: False) -> pd.DataFrame:
    """
    Reads raw test data from a given CSV and returns a Pandas DataFrame.
    
    This method does NOT filter data or apply biases, only returning the readings for each sensor.
    """

    # read test params from CSVP
    csvp = open(f"data/{test_name}.csvp")

    # create params array
    params = np.array([eval(line) for line in csvp])

    # read data from CSV 
    data = pd.read_csv(f"data/{test_name}.csv", names=AXES, index_col=False)

    sample_rate = params[7]

    # add time axis to data set
    time = np.arange(0, len(data)/sample_rate, 1/sample_rate)
    data.insert(0, "Time", time)

    # sign data
    data = data.applymap(lambda x: x-65535 if x > 32767 else x)

    # apply accel sensitivity
    acc_sens = params[9]
    data[ACC_COLS] = data[ACC_COLS].applymap(lambda x: x * acc_sens * GRAVITY / 32768)

    # apply gyro sensitivity
    gyro_sens = params[10]
    data[GYRO_COLS] = data[GYRO_COLS].applymap(lambda x: x * gyro_sens / 32768)

    # apply mag sensitivity
    mag_sens = 4800
    data[CSV_MAG_COLS] = data[CSV_MAG_COLS].applymap(lambda x: x * mag_sens / 8192)

    # invert X and Y mag axes to align with accel axes
    data[["MagX", "MagY"]] = -data[["MagX", "MagY"]]

    # reorder axes so that mag columns are in X-Y-Z order
    data = data[["Time"] + AXES]
    
    #fill null mag values with previous value
    data = data .fillna(method='ffill')

    # if enabled, remove every 10th row to create 96sps data
    if same_sps: 
        data = data.iloc[::10]
        params[7]/=10

    # for some reason, the first mag data point is always erroneous, so remove its row
    data = data.iloc[1:]

    return data, params