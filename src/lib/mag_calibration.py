import numpy as np
import pandas as pd
from scipy.linalg import sqrtm

def ellipsoid_fit(s):
    """
    Calculates ellipsoid parameters to normalize magnetometer data.

    From: https://teslabs.com/articles/magnetometer-calibration/
    """

    # D (samples)
    D = np.array([s[0]**2., s[1]**2., s[2]**2.,
                    2.*s[1]*s[2], 2.*s[0]*s[2], 2.*s[0]*s[1],
                    2.*s[0], 2.*s[1], 2.*s[2], np.ones_like(s[0])])

    # S, S_11, S_12, S_21, S_22 (eq. 11)
    S = np.dot(D, D.T)
    S_11 = S[:6,:6]
    S_12 = S[:6,6:]
    S_21 = S[6:,:6]
    S_22 = S[6:,6:]

    # C (Eq. 8, k=4)
    C = np.array([[-1,  1,  1,  0,  0,  0],
                  [ 1, -1,  1,  0,  0,  0],
                  [ 1,  1, -1,  0,  0,  0],
                  [ 0,  0,  0, -4,  0,  0],
                  [ 0,  0,  0,  0, -4,  0],
                  [ 0,  0,  0,  0,  0, -4]])

    # v_1 (eq. 15, solution)
    E = np.dot(np.linalg.inv(C),
                S_11 - np.dot(S_12, np.dot(np.linalg.inv(S_22), S_21)))

    E_w, E_v = np.linalg.eig(E)

    v_1 = E_v[:, np.argmax(E_w)]
    if v_1[0] < 0: v_1 = -v_1

    # v_2 (eq. 13, solution)
    v_2 = np.dot(np.dot(-np.linalg.inv(S_22), S_21), v_1)

    # quadric-form parameters
    M = np.array([[v_1[0], v_1[3], v_1[4]],
                  [v_1[3], v_1[1], v_1[5]],
                  [v_1[4], v_1[5], v_1[2]]])

    n = np.array([[v_2[0]],
                  [v_2[1]],
                  [v_2[2]]])

    d = v_2[3]

    return M, n, d

def calibrate(mag_data: pd.DataFrame, M=None, n=None, d=None, first=None, last=None) -> pd.DataFrame:
    """
    Returns a calibrated set of magnetometer samples.

    `mag_data` should be passed as a subset DataFrame with columns `[MagX, MagY, MagZ]`. 
    """

    # calculate ellipsoid parameters if not passed in
    if not (np.all(M) and np.all(n) and np.all(d)):
        if not (first is None or last is None):
            mag_data_clipped = mag_data.loc[first:last]
            print(mag_data_clipped.head)
            M, n, d = ellipsoid_fit(np.array(mag_data_clipped).T)

        else:
            M, n, d = ellipsoid_fit(np.array(mag_data).T)

    print(M, n, d)
    print(mag_data.head)

    # calculate calibration parameters for:
    # h_m = A @ h + b where h = A^-1 @ (h_m - b)
    M_1 = np.linalg.inv(M)
    b = -np.dot(M_1, n)
    A_1 = np.real(1 / np.sqrt(np.dot(n.T, np.dot(M_1, n)) - d) * sqrtm(M))

    # apply calibration to mag samples and return calibrated data 
    return mag_data.apply(__calibrate_row, args=(A_1, b), axis=1, result_type='expand')
    

def __calibrate_row(row, A_1, b):
    """
    Internal method to calculate correct magnetometer reading at a specific time.

    Assumes that `row` = `[MagX, MagY, MagZ]`.
    Uses correction equation `h = A^-1 @ (h_m - b)`. 
    """

    res = A_1 @ (np.c_[row] - b)
    return res.flatten().tolist()