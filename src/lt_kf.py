import numpy as np
import pandas as pd
from scipy import integrate

import lib.csv_reader as reader
import lib.mag_calibration as mag_cal
import lib.plotter as plotter
import matplotlib.pyplot as plt
from lib.constants import *

#=========================================
TEST_NAME = "euler_angles_2"
DRAW_GRAPHS = False
#=========================================

# read test data at 96 samples/second
data, params = reader.read(TEST_NAME, same_sps=False)
freq = 1/params[7]

# calibrate magnetometer data
data[MAG_COLS] = mag_cal.calibrate(data[MAG_COLS])

# DEBUG: plot data
# if DRAW_GRAPHS:
#     plotter.draw_mag_sphere(data["MagX"], data["MagY"], data["MagZ"])
#     plotter.draw_all(data)

#Kalman Filter step functions
#refer to section 2 (p. 17) of Long Tran's paper (https://digitalcommons.calpoly.edu/eesp/400/)
def predict_state(prev_A_ang, prev_G_angvel, prev_bias):
    '''
    Parameters: 
    prev_A_ang - accelerometer angle at t-1 state
    prev_G_angvel - gyroscope anglular velocity at t-1 state
    prev_bias - gyroscope bias at t-1 state
    Returns:
    tuple w/ estimated angle, gyro bias in current state
    '''
    est_angle = prev_A_ang + freq*(prev_G_angvel - prev_bias)
    est_bias = prev_bias
    return (est_angle, est_bias)

def predict_covariance(P, Q_angle, Q_bias):
    '''
    Parameters: 
    P - 2x2 numpy array - error covariance matrix at t-1 state
    Q_angle - process noise variance for angle
    Q_bias -  process noise variance for gyro bias
    Returns:
    2x2 numpy array - estimated error covariance matrix at current state
    '''
    P[0][0] += freq * (freq*P[1][1] - P[0][1] - P[1][0] + Q_angle)
    P[0][1] -= freq * P[1][1]
    P[1][0] -= freq * P[1][1]
    P[1][1] += Q_bias * freq
    return P
    
def kalman_gain(P, R_measure):
    '''
    Parameters: 
    P - 2x2 numpy array - estimated error covariance matrix at current state
    R_measure - measurement error variance
    Returns:
    1x2 numpy array - Kalman Gain matrix
    '''
    K = np.array([0,0])
    S = P[0][0] + R_measure
    K [0] = P[0][0] / S
    K [1] = P[1][0] / S
    return K 

def update_state(measured_angle, est_angle, est_bias, K):
    '''
    Parameters: 
    measured_angle - measured accelerometer angle at current state
    est_angle - angle at current state estimated in predict phase
    est_bias - gyro bias at current state estimated in predict phase
    K - 1x2 numpy array - Kalman Gain matrix
    Returns:
    tuple w/ angle and gyro bias for current state updated by Kalman Gain
    '''
    y = measured_angle - est_angle
    angle = est_angle + K[0]*y
    bias = est_bias + K[1]*y
    return (angle, bias)
    
def update_covariance(P, K):
    '''
    Parameters: 
    P - 2x2 numpy array - estimated error covariance matrix at current state
    K - 1x2 numpy array - Kalman Gain matrix
    Returns:
    2x2 numpy array - error covariance matrix at current state updated by Kalman Gain
    '''
    P[0][0] -= K[0] * P[0][0]
    P[0][1] -= K[0] * P[0][1]
    P[1][0] -= K[1] * P[0][0]
    P[1][1] -= K[1] * P[0][1]
    return P

def kalman_filter(A_angle, G_angvel):
    '''
    Parameters:
    A_angle - time series of accelerometer-derived Euler angles
    G_angle - time series of gyroscope angular velocity values
    Returns:
    Single Euler angle time series reconciling the two input series using KF model.
    '''
    #initialize output series
    output = np.zeros(len(A_angle))
    output[0] = A_angle[0]
    #initialize angle
    angle = A_angle[0]
    
    '''Initialize error/covariance parameters'''
    #Initialize process covariance matrix
    Pk_1 = np.array([[1000000,0],[0,1000000]]) #assume variables are independent
    
    #Next 3 based on Long Tran's values - not sure exactly how instrumentation compares
    #Initialize process noise variance for the angle
    Q_angle = 0.001 
    #Initialize process noise variance for the gyro bias
    Q_bias = 0.003 
    #Initialize measurement error variance
    R_measure = 0.03  
    #initialize gyro bias
    bias = 0.00625 #how much has gyro drifted since last state?
    #^Estimate by looking how far gyro drifts over given time

    for t in range(1, len(A_angle)):
        #predict:
        (est_angle, est_bias) = predict_state(angle, G_angvel[t-1], bias)
        
        Pk = predict_covariance(Pk_1, Q_angle, Q_bias)
        
        #update:
        K = kalman_gain(Pk, R_measure)
        
        (angle, bias) = update_state(A_angle[t], est_angle, est_bias, K)
        #write updated angle value to output
        output[t] = angle
        
        Pk_1 = update_covariance(Pk, K)
    
    return output


    
def calc_e_angles(data):
   
    '''
    Calculate Euler angles from Accelerometer and Magnetometer
    Parameters: 
    Raw data dataframe
    Returns: Roll, Pitch, and Yaw series
    '''

    A_roll = np.arctan2(-data["AccelY"], -data["AccelZ"])*RAD_TO_DEG
    A_pitch = np.arctan2(-data["AccelX"], np.sqrt(data["AccelY"]**2 + data["AccelZ"]**2))*RAD_TO_DEG
    A_roll = A_roll.to_numpy()
    A_pitch = A_pitch.to_numpy()
    M_x = data["MagX"] * np.cos(A_pitch) + data["MagZ"] * np.sin(A_pitch)
    M_y = data["MagX"] * np.sin(A_roll) * np.sin(A_pitch) + data["MagY"] \
     * np.cos(A_roll) - data["MagZ"] * np.sin(A_roll) * np.cos(A_pitch)
    # remove NaNs from mag params
    M_x = M_x[~np.isnan(M_x)]
    M_y = M_y[~np.isnan(M_y)]
    M_yaw = np.arctan2(-M_y,M_x).to_numpy()*RAD_TO_DEG

    return A_roll, A_pitch, M_yaw


def k_filtered_angles(data):
    '''
    Use Accelerometer/Magnetometer-derived Euler angles and Gyro data in 
    Kalman Filter to calculate more accurate angles
    '''

    A_roll, A_pitch, M_yaw = calc_e_angles(data)
    
    k_roll = kalman_filter(A_roll, data['GyroX'].to_numpy())
    k_pitch = kalman_filter(A_pitch, data['GyroY'].to_numpy())
    k_yaw = kalman_filter(M_yaw, data['GyroZ'].to_numpy())
    return k_roll, k_pitch, k_yaw

def main():

    k_roll, k_pitch, k_yaw = k_filtered_angles(data)

    plt.plot(data['Time'], k_roll, label= "Roll")
    plt.plot(data['Time'], k_pitch, label= "Pitch")
    plt.plot(data['Time'], k_yaw, label= "Yaw")
    plotter.show_plot("K. Filtered Angles")

if __name__ == "__main__":
    main()