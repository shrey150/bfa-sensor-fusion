import math

ACC_COLS = ["AccelX", "AccelY", "AccelZ"]
GYRO_COLS = ["GyroX", "GyroY", "GyroZ"]
MAG_COLS = ["MagX", "MagY", "MagZ"]
AXES = ACC_COLS + GYRO_COLS + MAG_COLS

# define flipped columns to correct mag axes from CSV
CSV_MAG_COLS = ["MagY", "MagX", "MagZ"]
CSV_COLS = ACC_COLS + GYRO_COLS + CSV_MAG_COLS

GRAVITY = 9.80665

RAD_TO_DEG = 180 / math.pi
DEG_TO_RAD = math.pi / 180