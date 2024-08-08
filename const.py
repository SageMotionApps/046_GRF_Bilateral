PRE_STATIONARY, POST_STATIONARY, WAIT_PREDICTION, PREDICTION_DONE = range(300, 304)
IMU_LIST = ["L_FOOT", "R_FOOT", "L_THIGH", "R_THIGH", "WAIST"]
(
    L_FOOT,
    R_FOOT,
    L_THIGH,
    R_THIGH,
    WAIST,
) = range(5)
IMU_FIELDS = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']
ACC_ALL = [field + '_' + sensor for sensor in IMU_LIST for field in IMU_FIELDS[:3]]
GYR_ALL = [field + '_' + sensor for sensor in IMU_LIST for field in IMU_FIELDS[3:]]

MAX_BUFFER_LEN = 152
GRAVITY = 9.81

WEIGHT_LOC=0
