# 046_GRF_Bilateral combined
The purpose of this GRF_Bilateral app is to predict and stream Ground Reaction Forces (GRFs) in real-time based on IMUs data, facilitating the analysis of force distribution during treadmill walking
### Installation Note:
The `third_party` folder has all the needed libraries and dependencies required.
Please refer to the App guide as a tutorial of how to use the App
For any technical assitance, please contact [Sagemotion](mailto:info@sagemotion.com) directly.

### Nodes Required: 7 
- Sensing (7): 
    - left foot (top of the foot, with switch pointing towards body)
    - right foot (top of the foot, with switch pointing towards body)
    - left shank (Midway between Femur Lateral Epicondyle and Fibula Apex of Lateral Malleolus, with switch pointing up)
    - right shank (Midway between Femur Lateral Epicondyle and Fibula Apex of Lateral Malleolus, with switch pointing up)
    - left thigh ( Midway between Femur Greater Trochanter and Femur Lateral Epicondyle, with switch pointing up)
    - right thigh ( Midway between Femur Greater Trochanter and Femur Lateral Epicondyle, with switch pointing up)
    - pelvis (Midway between Left and Right Anterior Superior Iliac Spine, with switch pointing up)

## Algorithm & Calibration
### Algorithm Information
Our deep learning framework is composed of multiple components designed to effectively process IMU data and predict ground reaction forces (GRFs) in realtime. The core architecture consists of two primary networks: the InertialNe for processing IMU data and the OutNet for predicting GRFs. Additionally, we introduce the LmfImuOnlyNet, which employs a low-rank multimodal fusion approach for combining accelerometer and gyroscope data.

The InertialNet is a GRU-based recurrent neural network designed to process sequential IMU data. Each InertialNet takes as input the data from either the accelerometer or gyroscope and outputs a high-dimensional representation.

Note: The algorithm and app is only valid for treadmill walking and has been trained on 17 healthy subjects, and the implementation and validation has been tested on subjects in the similar weight category as well. The data is publicly available on the [SimTK](https://simtk.org/projects/imukinetics) website.

### Example Data
Here you can find an [example data file (h5 format)](https://github.com/zakir300408/Ground_Reaction_Forces_Sagemotion/blob/main/trained_models_and_example_data/example_data.h5). It is a formatted example data file containing 10 walking step of 2 subjects. The dimension of each subject's data is: 10 steps * 152 samples * 256 data fields.

### Calibration Process:
No initial static calibration is performed to compensate for misalignment with the segment, so the user should be standing upright when starting the trial.


## Description of Data in Downloaded File
### Calculated Fields
- time (sec): time since trial start
- RGRF_x: X component of the GRF acting on the left foot (horizontal force in the medial-lateral
direction).
- RGRF_y: Y component of the GRF acting on the left foot (horizontal force in the anterior-posterior
direction).
- RGRF_z: Z component of the GRF acting on the left foot (vertical force in the superior-inferior
direction).
- LGRF_x: X component of the GRF acting on the right foot (horizontal force in the medial-lateral
direction).
- LGRF_y: Y component of the GRF acting on the right foot (horizontal force in the anterior-posterior
direction).
- LGRF_z: Z component of the GRF acting on the right foot (vertical force in the superior-inferior
direction).
- Stance_Flag_Right: 
  - 0 for swing phase or not walking
  - 1 for stance phase
- Stance_Flag_Left: 
  - 0 for swing phase or not walking
  - 1 for stance phase

### Sensor Raw Data Values 
Please Note: Each of the columns listed below will be repeated for each sensor
- SensorIndex: index of raw sensor data
- AccelX/Y/Z (m/s^2): raw acceleration data
- GyroX/Y/Z (deg/s): raw gyroscope data
- MagX/Y/Z (μT): raw magnetometer data
- Quat1/2/3/4: quaternion data (Scaler first order)
- Sampletime: timestamp of the sensor value
- Package: package number of the sensor value

# Development and App Processing Loop
The best place to start with developing an or modifying an app, is the [SageMotion Documentation](http://docs.sagemotion.com/index.html) page.
