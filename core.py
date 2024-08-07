import os
import sys
import time
from sage.base_app import BaseApp


if __name__ == "__main__":
    from GRFPredictor import GRFPredictor
    import numpy as np
    from gait_phase import GaitPhase
    from const import R_FOOT, L_FOOT
else:
    from .GRFPredictor import GRFPredictor
    import numpy as np
    from .gait_phase import GaitPhase
    from .const import R_FOOT, L_FOOT


class Core(BaseApp):
    ###########################################################
    # INITIALIZE APP
    ###########################################################
    def __init__(self, my_sage):
        BaseApp.__init__(self, my_sage, __file__)
        self.gait_phase_right = GaitPhase(R_FOOT)
        self.gait_phase_left = GaitPhase(L_FOOT)
        self.grf_predictor = GRFPredictor(self.config["weight"], self.config["height"])
        self.time_now = 0

        self.height_meter = self.config["height"]
        self.weight_kg = self.config["weight"]

    #############################################################
    # UPON STARTING THE APP
    # If you have anything that needs to happen before the app starts
    # collecting data, you can uncomment the following lines
    # and add the code in there. This function will be called before the
    # run_in_loop() function below.
    #############################################################
    # def on_start_event(self, start_time):
    #     print("In On Start Event")

    #############################################################
    # RUN APP IN LOOP
    #############################################################
    def run_in_loop(self):
        data = self.my_sage.get_next_data()

        # Update gait phases for both feet
        self.gait_phase_right.update_gaitphase(data[R_FOOT])
        self.gait_phase_left.update_gaitphase(data[L_FOOT])

        # Predict GRF for both feet
        data_and_grfs = {
            R_FOOT: self.grf_predictor.update_stream(data, self.gait_phase_right, R_FOOT),
            L_FOOT: self.grf_predictor.update_stream(data, self.gait_phase_left, L_FOOT)
        }

        # Initialize combined data with zeros
        self.time_now += 0.01
        combined_data = {
            "time": [self.time_now],
            "RGRF_x": [0.0],
            "RGRF_y": [0.0],
            "RGRF_z": [0.0],
            "LGRF_x": [0.0],
            "LGRF_y": [0.0],
            "LGRF_z": [0.0],
            "Stance_Flag_Right": [0],
            "Stance_Flag_Left": [0],
        }

        # Process and send data for both feet
        for foot, data_list in data_and_grfs.items():
            for values in data_list:
                if foot == R_FOOT:
                    _, _, _, _, GRF_X, GRF_Y, GRF_Z, stance_flag, _ = values
                    combined_data["RGRF_x"] = [GRF_X]
                    combined_data["RGRF_y"] = [GRF_Y]
                    combined_data["RGRF_z"] = [GRF_Z]
                    combined_data["Stance_Flag_Right"] = [stance_flag]
                else:
                    _, GRF_X, GRF_Y, GRF_Z, _, _, _, _, stance_flag = values
                    combined_data["LGRF_x"] = [GRF_X]
                    combined_data["LGRF_y"] = [GRF_Y]
                    combined_data["LGRF_z"] = [GRF_Z]
                    combined_data["Stance_Flag_Left"] = [stance_flag]

                # Send and save data immediately
                self.my_sage.send_stream_data(data, combined_data)
                self.my_sage.save_data(data, combined_data)

        return True


    #############################################################
    # UPON STOPPING THE APP
    # If you have anything that needs to happen after the app stops,
    # you can uncomment the following lines and add the code in there.
    # This function will be called after the data file is saved and
    # can be read back in for reporting purposes if needed.
    #############################################################
    # def on_stop_event(self, stop_time):
    #     print(f"In On Stop Event: {stop_time}")
