import os
import sys
import time
from sage.base_app import BaseApp

third_party_path = os.path.abspath(os.path.join(__file__, "../third_party"))
sys.path.insert(0, third_party_path)

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
sys.path.remove(third_party_path)


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

        # Predict GRF for right foot
        data_and_grfs_list_right = self.grf_predictor.update_stream(
            data, self.gait_phase_right, R_FOOT
        )

        # Predict GRF for left foot
        data_and_grfs_list_left = self.grf_predictor.update_stream(
            data, self.gait_phase_left, L_FOOT
        )

        # Process data for both feet
        if data_and_grfs_list_right or data_and_grfs_list_left:
            self.time_now += 0.01
            combined_data = {
                "time": [self.time_now],
                "weight_kg": [self.weight_kg],
                "height_meter": [self.height_meter],
                "RGRF_x": [0.0],
                "RGRF_y": [0.0],
                "RGRF_z": [0.0],
                "LGRF_x": [0.0],
                "LGRF_y": [0.0],
                "LGRF_z": [0.0],
                "Stance_Flag_Right": [0],
                "Stance_Flag_Left": [0],
            }

            if data_and_grfs_list_right:
                for (
                    _,
                    _,
                    _,
                    _,
                    GRF_X_right,
                    GRF_Y_right,
                    GRF_Z_right,
                    stance_flag_right,
                    _,
                ) in data_and_grfs_list_right:
                    combined_data.update(
                        {
                            "RGRF_x": [GRF_X_right],
                            "RGRF_y": [GRF_Y_right],
                            "RGRF_z": [GRF_Z_right],
                            "Stance_Flag_Right": [stance_flag_right],
                        }
                    )

            if data_and_grfs_list_left:
                for (
                    _,
                    GRF_X_left,
                    GRF_Y_left,
                    GRF_Z_left,
                    _,
                    _,
                    _,
                    _,
                    stance_flag_left,
                ) in data_and_grfs_list_left:
                    combined_data.update(
                        {
                            "LGRF_x": [GRF_X_left],
                            "LGRF_y": [GRF_Y_left],
                            "LGRF_z": [GRF_Z_left],
                            "Stance_Flag_Left": [stance_flag_left],
                        }
                    )

            # Send and save combined data
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
