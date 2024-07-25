import numpy as np
from collections import deque
import itertools
from .const import (
    PRE_STATIONARY,
    POST_STATIONARY,
    WAIT_PREDICTION,
    PREDICTION_DONE,
    GRAVITY,
    MAX_BUFFER_LEN,
    R_FOOT,
    L_FOOT,
)


class GaitPhase:
    def __init__(self, foot=None):
        self.acc_thd = 1.2
        self.gyr_thd = np.rad2deg(2.6)
        self.current_phase = PRE_STATIONARY
        self.stationary_check_len = 10
        self.acc_magnitude = deque(maxlen=self.stationary_check_len)
        self.gyr_magnitude = deque(maxlen=self.stationary_check_len)
        self.data_buffer = deque(maxlen=250)
        self.strike_package, self.off_package = 0, 0
        self.right_foot = foot == R_FOOT
        self.left_foot = foot == L_FOOT
        assert (
            self.right_foot != self.left_foot
        ), "You must select either pass 'R_FOOT' or 'L_FOOT' for the gait phase"

    def update_gaitphase(self, foot_data):
        self.data_buffer.append((foot_data["Package"], foot_data["GyroX"]))
        if self.current_phase == PRE_STATIONARY:
            self.acc_magnitude.append(
                np.abs(
                    np.linalg.norm(
                        [foot_data["AccelX"], foot_data["AccelY"], foot_data["AccelZ"]],
                        ord=2,
                    )
                    - GRAVITY
                )
            )
            self.gyr_magnitude.append(
                np.linalg.norm(
                    [foot_data["GyroX"], foot_data["GyroY"], foot_data["GyroZ"]], ord=2
                )
            )
            if len(self.acc_magnitude) < self.stationary_check_len:
                return

            elif (np.array(self.acc_magnitude) < self.acc_thd).all() and (
                np.array(self.gyr_magnitude) < self.gyr_thd
            ).all():
                self.acc_magnitude.clear()
                self.gyr_magnitude.clear()
                for i_sample in range(len(self.data_buffer) - 1, 0, -1):
                    if (
                        self.right_foot
                        and self.data_buffer[i_sample][1] < -self.gyr_thd
                    ):
                        slice_to_search = list(
                            itertools.islice(
                                self.data_buffer, i_sample, len(self.data_buffer)
                            )
                        )
                        buffer_data = np.array(slice_to_search)
                        strike_index = np.argmax(buffer_data[:, 1])
                        self.strike_package = buffer_data[strike_index, 0]
                        self.data_buffer.clear()
                        break
                    if self.left_foot and self.data_buffer[i_sample][1] > self.gyr_thd:
                        slice_to_search = list(
                            itertools.islice(
                                self.data_buffer, i_sample, len(self.data_buffer)
                            )
                        )
                        buffer_data = np.array(slice_to_search)
                        strike_index = np.argmin(buffer_data[:, 1])
                        self.strike_package = buffer_data[strike_index, 0]
                        self.data_buffer.clear()
                        break
                self.current_phase = POST_STATIONARY
                return

        elif self.current_phase == POST_STATIONARY:
            if foot_data["GyroX"] < -self.gyr_thd and self.right_foot:
                buffer_data = np.array(self.data_buffer)
                off_index = np.argmax(buffer_data[:, 1])
                self.off_package = buffer_data[off_index, 0]
                if self.off_package - self.strike_package < MAX_BUFFER_LEN:
                    self.current_phase = WAIT_PREDICTION
                else:
                    self.current_phase = PRE_STATIONARY
                return
            if foot_data["GyroX"] > self.gyr_thd and self.left_foot:
                buffer_data = np.array(self.data_buffer)
                off_index = np.argmin(buffer_data[:, 1])
                self.off_package = buffer_data[off_index, 0]
                if self.off_package - self.strike_package < MAX_BUFFER_LEN:
                    self.current_phase = WAIT_PREDICTION
                else:
                    self.current_phase = PRE_STATIONARY
                return

        elif self.current_phase == WAIT_PREDICTION:
            return

        elif self.current_phase == PREDICTION_DONE:
            self.current_phase = PRE_STATIONARY
            return
