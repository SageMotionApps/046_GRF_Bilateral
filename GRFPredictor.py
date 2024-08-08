import os
from collections import deque
import copy
import random
import numpy as np
import pickle
from .const import (
    WAIT_PREDICTION,
    IMU_LIST,
    IMU_FIELDS,
    PREDICTION_DONE,
    MAX_BUFFER_LEN,
    R_FOOT,
    L_FOOT,
    WEIGHT_LOC,
    ACC_ALL,
    GYR_ALL,
    GRAVITY,
)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch
import threading
import time
import sklearn

lstm_unit, fcnn_unit = 20, 20


class InertialNet(nn.Module):
    def __init__(self, x_dim, net_name, seed=0, nlayer=1):
        super(InertialNet, self).__init__()
        self.net_name = net_name
        torch.manual_seed(seed)
        self.rnn_layer = nn.GRU(x_dim, globals()['lstm_unit'], nlayer, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        for name, param in self.rnn_layer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def __str__(self):
        return self.net_name

    def forward(self, sequence, lens):
        sequence = pack_padded_sequence(sequence, lens, batch_first=True, enforce_sorted=False)
        sequence, _ = self.rnn_layer(sequence)
        sequence, _ = pad_packed_sequence(sequence, batch_first=True, total_length=152)
        sequence = self.dropout(sequence)
        return sequence


class OutNet(nn.Module):
    def __init__(self, input_dim, device, output_dim=6):
        super(OutNet, self).__init__()
        self.linear_1 = nn.Linear(input_dim, globals()['fcnn_unit'], bias=True).to(device)
        self.linear_2 = nn.Linear(globals()['fcnn_unit'], output_dim, bias=True).to(device)
        self.relu = nn.ReLU().to(device)
        for layer in [self.linear_1, self.linear_2]:
            nn.init.xavier_normal_(layer.weight)

    def forward(self, sequence, others):
        sequence = self.linear_1(sequence)
        sequence = self.relu(sequence)
        sequence = self.linear_2(sequence)
        weight = others[:, 0, WEIGHT_LOC].unsqueeze(1).unsqueeze(2)
        sequence = torch.div(sequence, weight * GRAVITY / 100)
        return sequence


class LmfImuOnlyNet(nn.Module):
    def __init__(self, acc_dim, gyr_dim):
        super(LmfImuOnlyNet, self).__init__()
        self.acc_subnet = InertialNet(acc_dim, 'acc net', seed=0)
        self.gyr_subnet = InertialNet(gyr_dim, 'gyr net', seed=0)
        self.rank = 10
        self.fused_dim = 40
        self.device = torch.device("cpu")
        self.acc_factor = Parameter(torch.Tensor(self.rank, 1, 2 * globals()['lstm_unit'] + 1, self.fused_dim)).to(
            self.device)
        self.gyr_factor = Parameter(torch.Tensor(self.rank, 1, 2 * globals()['lstm_unit'] + 1, self.fused_dim)).to(
            self.device)
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank)).to(self.device)
        self.fusion_bias = Parameter(torch.Tensor(1, self.fused_dim)).to(self.device)

        self.out_net = OutNet(self.fused_dim, self.device, output_dim=6)
        nn.init.xavier_normal_(self.acc_factor, 10)
        nn.init.xavier_normal_(self.gyr_factor, 10)
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def __str__(self):
        return 'LMF IMU only net'

    def set_scalars(self, scalars):
        self.scalars = scalars

    def set_fields(self, x_fields):
        self.acc_fields = x_fields['input_acc']
        self.gyr_fields = x_fields['input_gyr']

    def forward(self, acc_x, gyr_x, others, lens):
        # convert inputs to float32 as the model expects it to be float32
        acc_x = acc_x.float()
        gyr_x = gyr_x.float()
        others = others.float()

        acc_h = self.acc_subnet(acc_x, lens)
        gyr_h = self.gyr_subnet(gyr_x, lens)
        batch_size = acc_h.data.shape[0]
        data_type = torch.FloatTensor

        _acc_h = torch.cat(
            (
                torch.autograd.Variable(
                    torch.ones(batch_size, acc_h.shape[1], 1)
                    .to(self.device)
                    .type(data_type),
                    requires_grad=False,
                ),
                acc_h,
            ),
            dim=2,
        )

        _gyr_h = torch.cat(
            (
                torch.autograd.Variable(
                    torch.ones(batch_size, gyr_h.shape[1], 1)
                    .to(self.device)
                    .type(data_type),
                    requires_grad=False,
                ),
                gyr_h,
            ),
            dim=2,
        )

        fusion_acc = torch.matmul(_acc_h, self.acc_factor)
        fusion_gyr = torch.matmul(_gyr_h, self.gyr_factor)
        fusion_zy = fusion_acc * fusion_gyr
        # permute to make batch first
        sequence = (
            torch.matmul(self.fusion_weights, fusion_zy.permute(1, 2, 0, 3)).squeeze(
                dim=2
            )
            + self.fusion_bias
        )

        sequence = self.out_net(sequence, others)
        return sequence


class GRFPredictor:
    def __init__(self, weight):
        self.data_buffer = deque(maxlen=MAX_BUFFER_LEN)
        self.data_margin_before_step = 20
        self.data_margin_after_step = 20
        self.data_array_fields = [
            axis + "_" + sensor for sensor in IMU_LIST for axis in IMU_FIELDS
        ]
        self.device = torch.device("cpu")
        base_path = os.path.abspath(os.path.dirname(__file__))
        model_state_path = base_path + "/models/7IMU_FUSION40_LSTM20_model.pth"
        self.model = LmfImuOnlyNet(15, 15)
        self.model.eval()
        self.model.load_state_dict(
            torch.load(model_state_path, map_location=torch.device("cpu"))
        )

        self.model.set_fields({"input_acc": ACC_ALL, "input_gyr": GYR_ALL})
        scalar_path = base_path + "/models/scalars.pkl"
        
        # load the scalars and handle the clip attribute 
        scalars = pickle.load(open(scalar_path, "rb"))
        if 'input_acc' in scalars and not hasattr(scalars['input_acc'], 'clip'):
            scalars['input_acc'].clip = False
        if 'input_gyr' in scalars and not hasattr(scalars['input_gyr'], 'clip'):
            scalars['input_gyr'].clip = False

        self.model.set_scalars(scalars)
        self.model.acc_col_loc = [
            self.data_array_fields.index(field) for field in self.model.acc_fields
        ]
        self.model.gyr_col_loc = [
            self.data_array_fields.index(field) for field in self.model.gyr_fields
        ]
        self.weight = weight
        anthro_data = np.zeros([1, 152, 1], dtype=float)
        anthro_data[:, :, WEIGHT_LOC] = self.weight
        self.model_inputs = {
            "others": torch.from_numpy(anthro_data),
            "step_length": None,
            "input_acc": None,
            "input_gyr": None,
        }

    def update_stream(self, data, gait_phase, foot):
        self.data_buffer.append([data, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0])
        package = data[foot]["Package"]

        if gait_phase.current_phase == WAIT_PREDICTION:
            if package - gait_phase.off_package >= self.data_margin_after_step - 1:
                step_length = int(
                    gait_phase.off_package
                    - gait_phase.strike_package
                    + self.data_margin_before_step
                    + self.data_margin_after_step
                )
                if step_length <= len(self.data_buffer):
                    inputs = self.transform_input(
                        step_length, self.data_buffer, self.model_inputs
                    )
                    pred = self.model(
                        inputs["input_acc"],
                        inputs["input_gyr"],
                        inputs["others"],
                        inputs["step_length"],
                    )
                    pred = pred.detach().numpy().astype(float)[0]
                    pred = pred * self.weight

                    if foot == R_FOOT:
                        for i_sample in range(step_length):
                            # Update last three GRF values for right foot, setting left foot GRFs to 0
                            self.data_buffer[-step_length + i_sample][1:4] = [
                                0.0,
                                0.0,
                                0.0,
                            ]
                            self.data_buffer[-step_length + i_sample][4:7] = pred[
                                i_sample
                            ][-3:]
                        for i_sample in range(
                            self.data_margin_before_step,
                            step_length - self.data_margin_after_step,
                        ):
                            self.data_buffer[-step_length + i_sample][7] = 1
                    elif foot == L_FOOT:
                        for i_sample in range(step_length):
                            # Update first three GRF values for left foot, set right foot GRFs to 0
                            self.data_buffer[-step_length + i_sample][1:4] = pred[
                                i_sample
                            ][:3]
                            self.data_buffer[-step_length + i_sample][4:7] = [
                                0.0,
                                0.0,
                                0.0,
                            ]
                        for i_sample in range(
                            self.data_margin_before_step,
                            step_length - self.data_margin_after_step,
                        ):
                            self.data_buffer[-step_length + i_sample][8] = 1

                gait_phase.current_phase = PREDICTION_DONE

        if len(self.data_buffer) == MAX_BUFFER_LEN:
            return [self.data_buffer.popleft()]

        return []

    def transform_input(self, step_length, data_buffer, model_inputs):
        raw_data = []
        for sample_data in list(data_buffer)[-step_length:]:
            raw_data_one_row = []
            for i_sensor in range(len(IMU_LIST)):
                raw_data_one_row.extend(
                    [sample_data[0][i_sensor][field] for field in IMU_FIELDS]
                )
            raw_data.append(raw_data_one_row)
        data = np.array(raw_data, dtype=float)
        data[:, self.model.acc_col_loc] = self.normalize_array_separately(
            data[:, self.model.acc_col_loc],
            self.model.scalars["input_acc"],
            "transform",
        )
        model_inputs["input_acc"] = torch.from_numpy(
            np.expand_dims(data[:, self.model.acc_col_loc], axis=0)
        )
        data[:, self.model.gyr_col_loc] = self.normalize_array_separately(
            data[:, self.model.gyr_col_loc],
            self.model.scalars["input_gyr"],
            "transform",
        )
        model_inputs["input_gyr"] = torch.from_numpy(
            np.expand_dims(data[:, self.model.gyr_col_loc], axis=0)
        )

        model_inputs["step_length"] = torch.tensor([step_length], dtype=torch.int32)
        return model_inputs

    @staticmethod
    def normalize_array_separately(data, scalar, method, scalar_mode="by_each_column"):
        input_data = data.copy()
        original_shape = input_data.shape
        target_shape = (
            [-1, input_data.shape[1]] if scalar_mode == "by_each_column" else [-1, 1]
        )
        input_data = input_data.reshape(target_shape)
        scaled_data = getattr(scalar, method)(input_data)
        scaled_data = scaled_data.reshape(original_shape)
        return scaled_data