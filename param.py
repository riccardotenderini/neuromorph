# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pathlib
import os
from datetime import datetime

path_curr = str(pathlib.Path(__file__).parent.absolute())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_cpu = torch.device('cpu')

data_folder = os.path.join(os.getcwd(), "data")
meshes_folder = os.path.join(data_folder, "meshes")

data_folder_faust_remeshed = os.path.join(meshes_folder, "FAUST_r", "mat")
data_folder_mano_right = os.path.join(meshes_folder, "MANO_right", "mat")
data_folder_mano_test = os.path.join(meshes_folder, "MANO_test", "mat")
data_folder_shrec20 = os.path.join(meshes_folder, "SHREC_r", "mat")
data_folder_aortas_train = os.path.join(meshes_folder, "aortas_train", "mat")
data_folder_aortas_test = os.path.join(meshes_folder, "aortas_test", "mat")

chkpt_folder = os.path.join(data_folder, "checkpoint")
if not os.path.isdir(chkpt_folder):
    os.makedirs(chkpt_folder)
data_folder_out = os.path.join(data_folder, "out")
if not os.path.isdir(data_folder_out):
    os.makedirs(data_folder_out)


def get_timestr():
    now = datetime.now()
    time_stamp = now.strftime("%Y_%m_%d__%H_%M_%S")
    print("Time stamp: ", time_stamp)
    return time_stamp


def save_path(folder_str=None):
    if folder_str is None:
        folder_str = get_timestr()

    folder_path_models = os.path.join(chkpt_folder, folder_str)
    print("Checkpoint path: ", folder_path_models)
    return folder_path_models
