import json
import os

import imageio
import numpy as np
import torch


project_root = "/path/to/FluidNexusRoot"
info_json_path = f"{project_root}/ScalarReal/info.json"


def get_w2c_RT_from_c2w(c2w):
    R = c2w[:3, :3]
    T = c2w[:3, 3]
    R_transpose = R.T
    T_w2c = -R_transpose @ T
    return R_transpose, T_w2c


with open(info_json_path, "r") as fp:
    # read render settings
    meta = json.load(fp)

splits = ["train", "test"]


video_name_to_transform = {}

for split in splits:
    video_list = meta[split + "_videos"]

    for video_id, train_video in enumerate(video_list):

        camera_hw = train_video["camera_hw"]
        H, W = camera_hw
        camera_angle_x = float(train_video["camera_angle_x"])
        Focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

        video_name_to_transform[train_video["file_name"].replace(".mp4", "")] = np.array(
            train_video["transform_matrix"]
        )

video_name_to_transform = sorted(video_name_to_transform.items(), key=lambda x: x[0])

all_poses = []
all_poses_set = []


zero123_dataset_path = "/drive/Dynamics/ScalarFlow_zero123_dataset"
camera_path = os.path.join(zero123_dataset_path, "camera")
os.makedirs(camera_path, exist_ok=True)

for i, (k, v) in enumerate(video_name_to_transform):

    C2W = np.array(v)
    C2W_set = C2W.copy()
    C2W_set[0, :] = C2W[2, :]
    C2W_set[1, :] = C2W[0, :]
    C2W_set[2, :] = C2W[1, :]

    C2W_rotZ = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ C2W
    C2W_rotY = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ C2W

    all_poses.append(C2W)
    all_poses_set.append(C2W_set)

    w2c_R, w2c_T = get_w2c_RT_from_c2w(C2W_set)
    zero123_camera = np.concatenate([w2c_R, w2c_T[:, None]], axis=1)
    zero123_camera_path = os.path.join(camera_path, f"{i:02d}.npy")
    print(k, zero123_camera_path)
    np.save(zero123_camera_path, zero123_camera)
