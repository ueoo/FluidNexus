import json
import os
import platform
import sys

from multiprocessing import Pool
from shutil import copyfile

import cv2
import numpy as np

from p_tqdm import p_imap, p_map, p_uimap, p_umap
from tqdm import tqdm, trange


sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))
from utils.image_utils import pad_square


project_root = "/path/to/FluidNexusRoot"


dataset_name = "FluidNexus-Smoke"
# dataset_name = "FluidNexus-Ball"
# dataset_name = "ScalarReal"

camera_prefix = "camera" if "FluidNexus" in dataset_name else "train"


fluid_nexus_data_root = f"{project_root}/{dataset_name}/"
output_dataset_root = f"{project_root}/{dataset_name}/zero123_dataset"


num_cameras = 5


def one_process(cam_id: int):
    cam_folder = os.path.join(fluid_nexus_data_root, f"{camera_prefix}{cam_id:02d}")
    frame_names = os.listdir(cam_folder)
    for frame_name in tqdm(frame_names):
        frame_path = os.path.join(cam_folder, frame_name)
        frame_id = int(frame_name.split(".")[0])

        img = cv2.imread(frame_path)
        img_square = pad_square(img)

        out_path = os.path.join(output_dataset_root, f"frame_{frame_id:03d}")
        os.makedirs(out_path, exist_ok=True)
        out_frame_path = os.path.join(out_path, f"{cam_id:02d}.png")
        img = cv2.resize(img_square, (512, 512), cv2.INTER_AREA)
        cv2.imwrite(out_frame_path, img)


n_process = num_cameras
camera_ids = [i for i in range(num_cameras)]
p_umap(one_process, camera_ids, num_cpus=n_process)
