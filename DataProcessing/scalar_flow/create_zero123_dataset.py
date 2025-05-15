import os
import sys

from multiprocessing import Pool

import cv2
import numpy as np

from p_tqdm import p_imap, p_map, p_uimap, p_umap
from tqdm import tqdm, trange


sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))

from utils.image_utils import pad_square


project_root = "/path/to/FluidNexusRoot"
scalar_flow_data_root = f"{project_root}/ScalarFlow/input_views_2023_03_31"
output_scalar_flow_data_root = f"{project_root}/ScalarFlow_zero123_dataset"


sim_ids = [i for i in range(104)]
frame_ids = [i for i in range(1, 162)]
camera_ids = [i for i in range(5)]


def one_process(sim_id: int):
    sim_input_dir = os.path.join(scalar_flow_data_root, f"sim_{sim_id:06d}", "input")
    for frame_id in tqdm(frame_ids, desc=f"sim_{sim_id:06d}"):
        for cam_id in camera_ids:
            # bg_frame_path = os.path.join(sim_input_dir, f"cam{cam_id}_no_bg", f"imgs_000000.png")
            frame_path = os.path.join(
                sim_input_dir, f"cam{cam_id}_no_denoise_no_bg_scale145", f"imgs_{frame_id:06d}.png"
            )
            if not os.path.exists(frame_path):
                print(f"frame {frame_path} does not exist")
                break
            img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            img_square = pad_square(img)

            out_path = os.path.join(output_scalar_flow_data_root, f"sim_{sim_id:03d}_frame_{frame_id:03d}")
            os.makedirs(out_path, exist_ok=True)
            out_frame_path = os.path.join(out_path, f"{cam_id:02d}.png")
            img = cv2.resize(img_square, (512, 512), cv2.INTER_AREA)
            cv2.imwrite(out_frame_path, img)
            # print(out_frame_path)


n_process = 64

p_umap(one_process, sim_ids, num_cpus=n_process)
