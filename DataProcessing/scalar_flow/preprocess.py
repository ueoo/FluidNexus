import os
import sys

from multiprocessing import Pool

import cv2
import numpy as np

from p_tqdm import p_imap, p_map, p_uimap, p_umap
from tqdm import tqdm, trange


sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))
from scalar_flow.helpers import denoise, separate_background
from utils.video_utils import images_to_video


first_data = None
camera_count = 5
camera_ids = [2, 1, 0, 4, 3]  # this order is aligned to the info.json
n_process = os.cpu_count()
n_sims = 104


def one_process(sim_id: int):

    sim_input_path = f"/path/to/FluidNexusRoot/ScalarFlow/input_views_2023_03_31/sim_{sim_id:06d}/input"
    cam_data_path = f"{sim_input_path}/cam"

    file_names = sorted(os.listdir(cam_data_path))

    for t in trange(len(file_names), desc=f"sim_{sim_id:06d} processing"):
        file_name = file_names[t]
        file_path = os.path.join(cam_data_path, file_name)
        assert os.path.exists(file_path), f"{file_path} does not exist"
        # print(file_path)
        try:
            npz_data = np.load(file_path)
        except:
            print(f"Error loading {file_path}")
            continue
        assert "data" in npz_data, f"no data in {file_path}"
        frames_data = npz_data["data"]

        # print(f"frames: {frames_data.shape} {frames_data.dtype} {np.min(frames_data)} {np.max(frames_data)}")
        cur_filename = file_name.replace(".npz", ".png").replace("Unproc", "")

        for idx, camera_id in enumerate(camera_ids):
            camera_raw_path = f"{sim_input_path}/cam{camera_id}_raw"

            os.makedirs(camera_raw_path, exist_ok=True)
            cur_image = frames_data[idx]

            cur_image = cur_image * 255
            cur_image = cur_image.astype(np.uint8)
            cur_image = np.flip(cur_image, axis=0)

            out_path = f"{camera_raw_path}/{cur_filename}"
            # print(out_path)
            # print(cur_image.shape, cur_image.dtype, np.max(cur_image), np.min(cur_image))
            cv2.imwrite(out_path, cur_image)

            camera_denoise_path = f"{sim_input_path}/cam{camera_id}_denoise"
            os.makedirs(camera_denoise_path, exist_ok=True)
            denoise(cur_filename, camera_raw_path, camera_denoise_path)

            if t == 0:
                continue

            first_frame_name = f"imgs_000000.png"
            sep_bg_path = f"{sim_input_path}/cam{camera_id}_no_bg"
            os.makedirs(sep_bg_path, exist_ok=True)
            separate_background(cur_filename, first_frame_name, camera_denoise_path, sep_bg_path)

            sep_bg_no_denoise_path = f"{sim_input_path}/cam{camera_id}_no_denoise_no_bg"
            os.makedirs(sep_bg_no_denoise_path, exist_ok=True)
            separate_background(cur_filename, first_frame_name, camera_raw_path, sep_bg_no_denoise_path)

            sep_bg_scale_no_denoise_path = f"{sim_input_path}/cam{camera_id}_no_denoise_no_bg_scale145"
            os.makedirs(sep_bg_scale_no_denoise_path, exist_ok=True)
            separate_background(
                cur_filename, first_frame_name, camera_raw_path, sep_bg_scale_no_denoise_path, scale=1.45
            )

    for camera_id in trange(camera_count, desc=f"sim_{sim_id:06d} video"):
        images_to_video(
            f"{sim_input_path}/cam{camera_id}_raw",
            "imgs_*.png",
            f"{sim_input_path}/cam{camera_id}_raw.mp4",
        )
        images_to_video(
            f"{sim_input_path}/cam{camera_id}_denoise",
            "imgs_*.png",
            f"{sim_input_path}/cam{camera_id}_denoise.mp4",
        )
        images_to_video(
            f"{sim_input_path}/cam{camera_id}_no_bg",
            "imgs_*.png",
            f"{sim_input_path}/cam{camera_id}_no_bg.mp4",
        )
        images_to_video(
            f"{sim_input_path}/cam{camera_id}_no_denoise_no_bg",
            "imgs_*.png",
            f"{sim_input_path}/cam{camera_id}_no_denoise_no_bg.mp4",
        )
        images_to_video(
            f"{sim_input_path}/cam{camera_id}_no_denoise_no_bg_scale145",
            "imgs_*.png",
            f"{sim_input_path}/cam{camera_id}_no_denoise_no_bg_scale145.mp4",
        )

    return sim_id


sim_ids = [i for i in range(n_sims)]

p_umap(one_process, sim_ids, num_cpus=n_process)
