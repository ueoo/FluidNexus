import os
import sys

from multiprocessing import Pool

import cv2
import numpy as np

from p_tqdm import p_imap, p_map, p_uimap, p_umap
from tqdm import tqdm, trange


sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))

from utils.image_utils import prepare_generative_image
from utils.video_utils import images_to_video


project_root = "/path/to/FluidNexusRoot"
scalar_flow_data_root = f"{project_root}/ScalarFlow/input_views_2023_03_31"
output_dataset_root = f"{project_root}/ScalarFlow_cogvideox_dataset"
output_dataset_frames_folder = os.path.join(output_dataset_root, "frames")
output_dataset_videos_folder = os.path.join(output_dataset_root, "videos")
os.makedirs(output_dataset_frames_folder, exist_ok=True)
os.makedirs(output_dataset_videos_folder, exist_ok=True)

num_sims = 104
num_cams = 5
min_frame_id = 10  # there is no smoke in first 6 frames
max_frames = 160
num_frames = 49
frame_step = 10
sim_ids = [i for i in range(num_sims)]
camera_ids = [i for i in range(num_cams)]
start_frame_idx = [i for i in range(min_frame_id, max_frames - num_frames + 1, frame_step)]

fps = 8
cogvidx_width = 720
cogvidx_height = 480


def one_process(sim_id: int):
    sim_input_dir = os.path.join(scalar_flow_data_root, f"sim_{sim_id:06d}", "input")
    for start_frame_id in tqdm(start_frame_idx, desc=f"sim_{sim_id:06d}"):
        for cam_id in camera_ids:
            cur_video_frames_path = os.path.join(
                output_dataset_frames_folder,
                f"sim_{sim_id:06d}_cam_{cam_id:02d}_start_{start_frame_id:03d}_frames_{num_frames:03d}",
            )
            os.makedirs(cur_video_frames_path, exist_ok=True)

            error_offset = 0
            for frame_id in range(start_frame_id, start_frame_id + num_frames):
                frame_path = os.path.join(
                    sim_input_dir, f"cam{cam_id}_no_denoise_no_bg_scale145", f"imgs_{frame_id+error_offset:06d}.png"
                )
                while not os.path.exists(frame_path):
                    print(f"Frame {frame_path} does not exist.")
                    error_offset += 1
                    frame_path = os.path.join(
                        sim_input_dir,
                        f"cam{cam_id}_no_denoise_no_bg_scale145",
                        f"imgs_{frame_id+error_offset:06d}.png",
                    )

                out_frame_path = os.path.join(cur_video_frames_path, f"{frame_id+error_offset:03d}.png")
                prepare_generative_image(
                    frame_path,
                    out_frame_path,
                    white_out_path=None,
                    width_new=cogvidx_width,
                    height_new=cogvidx_height,
                )
            cur_video_path = os.path.join(
                output_dataset_videos_folder,
                f"sim_{sim_id:06d}_cam_{cam_id:02d}_start_{start_frame_id:03d}_frames_{num_frames:03d}.mp4",
            )
            images_to_video(cur_video_frames_path, "*.png", cur_video_path, fps=fps)


n_process = os.cpu_count()

p_umap(one_process, sim_ids, num_cpus=n_process, leave=True)
