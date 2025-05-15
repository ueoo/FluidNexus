import json
import os
import sys

from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from p_tqdm import p_imap, p_map, p_uimap, p_umap
from tqdm import tqdm, trange


sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))

from utils.image_utils import prepare_generative_image
from utils.video_utils import images_to_video


project_root = "/path/to/FluidNexusRoot"

dataset_name = "FluidNexusSmoke"
# dataset_name = "FluidNexusBall"

real_capture_data_root = f"{project_root}/{dataset_name}All"
output_dataset_root = f"{project_root}/{dataset_name}All_cogvideox_dataset"
output_dataset_frames_folder = os.path.join(output_dataset_root, "frames")
output_dataset_videos_folder = os.path.join(output_dataset_root, "videos")
os.makedirs(output_dataset_frames_folder, exist_ok=True)
os.makedirs(output_dataset_videos_folder, exist_ok=True)

csv_file = f"{real_capture_data_root}/capture_set.csv"

# Read CSV file
df = pd.read_csv(csv_file)

# Iterate over rows
seqs = df.values.tolist()

num_cams = 5
min_frame_id = 15
if dataset_name == "FluidNexusSmoke":
    num_all_frames = 370
elif dataset_name == "FluidNexusBall":
    num_all_frames = 480


start_frame_step = 5

frame_step = 2
num_frames = 49
fps = 8
cogvidx_width = 720
cogvidx_height = 480


def one_process(seq):
    sequence = seq[0]
    cur_sequence_dir = os.path.join(real_capture_data_root, sequence)

    start_indices = range(min_frame_id, num_all_frames - num_frames * frame_step, start_frame_step)
    for cam_id in range(num_cams):
        for start_idx in start_indices:
            cur_video_frames_path = os.path.join(
                output_dataset_frames_folder,
                f"seq_{sequence}_cam_{cam_id:02d}_start_{start_idx:03d}_frames_{num_frames:03d}",
            )
            os.makedirs(cur_video_frames_path, exist_ok=True)

            for frame_id in range(start_idx, start_idx + num_frames * frame_step, frame_step):

                frame_path = os.path.join(cur_sequence_dir, f"camera{cam_id:02d}", f"{frame_id:03d}.png")

                out_frame_path = os.path.join(cur_video_frames_path, f"{frame_id:03d}.png")
                prepare_generative_image(
                    frame_path,
                    out_frame_path,
                    white_out_path=None,
                    width_new=cogvidx_width,
                    height_new=cogvidx_height,
                )
            cur_video_path = os.path.join(
                output_dataset_videos_folder,
                f"seq_{sequence}_cam_{cam_id:02d}_start_{start_idx:03d}_frames_{num_frames:03d}.mp4",
            )
            images_to_video(cur_video_frames_path, "*.png", cur_video_path, fps=fps)


n_process = os.cpu_count()

p_umap(one_process, seqs, num_cpus=n_process, leave=True)
