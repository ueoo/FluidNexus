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
from utils.image_utils import pad_square


dataset_name = "FluidNexusSmoke"
# dataset_name = "FluidNexusBall"

project_root = "/path/to/FluidNexusRoot"
real_capture_data_root = f"{project_root}/{dataset_name}All"
output_dataset_root = f"{project_root}/{dataset_name}All_zero123_dataset"

csv_file = f"{real_capture_data_root}/capture_set.csv"

# Read CSV file
df = pd.read_csv(csv_file)

# Iterate over rows
seqs = df.values.tolist()


def one_process(seq):
    sequence = seq[0]
    cam_nums = 5

    for cam_id in range(cam_nums):
        cur_frames_folder = os.path.join(real_capture_data_root, sequence, f"camera{cam_id:02d}")
        assert os.path.exists(cur_frames_folder), f"Folder {cur_frames_folder} does not exist"
        frames = os.listdir(cur_frames_folder)
        frames = [f for f in frames if f.endswith(".png")]
        frames = sorted(frames)
        for frame_id, frame in enumerate(frames):
            frame_path = os.path.join(real_capture_data_root, sequence, f"camera{cam_id:02d}", frame)
            img = cv2.imread(frame_path)
            img_square = pad_square(img)

            out_path = os.path.join(output_dataset_root, sequence, f"frame_{frame_id:03d}")
            os.makedirs(out_path, exist_ok=True)
            out_frame_path = os.path.join(out_path, f"{cam_id:02d}.png")
            img = cv2.resize(img_square, (512, 512), cv2.INTER_AREA)
            cv2.imwrite(out_frame_path, img)


n_process = os.cpu_count()

p_umap(one_process, seqs, num_cpus=n_process, leave=True)
