import json
import os
import sys

from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from p_tqdm import p_imap, p_map, p_uimap, p_umap
from tqdm import tqdm, trange


dataset_name = "FluidNexusSmoke"
# dataset_name = "FluidNexusBall"

project_root = "/path/to/FluidNexusRoot"

cogvx_dataset_root = f"{project_root}/{dataset_name}All_cogvideox_dataset"

csv_file = f"{project_root}/{dataset_name}All/capture_set.csv"

# Read CSV file
df = pd.read_csv(csv_file)

# Iterate over rows
seqs = df.values.tolist()
sequences = [seq[0] for seq in seqs]

num_val_sequences = 20
train_sequences = sequences[num_val_sequences:]
val_sequences = sequences[:num_val_sequences]


paths_post = "20"

cam = -1
all_video_names = os.listdir(os.path.join(cogvx_dataset_root, "videos"))
all_video_names = [video_name for video_name in all_video_names if f".mp4" in video_name]
if cam != -1:
    cam_str = f"cam_{cam:02d}"
    all_video_names = [video_name for video_name in all_video_names if cam_str in video_name]
else:
    cam_str = "all"

print(f"Number of {cam_str} videos: {len(all_video_names)}")

all_video_names = sorted(all_video_names)
cur_cam_train_video_names = []
cur_cam_val_video_names = []

for video_name in all_video_names:
    # seq_10_23_22_53_16_cam_00_start_015_frames_049.txt
    cur_seq_name = video_name.split("_cam_")[0][4:]
    if cur_seq_name in train_sequences:
        cur_cam_train_video_names.append(video_name)
    elif cur_seq_name in val_sequences:
        cur_cam_val_video_names.append(video_name)

print(f"Number of train videos: {len(cur_cam_train_video_names)}")
print(f"Number of val videos: {len(cur_cam_val_video_names)}")

train_paths_json = f"{cogvx_dataset_root}/{cam_str}_train_paths{paths_post}.json"
with open(train_paths_json, "w") as f:
    json.dump(cur_cam_train_video_names, f)

val_paths_json = f"{cogvx_dataset_root}/{cam_str}_val_paths{paths_post}.json"
with open(val_paths_json, "w") as f:
    json.dump(cur_cam_val_video_names, f)
