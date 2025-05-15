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


project_root = "/path/to/FluidNexusRoot"

dataset_name = "FluidNexusSmoke"
# dataset_name = "FluidNexusBall"

real_capture_data_root = f"{project_root}/{dataset_name}All"
output_dataset_root = f"{project_root}/{dataset_name}All_zero123_dataset"

csv_file = f"{real_capture_data_root}/capture_set.csv"

# Read CSV file
df = pd.read_csv(csv_file)

# Iterate over rows
seqs = df.values.tolist()
sequences = [seq[0] for seq in seqs]

sequence_to_cam = {}
for sequence in sequences:
    sequence_to_cam[sequence] = 1

out_seq_to_cam_json = f"{output_dataset_root}/seq_to_cam.json"
with open(out_seq_to_cam_json, "w") as f:
    json.dump(sequence_to_cam, f)


num_val_sequences = 20
train_sequences = sequences[num_val_sequences:]
val_sequences = sequences[:num_val_sequences]


paths_post = "20"

train_paths = []
for sequence in train_sequences:
    frames_folder = os.path.join(output_dataset_root, sequence)
    frames_names = os.listdir(frames_folder)
    frames_names = [os.path.join(sequence, f) for f in frames_names]
    train_paths.extend(frames_names)

train_paths_json = f"{output_dataset_root}/train_paths{paths_post}.json"
with open(train_paths_json, "w") as f:
    json.dump(train_paths, f)

val_paths = []
for sequence in val_sequences:
    frames_folder = os.path.join(output_dataset_root, sequence)
    frames_names = os.listdir(frames_folder)
    frames_names = [os.path.join(sequence, f) for f in frames_names]
    val_paths.extend(frames_names)
val_paths_json = f"{output_dataset_root}/val_paths{paths_post}.json"
with open(val_paths_json, "w") as f:
    json.dump(val_paths, f)
