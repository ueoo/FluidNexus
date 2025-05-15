import os
import sys

from hmac import new
from multiprocessing import Pool
from shutil import copyfile, rmtree

import cv2
import numpy as np

from p_tqdm import p_imap, p_map, p_uimap, p_umap
from tqdm import tqdm, trange


from utils.image_utils import crop_and_resize
from utils.video_utils import images_to_video


run_id = "wind_smoke"

dataset_name = "FluidNexusSmoke"
frame_nums = 65
start_frame_idx = 50
sdedit_strengths = [0.55]

project_root = "/path/to/FluidNexusRoot"
cogvxlora_root = f"{project_root}/cogvideox_lora_outputs"
gen_pred_root = f"{cogvxlora_root}/5b_lora_all_sdedit_wind_pi2v_fluid_nexus_smoke"
gen_dataset_root = f"{project_root}/{dataset_name}"

view_ids = [0, 1, 2, 3, 4]
# view_ids = [0]

fps = 30
prefix_frame_nums = 9
to_gray = False
camera_prefix = "camera" if "FluidNexus" in dataset_name else "train"


def one_process(view_strength):
    view_id, sdedit_strength = view_strength
    strength_str = str(round(sdedit_strength, 2)).replace(".", "d")

    cur_dataset_sub_folder = f"{camera_prefix}{view_id:02d}_cogvxlora5b_prefix{prefix_frame_nums}_i2v3_strength{strength_str}_start{start_frame_idx}_{run_id}"
    cur_strength_dataset_folder = os.path.join(gen_dataset_root, cur_dataset_sub_folder)

    rawsize_frames_folder = f"{cur_strength_dataset_folder}_rawsize"
    # rmtree(rawsize_frames_folder, ignore_errors=True)
    os.makedirs(rawsize_frames_folder, exist_ok=True)

    cogvxlora_folder = f"{camera_prefix}{view_id:02d}_for_cogvideox_5b_all_pred_prefix_start{start_frame_idx}_future"
    out_frames_folder = f"output_sfi{start_frame_idx:03d}_nf{frame_nums}_strength{strength_str}"
    cogvxlora_out = os.path.join(gen_pred_root, cogvxlora_folder, out_frames_folder)

    for frame_idx in trange(prefix_frame_nums, frame_nums, desc="Processing frames", leave=False):
        frame_name = f"{frame_idx:03d}.png"
        frame_path = os.path.join(cogvxlora_out, frame_name)
        assert os.path.exists(frame_path), f"Frame not found: {frame_path}"
        dataset_frame_name = f"frame_{frame_idx-prefix_frame_nums+start_frame_idx:06d}.png"
        out_frame_path = os.path.join(rawsize_frames_folder, dataset_frame_name)
        crop_and_resize(frame_path, out_frame_path, 1080, 1920, to_gray=to_gray)

    raw_out_video_path = f"{rawsize_frames_folder}.mp4"
    images_to_video(rawsize_frames_folder, f"*.png", raw_out_video_path, fps=fps)


views_strengths = [(view_id, strength) for view_id in view_ids for strength in sdedit_strengths]

n_process = len(views_strengths)

p_umap(one_process, views_strengths, num_cpus=n_process, desc="Processing views strengths", leave=True)
