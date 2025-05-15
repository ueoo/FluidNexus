import os
import sys

from shutil import copyfile

import cv2

from tqdm import tqdm
from utils.image_utils import prepare_generative_image, unshift


project_root = "/path/to/FluidNexusRoot"
simulation_exp_name = "scalar_real_fluid_future_simulation"

exp_path = f"{project_root}/fluid_nexus_dynamics_logs/{simulation_exp_name}/"

render_sub_dir = "training_render"
unshift_sub_dir = "training_render_unshift"
future_pred_cogvideox_images_dir = "training_render_for_cogvideox"

identifier_name = "0000"

os.makedirs(os.path.join(exp_path, unshift_sub_dir), exist_ok=True)
os.makedirs(os.path.join(exp_path, future_pred_cogvideox_images_dir), exist_ok=True)

all_frames = os.listdir(os.path.join(exp_path, render_sub_dir))
frames = [frame for frame in all_frames if identifier_name in frame]

for frame in tqdm(frames, desc=f"exp: {simulation_exp_name}"):
    src_path = os.path.join(exp_path, render_sub_dir, frame)
    unshift_path = os.path.join(exp_path, unshift_sub_dir, frame)
    view_name = frame.split("_")[2]
    unshift(src_path, unshift_path, view_name, hack_type="scalar")
    out_path = os.path.join(exp_path, future_pred_cogvideox_images_dir, frame)
    prepare_generative_image(unshift_path, out_path, white_out_path=None, width_new=720, height_new=480)
