import os
import sys

from shutil import copyfile

import cv2

from tqdm import tqdm
from utils.image_utils import prepare_generative_image


project_root = "/path/to/FluidNexusRoot"


simulation_exp_name = "fluid_nexus_smoke_physical_future_simulation"
# simulation_exp_name = "fluid_nexus_smoke_physical_wind_simulation"
# simulation_exp_name = "fluid_nexus_ball_physical_future_simulation"

exp_path = f"{project_root}/fluid_nexus_dynamics_logs/{simulation_exp_name}/"

render_sub_dir = "training_render"
future_pred_cogvideox_images_dir = "training_render_for_cogvideox"

identifier_name = "0000"

os.makedirs(os.path.join(exp_path, future_pred_cogvideox_images_dir), exist_ok=True)

all_frames = os.listdir(os.path.join(exp_path, render_sub_dir))
frames = [frame for frame in all_frames if identifier_name in frame]

for frame in tqdm(frames, desc=f"{simulation_exp_name}"):
    src_path = os.path.join(exp_path, render_sub_dir, frame)
    out_path = os.path.join(exp_path, future_pred_cogvideox_images_dir, frame)

    prepare_generative_image(src_path, out_path, white_out_path=None, width_new=720, height_new=480)
