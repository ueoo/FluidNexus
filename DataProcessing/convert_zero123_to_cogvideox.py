import os
import sys

from multiprocessing import Pool

import cv2
import numpy as np

from p_tqdm import p_imap, p_map, p_uimap, p_umap
from tqdm import tqdm, trange


sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))

from utils.image_utils import prepare_generative_image_crop_first
from utils.video_utils import images_to_video_gif


dataset_name_steps = [
    ("FluidNexus-Smoke", 52000),
    ("FluidNexus-Ball", 80000),
    ("ScalarReal", 15500),
]


src_cams = [2]
tgt_cams = [0, 1, 3, 4]


cogvidx_width = 720
cogvidx_height = 480

project_root = "/path/to/FluidNexusRoot"


def main_process(dataset_name_step):
    fps = 50 if "FluidNexus" in dataset_name_step[0] else 30

    zero123_out_root = f"{project_root}/{dataset_name_step[0]}/zero123_finetune_{dataset_name_step[1]}"

    def one_process(src_tgt_cam):
        source_cam, target_cam = src_tgt_cam
        if source_cam == target_cam:
            return
        zero123_output_folder = f"{zero123_out_root}_cam{source_cam}to{target_cam}"

        zero123_out_video_path = zero123_output_folder + ".mp4"
        images_to_video_gif(zero123_output_folder, "*.png", zero123_out_video_path, fps=fps)

        for_cogvx_output_folder = f"{zero123_output_folder}_for_cogvideox"
        os.makedirs(for_cogvx_output_folder, exist_ok=True)
        frame_names = os.listdir(zero123_output_folder)
        frame_names = [frame_name for frame_name in frame_names if frame_name.endswith(".png")]

        for frame_name in tqdm(frame_names):
            zero123_output_frame_path = os.path.join(zero123_output_folder, frame_name)
            for_cogvx_output_frame_path = os.path.join(for_cogvx_output_folder, frame_name)

            prepare_generative_image_crop_first(
                zero123_output_frame_path,
                for_cogvx_output_frame_path,
                white_out_path=None,
                width_new=cogvidx_width,
                height_new=cogvidx_height,
                source_is_white=False,
            )

        for_cogvx_video_path = for_cogvx_output_folder + ".mp4"
        images_to_video_gif(for_cogvx_output_folder, "*.png", for_cogvx_video_path, fps=fps)

    jobs = [(source_cam, target_cam) for source_cam in src_cams for target_cam in tgt_cams]
    n_process = len(jobs)

    p_umap(one_process, jobs, num_cpus=n_process, leave=True)


pool_num = len(dataset_name_steps)
pool = Pool(pool_num)
pool.imap_unordered(main_process, dataset_name_steps)
pool.close()
pool.join()
