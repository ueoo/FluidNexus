import os
import sys

from shutil import copyfile

from p_tqdm import p_umap
from tqdm import trange


sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))

from utils.image_utils import crop_and_resize
from utils.video_utils import images_to_video


project_root = "/path/to/FluidNexusRoot"
cogvx_out_root = f"{project_root}/cogvideox_lora_outputs"

part = "smoke"

if part == "smoke":
    gen_dataset_root = f"{project_root}/FluidNexus-Smoke"
    gen_pred_root = f"{cogvx_out_root}/5b_lora_all_sdedit_zero123_pi2v_long_smoke"
    zeroprefix = "zero123_finetune"
    sub_start_is = [55, 167, 279]
    finetune_step = 52000
    sdedit_strengths = [0.5]

elif part == "ball":
    gen_dataset_root = f"{project_root}/FluidNexus-Ball"
    gen_pred_root = f"{cogvx_out_root}/5b_lora_all_sdedit_zero123_pi2v_long_ball"
    zeroprefix = "zero123_finetune"
    sub_start_is = [33, 145, 257]
    finetune_step = 88000
    sdedit_strengths = [0.5]

elif part == "scalar":
    gen_dataset_root = f"{project_root}/ScalarReal"
    gen_pred_root = f"{cogvx_out_root}/5b_lora_all_sdedit_zero123_pi2v_long_scalar"
    zeroprefix = "zero123_finetune"
    sub_start_is = [20, 76, 131]
    finetune_step = 15500
    sdedit_strengths = [0.5]

src_view = 2
tgt_views = [0, 1, 3, 4]

fps = 30


def one_process(view_strength):
    tgt_view_id, sdedit_strength = view_strength
    strength_str = str(round(sdedit_strength, 2)).replace(".", "d")

    # one, two, three stands for the iteration number for long video generation
    subs = ["one", "two", "three"]

    frame_nums = 56
    cur_start_out_index = 0

    zero_cur_dataset_sub_folder = f"{zeroprefix}_{finetune_step}_cam{src_view}to{tgt_view_id}_cogvideox"
    zero_cur_strength_dataset_folder = os.path.join(gen_dataset_root, zero_cur_dataset_sub_folder)
    os.makedirs(zero_cur_strength_dataset_folder, exist_ok=True)

    cur_dataset_sub_folder = (
        f"{zeroprefix}_{finetune_step}_cam{src_view}to{tgt_view_id}_cogvxlora5b_strength{strength_str}"
    )
    cur_strength_dataset_folder = os.path.join(gen_dataset_root, cur_dataset_sub_folder)
    os.makedirs(cur_strength_dataset_folder, exist_ok=True)

    rawsize_frames_folder = f"{cur_strength_dataset_folder}_rawsize"
    # rmtree(rawsize_frames_folder, ignore_errors=True)
    os.makedirs(rawsize_frames_folder, exist_ok=True)

    for sub, sub_start_i in zip(subs, sub_start_is):
        cur_out_sub_folder = (
            f"{zeroprefix}_{finetune_step}_cam{src_view}to{tgt_view_id}_cogvideox_5b_all_pred_prefix_{sub}"
        )
        zero123_out = os.path.join(gen_pred_root, cur_out_sub_folder, f"input_sfi{sub_start_i}_nf65_fps8")
        cogvxlora_out = os.path.join(
            gen_pred_root, cur_out_sub_folder, f"output_sfi{sub_start_i:03d}_nf65_strength{strength_str}"
        )
        cur_start_in_index = 9

        for _ in trange(frame_nums, desc="Processing frames", leave=False):

            zero_out_frame_name = f"{cur_start_in_index:03d}.png"
            zero_out_frame_path = os.path.join(zero123_out, zero_out_frame_name)
            zero_copy_out_frame_name = f"frame_{cur_start_out_index:06d}.png"
            zero_copy_out_frame_path = os.path.join(zero_cur_strength_dataset_folder, zero_copy_out_frame_name)
            copyfile(zero_out_frame_path, zero_copy_out_frame_path)

            frame_name = f"{cur_start_in_index:03d}.png"
            frame_path = os.path.join(cogvxlora_out, frame_name)
            assert os.path.exists(frame_path), f"Frame not found: {frame_path}"
            dataset_frame_name = f"frame_{cur_start_out_index:06d}.png"
            cogvx_out_frame_path = os.path.join(cur_strength_dataset_folder, dataset_frame_name)
            copyfile(frame_path, cogvx_out_frame_path)

            rawsize_out_frame_path = os.path.join(rawsize_frames_folder, dataset_frame_name)
            crop_and_resize(frame_path, rawsize_out_frame_path, 1080, 1920, to_gray=False)
            cur_start_in_index += 1
            cur_start_out_index += 1

    zero_cur_video_path = f"{zero_cur_strength_dataset_folder}.mp4"
    images_to_video(zero_cur_strength_dataset_folder, f"*.png", zero_cur_video_path, fps=fps)

    raw_out_video_path = f"{rawsize_frames_folder}.mp4"
    images_to_video(rawsize_frames_folder, f"*.png", raw_out_video_path, fps=fps)


tgt_views_strengths = [(tgt_view_id, strength) for tgt_view_id in tgt_views for strength in sdedit_strengths]

n_process = len(tgt_views_strengths)

p_umap(one_process, tgt_views_strengths, num_cpus=n_process, desc="Processing views strengths", leave=True)
