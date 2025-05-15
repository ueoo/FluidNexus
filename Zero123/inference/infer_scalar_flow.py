import os
import sys

import cv2
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from tqdm import trange


sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))

from helpers.camera_utils import get_T
from helpers.functions import load_model_from_config
from helpers.test_helpers import main_run_simple


def main_demo(project_root="/path/to/FluidNexusRoot", tgt_cam=0, cuda_idx=0, finetune_step=15500):
    src_cam = 2
    assert src_cam != tgt_cam, f"src {src_cam} and tgt {tgt_cam} cannot be the same"
    msg = f"src_cam: {src_cam}, tgt_cam: {tgt_cam}, cuda_idx: {cuda_idx}, finetune_step: {finetune_step}"
    print(msg)
    ckpt_root = f"{project_root}/zero123_finetune/logs"
    ckpt_name = f"step={finetune_step-1:09d}.ckpt" if isinstance(finetune_step, int) else "last.ckpt"

    dataset_name = "ScalarReal"

    data_root = f"{project_root}/{dataset_name}/zero123_dataset"

    white_bg = True
    ckpt_dir = "2024-04-24T14-04-16_scalar_flow"
    config = "configs/scalar_flow.yaml"
    out_root = f"{project_root}/{dataset_name}/zero123_finetune_{finetune_step}"

    ckpt = f"{ckpt_root}/{ckpt_dir}/checkpoints/{ckpt_name}"

    device = f"cuda:{cuda_idx}"
    config = OmegaConf.load(config)

    # Instantiate all models beforehand for efficiency.
    models = dict()
    print("Instantiating LatentDiffusion...")
    models["turncam"] = load_model_from_config(config, ckpt, device=device)

    num_frames = 161

    out_path = f"{out_root}_cam{src_cam}to{tgt_cam}"
    os.makedirs(out_path, exist_ok=True)

    for frame_id in (pbr := trange(num_frames, leave=False)):

        cond_img_path = f"{data_root}/frame_{frame_id:06d}/{src_cam:02d}.png"
        gt_img_path = f"{data_root}/frame_{frame_id:06d}/{tgt_cam:02d}.png"
        assert os.path.exists(cond_img_path), f"cond_img_path {cond_img_path} does not exist"
        assert os.path.exists(gt_img_path), f"gt_img_path {gt_img_path} does not exist"

        cond_cam_path = f"{data_root}/camera/{src_cam:02d}.npy"
        target_cam_path = f"{data_root}/camera/{tgt_cam:02d}.npy"
        cond_RT = np.load(cond_cam_path)
        target_RT = np.load(target_cam_path)

        d_T = get_T(target_RT, cond_RT)

        raw_im = cv2.imread(cond_img_path, cv2.IMREAD_GRAYSCALE)
        if white_bg:
            raw_im = 255 - raw_im

        gt_im = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
        if white_bg:
            gt_im = 255 - gt_im

        raw_im = cv2.cvtColor(raw_im, cv2.COLOR_GRAY2RGB)
        raw_image = Image.fromarray(raw_im)
        input_im = T.ToTensor()(raw_image).unsqueeze(0).to(device)
        input_im = input_im * 2 - 1
        input_im = TF.resize(input_im, [256, 256])

        gt_im = cv2.cvtColor(gt_im, cv2.COLOR_GRAY2RGB)
        gt_image = Image.fromarray(gt_im)
        target_im = T.ToTensor()(gt_image).unsqueeze(0).to(device)
        target_im = target_im * 2 - 1
        target_im = TF.resize(target_im, [256, 256])

        out_imgs = main_run_simple(models, device, d_T, raw_im=input_im, n_samples=1)
        img = out_imgs[0]

        save_path = f"{out_path}/frame_{frame_id:06d}.png"
        img.save(save_path)


if __name__ == "__main__":

    Fire(main_demo)
