# =============================================

# This license is additionally subject to the following restrictions:

# Licensor grants non-exclusive rights to use the Software for research purposes
# to research users (both academic and industrial), free of charge, without right
# to sublicense. The Software may be used "non-commercially", i.e., for research
# and/or evaluation purposes only.

# Subject to the terms and conditions of this License, you are granted a
# non-exclusive, royalty-free, license to reproduce, prepare derivative works of,
# publicly display, publicly perform and distribute its Work and any resulting
# derivative works in any form.
#

import json
import os
import uuid

from argparse import Namespace

import cv2
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(os.path.join(args.model_path, "training_render"), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, "simulation"), exist_ok=True)

    # Create Tensorboard writer
    tb_writer = None
    tb_writer = SummaryWriter(args.model_path)

    return tb_writer


def freeze_weights(model, key_list):
    for k in key_list:
        grad_tensor = getattr(getattr(model, k), "grad")
        new_grad = torch.zeros_like(grad_tensor)
        setattr(getattr(model, k), "grad", new_grad)
    return


def freeze_weights_by_mask(model, key_list, mask):
    for k in key_list:
        grad_tensor = getattr(getattr(model, k), "grad")
        new_grad = mask.unsqueeze(1) * grad_tensor  # torch.zeros_like(grad_tensor)
        setattr(getattr(model, k), "grad", new_grad)
    return


def freeze_weights_by_mask_no_unsqueeze(model, key_list, mask):
    for k in key_list:
        grad_tensor = getattr(getattr(model, k), "grad")
        new_grad = mask * grad_tensor  # torch.zeros_like(grad_tensor)
        setattr(getattr(model, k), "grad", new_grad)
    return


def remove_min_max(gaussians, max_bounds, min_bounds):
    max_x, max_y, max_z = max_bounds
    min_x, min_y, min_z = min_bounds
    xyz = gaussians._xyz
    mask0 = xyz[:, 0] > max_x.item()
    mask1 = xyz[:, 1] > max_y.item()
    mask2 = xyz[:, 2] > max_z.item()

    mask3 = xyz[:, 0] < min_x.item()
    mask4 = xyz[:, 1] < min_y.item()
    mask5 = xyz[:, 2] < min_z.item()
    mask = logical_or_list([mask0, mask1, mask2, mask3, mask4, mask5])
    gaussians.prune_points(mask)
    torch.cuda.empty_cache()


def control_gaussians(
    opt,
    gaussians,
    densify,
    iteration,
    scene,
    visibility_filter,
    radii,
    viewspace_point_tensor,
    flag,
    train_camera_with_distance=None,
    max_bounds=None,
    min_bounds=None,
    white_background=False,
    # max_timestamp=1.0,
    clone=True,
    split=True,
    split_prune=True,
    prune=True,
    level=0,
    no_densify_prune=False,
):
    if no_densify_prune:
        return

    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
        gaussians.densify_and_prune(
            opt.densify_grad_threshold,
            0.005,
            scene.cameras_extent,
            size_threshold,
            # max_timestamp=max_timestamp,
            clone=clone,
            split=split,
            split_prune=split_prune,
            prune=prune,
        )

    if iteration % opt.opacity_reset_interval == 0 or (white_background and iteration == opt.densify_from_iter):
        gaussians.reset_opacity()

    return flag


def logical_or_list(tensor_list):
    mask = None
    for idx, ele in enumerate(tensor_list):
        if idx == 0:
            mask = ele

        else:
            mask = torch.logical_or(mask, ele)
    return mask


def record_points_helper(model_path, numpoints, iteration, string):
    txt_path = os.path.join(model_path, "N_points_log.txt")

    with open(txt_path, "a") as file:
        file.write(f"Iter: {iteration}, Name: {string}, Num: {numpoints}\n")


def reload_helper(gaussians, opt, max_x=None, max_y=None, max_z=None, min_x=None, min_y=None, min_z=None):
    given_path = opt.prev_path

    gaussians.load_ply(given_path)
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
    return
