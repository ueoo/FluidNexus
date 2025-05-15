import os
import sys

from argparse import Namespace
from random import randint

import lovely_tensors as lt
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm, trange


sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_splatting.gm_background import GaussianModel
from helpers.helper_gaussian import get_model
from helpers.helper_parser import get_parser, write_args_to_file
from helpers.helper_pipe import get_render_pipe
from helpers.helper_train import prepare_output_and_logger
from scene import Scene
from utils.graphics_utils import get_world_2_view
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


def train(args: Namespace, model_args: ModelParams, optim_args: OptimizationParams, pipe_args: PipelineParams):

    write_args_to_file(args, model_args, optim_args, pipe_args, "training")
    tb_writer = prepare_output_and_logger(model_args)
    rendering_folder = os.path.join(args.model_path, "training_render")
    render_func, GRsetting, GRzer = get_render_pipe(pipe_args.rd_pipe)

    print(f"Model: {model_args.model}")
    Gaussian = get_model(model_args.model)

    gaussians: GaussianModel = Gaussian(model_args.sh_degree)

    scene = Scene(model_args, gaussians, loader=model_args.loader)

    gaussians.training_setup(optim_args)

    num_channel = 3  # this is the render channel

    bg_color = 1 if model_args.white_background else 0
    bg_color = [bg_color] * num_channel
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        all_cam_poses = []
        all_cam_trans_gl = []
        viewpoint_stack = scene.get_train_cameras().copy()
        for viewpoint in viewpoint_stack:
            rendered_pkg = render_func(
                viewpoint,
                gaussians,
                pipe_args,
                background,
                GRsetting=GRsetting,
                GRzer=GRzer,
            )
            initial_image = rendered_pkg["render"]
            save_image(
                initial_image,
                os.path.join(rendering_folder, f"initial_render_{viewpoint.image_name}.png"),
            )
            gt_image = viewpoint.original_image.cuda()
            save_image(
                gt_image,
                os.path.join(rendering_folder, f"gt_{viewpoint.image_name}.png"),
            )

            W2C = get_world_2_view(viewpoint.R, viewpoint.T)
            C2W = np.linalg.inv(W2C)  # this cam2world is COLMAP convention
            C2W_gl = C2W.copy()
            C2W_gl[:3, 1:3] *= -1
            C2W_gl_trans = C2W_gl[:3, 3]

            all_cam_poses.append(C2W)
            all_cam_trans_gl.append(C2W_gl_trans)

        all_cam_poses = np.stack(all_cam_poses, axis=0)
        np.save(os.path.join(scene.model_path, "gs_all_cam_poses.npy"), all_cam_poses)

        all_cam_trans_gl = np.stack(all_cam_trans_gl, axis=0)

    if optim_args.prune_near_cam_interval > 0:
        gaussians.set_cam_locations(all_cam_trans_gl)

    if optim_args.prune_near_interval > 0:
        gaussians.set_near_params(optim_args)

    # iter_start = torch.cuda.Event(enable_timing=True)
    # iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    desc_str = "Training progress"

    # current_xyz = gaussians.get_xyz
    # xyz_min = torch.min(current_xyz, dim=0).values
    # xyz_max = torch.max(current_xyz, dim=0).values
    # print(f"Initial XYZ min: {xyz_min}, max: {xyz_max}")

    for iteration in (pbar := trange(1, optim_args.iterations + 1, desc=desc_str)):
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         (
        #             custom_cam,
        #             do_training,
        #             pipe_args.convert_SHs_python,
        #             pipe_args.compute_cov3D_python,
        #             keep_alive,
        #             scaling_modifier,
        #         ) = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render_func(custom_cam, gaussians, pipe_args, background, scaling_modifier)["render"]
        #             net_image_bytes = memoryview(
        #                 (torch.clamp(net_image, min=0, max=1.0) * 255)
        #                 .byte()
        #                 .permute(1, 2, 0)
        #                 .contiguous()
        #                 .cpu()
        #                 .numpy()
        #             )
        #         network_gui.send(net_image_bytes, model_args.data_path)
        #         if do_training and ((iteration < int(optim_args.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        # iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.one_up_sh_degree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.get_train_cameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        bg = torch.rand((3), device="cuda") if model_args.random_background else background

        rendered_pkg = render_func(
            viewpoint_cam,
            gaussians,
            pipe_args,
            bg,
            GRsetting=GRsetting,
            GRzer=GRzer,
        )

        image = rendered_pkg["render"]
        viewspace_point_tensor = rendered_pkg["viewspace_points"]
        visibility_filter = rendered_pkg["visibility_filter"]
        radii = rendered_pkg["radii"]
        # print(f"iter: {iteration}")
        # print(f"image: {image.shape} {image.min()} {image.max()}")
        # print(f"viewspace_point_tensor: {viewspace_point_tensor.shape} {viewspace_point_tensor.min()} {viewspace_point_tensor.max()}")
        # print(f"visibility_filter: {visibility_filter.shape} {visibility_filter.min()} {visibility_filter.max()}")
        # print(f"radii: {radii.shape} {radii.min()} {radii.max()}")

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if iteration % 1000 == 0:
            save_image(
                image,
                os.path.join(rendering_folder, f"render_{viewpoint_cam.image_name}_{iteration:05d}.png"),
            )
            # save_image(
            #     gt_image,
            #     os.path.join(rendering_folder, f"gt_{viewpoint_cam.image_name}_{iteration:05d}.png"),
            # )
            # current_xyz = gaussians.get_xyz
            # xyz_min = torch.min(current_xyz, dim=0).values
            # xyz_max = torch.max(current_xyz, dim=0).values
            # print(f"Iter {iteration} min: {xyz_min}, max: {xyz_max}")

        l1_value = l1_loss(image, gt_image)
        ssim_value = 1.0 - ssim(image, gt_image)
        loss = 0.0
        loss += (1.0 - optim_args.lambda_dssim) * l1_value
        loss += optim_args.lambda_dssim * ssim_value

        if optim_args.lambda_reg_scaling > 0:
            scaling = gaussians.get_scaling
            scaling_max = torch.max(scaling, dim=1).values
            scaling_min = torch.min(scaling, dim=1).values
            scale_ratio_threshold = optim_args.scaling_reg_ratio_threshold
            scaling_reg = torch.max(scaling_max / scaling_min - scale_ratio_threshold, torch.zeros_like(scaling_min))
            scaling_reg = scaling_reg.mean()
            loss += optim_args.lambda_reg_scaling * scaling_reg

        loss.backward()

        # iter_end.record()
        # torch.cuda.synchronize()

        with torch.no_grad():
            # elapsed = iter_start.elapsed_time(iter_end)
            if tb_writer:
                tb_writer.add_scalar("train_loss/l1_loss", l1_value.item(), iteration)
                tb_writer.add_scalar("train_loss/total_loss", loss.item(), iteration)
                # tb_writer.add_scalar("iter_time", elapsed, iteration)

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            pbar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}", "Points": gaussians.get_xyz.shape[0]})

            # Log and save
            training_report(
                tb_writer,
                iteration,
                args.test_iterations,
                scene,
                render_func,
                rendering_folder,
                pipe_args,
                background,
                GRsetting,
                GRzer,
            )
            if iteration in args.save_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # Densification
            if iteration < optim_args.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > optim_args.densify_from_iter and iteration % optim_args.densification_interval == 0:
                    size_threshold = 20 if iteration > optim_args.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        optim_args.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % optim_args.opacity_reset_interval == 0 or (
                    model_args.white_background and iteration == optim_args.densify_from_iter
                ):
                    gaussians.reset_opacity()

            if optim_args.prune_near_interval > 0 and iteration % optim_args.prune_near_interval == 0:
                gaussians.prune_near_points(optim_args.prune_near_with_object)

            if optim_args.prune_near_cam_interval > 0 and iteration % optim_args.prune_near_cam_interval == 0:
                gaussians.prune_near_cam_points()

            if optim_args.prune_large_interval > 0 and iteration % optim_args.prune_large_interval == 0:
                gaussians.prune_large_points()

            ## bbox pruning not working, cause optimizer state to None, directly use the range to initialize points
            # gaussians.prune_points_bbox(scene.bbox_model)

            # Optimizer step
            if iteration < optim_args.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in args.checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + f"/ckpt_{iteration}.pth")


def training_report(
    tb_writer: SummaryWriter,
    iteration: int,
    test_iterations,
    scene: Scene,
    render_func: callable,
    rendering_folder: str,
    pipe_args: PipelineParams,
    background: torch.Tensor,
    GRsetting: dict,
    GRzer: dict,
):
    # Report test and samples of training set
    if iteration in test_iterations:

        validation_configs = (
            {"name": "test", "cameras": scene.get_test_cameras()},
            {"name": "train", "cameras": scene.get_train_cameras()},
            # {
            #     "name": "train",
            #     "cameras": [scene.get_train_cameras()[idx % len(scene.get_train_cameras())] for idx in range(5, 30, 5)],
            # },
        )

        for config in validation_configs:
            if not config["cameras"] or len(config["cameras"]) == 0:
                continue
            l1_test = 0.0
            psnr_test = 0.0
            for idx, viewpoint in enumerate(config["cameras"]):
                rendered = render_func(
                    viewpoint,
                    scene.gaussians,
                    pipe_args,
                    background,
                    override_color=None,
                    GRsetting=GRsetting,
                    GRzer=GRzer,
                )
                image = torch.clamp(rendered["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                render_img_path = os.path.join(
                    rendering_folder, f"test_render_{viewpoint.image_name}_{iteration:05d}.png"
                )
                # gt_img_path = os.path.join(rendering_folder, f"test_gt_{viewpoint.image_name}_{iteration:05d}.png")
                save_image(image, render_img_path)
                # save_image(gt_image, gt_img_path)

                if tb_writer and (idx < 5):
                    tb_writer.add_images(
                        config["name"] + f"_view_{viewpoint.image_name}/render",
                        image[None],
                        global_step=iteration,
                    )
                    if iteration == test_iterations[0]:
                        tb_writer.add_images(
                            config["name"] + f"_view_{viewpoint.image_name}/ground_truth",
                            gt_image[None],
                            global_step=iteration,
                        )
                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
            psnr_test /= len(config["cameras"])
            l1_test /= len(config["cameras"])
            print(f"[ITER {iteration}] Evaluating {config["name"]}: L1 {l1_test} PSNR {psnr_test}")
            if tb_writer:
                tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)


if __name__ == "__main__":
    lt.monkey_patch()
    args, mp_extract, op_extract, pp_extract = get_parser()
    train(args, mp_extract, op_extract, pp_extract)

    # All done
    print("Training complete.")
