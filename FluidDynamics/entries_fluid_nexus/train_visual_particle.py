import os
import random
import sys

from argparse import Namespace

import lovely_tensors as lt
import torch

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import trange


sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_splatting.gm_dynamics import GaussianModel
from helpers.helper_gaussian import get_model
from helpers.helper_parser import get_parser, write_args_to_file
from helpers.helper_pipe import get_render_pipe
from helpers.helper_train import prepare_output_and_logger
from scene import Scene
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, l2_loss_consistency, ssim


def train(args: Namespace, model_args: ModelParams, optim_args: OptimizationParams, pipe_args: PipelineParams):

    write_args_to_file(args, model_args, optim_args, pipe_args, "training_level_two")

    tb_writer = prepare_output_and_logger(model_args)
    render_func, GRsetting, GRzer = get_render_pipe(pipe_args.rd_pipe)

    print(f"Model: {model_args.model}")
    Gaussian = get_model(model_args.model)

    gaussians: GaussianModel = Gaussian()

    scene = Scene(model_args, gaussians, loader=model_args.loader)

    num_channel = 3  # this is the render channel

    bg_color = 1 if model_args.white_background else 0
    bg_color = [bg_color] * num_channel
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_camera_list = scene.get_train_cameras().copy()
    train_cam_dict = {}
    unique_timestamps = sorted(list(set([cam.timestamp for cam in train_camera_list])))

    for i, timestamp in enumerate(unique_timestamps):
        train_cam_dict[i] = [cam for cam in train_camera_list if cam.timestamp == timestamp]

    test_camera_list = scene.get_test_cameras().copy()
    test_cam_dict = {}
    unique_timestamps = sorted(list(set([cam.timestamp for cam in test_camera_list])))

    for i, timestamp in enumerate(unique_timestamps):
        test_cam_dict[i] = [cam for cam in test_camera_list if cam.timestamp == timestamp]

    gaussians.setup_constants(optim_args)

    gs_load_ply_path = os.path.join(
        model_args.bg_load_path,
        "point_cloud",
        f"iteration_{model_args.bg_load_iteration:05d}",
        "point_cloud.ply",
    )
    gaussians.load_ply(gs_load_ply_path)

    print("Start training level two")
    print(f"Num of unique training timestamps: {len(unique_timestamps)}")

    total_iterations = 0

    checkpoint_load_path = os.path.join(scene.model_path, "checkpoint")
    if args.load_path != "":
        checkpoint_load_path = os.path.join(args.load_path, "checkpoint")
    if args.level_two_load_path != "":
        checkpoint_load_path = os.path.join(args.level_two_load_path, "checkpoint_level_two")
    print("checkpoint_load_path", checkpoint_load_path)
    checkpoint_path = os.path.join(scene.model_path, "checkpoint_level_two")
    quantities_optim_path = os.path.join(scene.model_path, "quantities_level_two_optim")

    ####################################################################################################
    ####################### Current frame, fitting level two ###########################################
    ####################################################################################################

    prev_color = None
    prev_opacity = None
    prev_scales = None
    prev_rotation = None
    data_2_since = model_args.data_2_since

    for cur_time_index in trange(len(train_cam_dict), desc="Fitting level two"):

        if data_2_since >= 0 and cur_time_index == data_2_since:
            gs_load_ply_path_2 = os.path.join(
                model_args.bg_2_load_path,
                "point_cloud",
                f"iteration_{model_args.bg_load_iteration:05d}",
                "point_cloud.ply",
            )
            gaussians.load_ply(gs_load_ply_path_2)

        gaussians.load_visual(
            checkpoint_load_path, cur_time_index, scale=False, color_3ch=model_args.level_two_color_3ch
        )

        gaussians.init_quantities_current_level_two(optim_args, prev_color, prev_opacity, prev_scales, prev_rotation)

        gaussians.training_setup_current_level_two(optim_args)

        cur_viewpoint_set = train_cam_dict[cur_time_index]
        cur_test_viewpoint_set = test_cam_dict[cur_time_index]
        iters_min = optim_args.iterations_per_time_current_level_two
        iters_max = optim_args.iterations_per_time_current_level_two_max
        current_time_iterations = iters_min + (iters_max - iters_min) * cur_time_index / len(train_cam_dict)
        current_time_iterations = int(current_time_iterations)
        # testing_iterations = [1, current_time_iterations // 2, current_time_iterations]
        # testing_iterations = (
        #     [1] + [itr for itr in range(100, current_time_iterations + 1, 100)] + [current_time_iterations]
        # )
        # testing_iterations = list(set(testing_iterations))
        testing_iterations = [current_time_iterations]

        # Here we save the visual particles for the current frame before optimization
        gaussians.save_particles_optimization_level_two(quantities_optim_path, cur_time_index, 0)

        desc_str = f"Optimizing level two frame {cur_time_index}"
        postfix = {"Visual": gaussians._visual_xyz.shape[0]}
        for itr in (pbar := trange(1, current_time_iterations + 1, desc=desc_str, postfix=postfix, leave=False)):
            total_iterations += 1

            gaussians.zero_gradient_cache_current_level_two()

            cam_index = random.sample(cur_viewpoint_set, optim_args.batch)
            for i in range(optim_args.batch):
                viewpoint_cam = cam_index[i]
                render_pkg = render_func(
                    viewpoint_cam,
                    gaussians,
                    pipe_args,
                    background,
                    GRsetting=GRsetting,
                    GRzer=GRzer,
                    pos_type="visual",
                )
                image = render_pkg["render"]

                gt_image = viewpoint_cam.original_image.float().cuda()
                view_name = viewpoint_cam.image_name

                l1_value = l1_loss(image, gt_image)
                ssim_value = 1.0 - ssim(image, gt_image)
                weight_loss = 0.0
                weight_loss = weight_loss + (1.0 - optim_args.lambda_dssim) * l1_value * optim_args.lambda_image
                weight_loss = weight_loss + optim_args.lambda_dssim * ssim_value * optim_args.lambda_image

                if gaussians.fit_color:
                    color_closs_value = l2_loss_consistency(
                        gaussians._visual_color, prev_color, optim_args.consistency_color_threshold
                    )
                    weight_loss = weight_loss + optim_args.lambda_consistency_color * color_closs_value

                if gaussians.fit_opacity:
                    opacity_closs_value = l2_loss_consistency(
                        gaussians._visual_opacity, prev_opacity, optim_args.consistency_opacity_threshold
                    )
                    weight_loss = weight_loss + optim_args.lambda_consistency_opacity * opacity_closs_value

                if gaussians.fit_scales:
                    scales_closs_value = l2_loss_consistency(
                        gaussians._visual_scales, prev_scales, optim_args.consistency_scales_threshold
                    )
                    weight_loss = weight_loss + optim_args.lambda_consistency_scales * scales_closs_value

                    if optim_args.lambda_reg_scaling > 0:
                        scaling = gaussians.get_visual_scaling
                        scaling_max = torch.max(scaling, dim=1).values
                        scaling_min = torch.min(scaling, dim=1).values
                        scale_ratio_threshold = optim_args.scaling_reg_ratio_threshold
                        scaling_reg = torch.max(
                            scaling_max / scaling_min - scale_ratio_threshold, torch.zeros_like(scaling_min)
                        )
                        scaling_reg = scaling_reg.mean()
                        weight_loss = weight_loss + optim_args.lambda_reg_scaling * scaling_reg

                if gaussians.fit_rotation:
                    rotation_closs_value = l2_loss_consistency(
                        gaussians._visual_rotation, prev_rotation, optim_args.consistency_rotation_threshold
                    )
                    weight_loss = weight_loss + optim_args.lambda_consistency_rotation * rotation_closs_value

                loss = weight_loss

                t_idx = cur_time_index
                loss_str = f"train_loss_frame_{t_idx:03d}"
                tb_writer.add_scalar(f"{loss_str}/l1_{view_name}", l1_value.item(), itr)
                tb_writer.add_scalar(f"{loss_str}/ssim_{view_name}", ssim_value.item(), itr)

                if gaussians.fit_color:
                    tb_writer.add_scalar(f"{loss_str}/color_cons_{view_name}", color_closs_value.item(), itr)
                if gaussians.fit_opacity:
                    tb_writer.add_scalar(f"{loss_str}/opacity_cons_{view_name}", opacity_closs_value.item(), itr)
                if gaussians.fit_scales:
                    tb_writer.add_scalar(f"{loss_str}/scales_cons_{view_name}", scales_closs_value.item(), itr)
                    if optim_args.lambda_reg_scaling > 0:
                        tb_writer.add_scalar(f"{loss_str}/scaling_reg_{view_name}", scaling_reg.item(), itr)
                if gaussians.fit_rotation:
                    tb_writer.add_scalar(f"{loss_str}/rotation_cons_{view_name}", rotation_closs_value.item(), itr)

                tb_writer.add_scalar(f"{loss_str}/total_{view_name}", loss.item(), itr)

                loss.backward()
                gaussians.cache_gradient_current_level_two()
                gaussians.optimizer.zero_grad()

            gaussians.set_batch_gradient_current_level_two(optim_args.batch)
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad()

            if itr % 10 == 0:
                gaussians.save_particles_optimization_level_two(quantities_optim_path, cur_time_index, itr)

            if itr in testing_iterations:
                training_report(
                    cur_time_index,
                    cur_viewpoint_set,
                    cur_test_viewpoint_set,
                    tb_writer,
                    itr,
                    scene,
                    render_func,
                    pipe_args,
                    background,
                    GRsetting,
                    GRzer,
                    pos_type="visual",
                    save_gt=itr == testing_iterations[0],
                )
        tb_writer.add_scalar("num_particles/visual", gaussians._visual_xyz.shape[0], cur_time_index)
        gaussians.save_visual(checkpoint_path, cur_time_index, scale=False)

        if gaussians.fit_color:
            prev_color = gaussians._visual_color.detach().clone().requires_grad_(False)
        if gaussians.fit_opacity:
            prev_opacity = gaussians._visual_opacity.detach().clone().requires_grad_(False)
        if gaussians.fit_scales:
            prev_scales = gaussians._visual_scales.detach().clone().requires_grad_(False)
        if gaussians.fit_rotation:
            prev_rotation = gaussians._visual_rotation.detach().clone().requires_grad_(False)


@torch.no_grad
def training_report(
    cur_time_index: int,
    cur_viewpoint_set: list,
    cur_test_viewpoint_set: list,
    tb_writer: SummaryWriter,
    cur_iteration: int,
    scene: Scene,
    render_func: callable,
    pipe_args: PipelineParams,
    background: torch.Tensor,
    GRsetting: dict,
    GRzer: dict,
    pos_type: str = "visual",
    save_gt=True,
    scale=False,
):

    validation_configs = (
        {"name": "test", "viewpoint_set": cur_test_viewpoint_set},
        {"name": "train", "viewpoint_set": cur_viewpoint_set},
    )

    for config in validation_configs:
        l1_test, l1_test_real = 0.0, 0.0
        psnr_test, psnr_test_real = 0.0, 0.0
        for idx, viewpoint in enumerate(config["viewpoint_set"]):
            rendered = render_func(
                viewpoint,
                scene.gaussians,
                pipe_args,
                background,
                override_color=None,
                GRsetting=GRsetting,
                GRzer=GRzer,
                pos_type=pos_type,
                scale=scale,
            )
            # rendered_gpf_only = render_func(
            #     viewpoint,
            #     scene.gaussians,
            #     pipe_args,
            #     background,
            #     override_color=None,
            #     GRsetting=GRsetting,
            #     GRzer=GRzer,
            #     pos_type=pos_type,
            #     scale=scale,
            #     gpf_only=True,
            # )

            image = torch.clamp(rendered["render"], 0.0, 1.0)
            # image_gpf_only = torch.clamp(rendered_gpf_only["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            real_gt_image = torch.clamp(viewpoint.original_image_real.to("cuda"), 0.0, 1.0)

            save_image(
                image,
                os.path.join(
                    scene.model_path,
                    "training_render",
                    f"render_frame{cur_time_index:03d}_{viewpoint.image_name}_{cur_iteration:08d}.png",
                ),
            )
            # save_image(
            #     image_gpf_only,
            #     os.path.join(
            #         scene.model_path,
            #         "training_render",
            #         f"render_gpf_frame{cur_time_index:03d}_{viewpoint.image_name}_{cur_iteration:08d}.png",
            #     ),
            # )
            if save_gt:
                save_image(
                    gt_image,
                    os.path.join(
                        scene.model_path,
                        "training_render",
                        f"gt_frame{cur_time_index:03d}_{viewpoint.image_name}.png",
                    ),
                )
                save_image(
                    real_gt_image,
                    os.path.join(
                        scene.model_path,
                        "training_render",
                        f"real_frame{cur_time_index:03d}_{viewpoint.image_name}.png",
                    ),
                )

            tb_writer.add_images(
                f"frame_{cur_time_index:03d}_view_{viewpoint.image_name}/render",
                image[None],
                global_step=cur_iteration,
            )
            # tb_writer.add_images(
            #     f"frame_gpf_{cur_time_index:03d}_view_{viewpoint.image_name}/render",
            #     image_gpf_only[None],
            #     global_step=cur_iteration,
            # )
            if save_gt:
                tb_writer.add_images(
                    f"frame_{cur_time_index:03d}_view_{viewpoint.image_name}/ground_truth",
                    gt_image[None],
                    global_step=cur_iteration,
                )

            l1_test += l1_loss(image, gt_image).mean()
            psnr_test += psnr(image, gt_image).mean()
            l1_test_real = l1_loss(image, real_gt_image).mean()
            psnr_test_real = psnr(image, real_gt_image).mean()

        l1_test /= len(config["viewpoint_set"])
        psnr_test /= len(config["viewpoint_set"])
        l1_test_real /= len(config["viewpoint_set"])
        psnr_test_real /= len(config["viewpoint_set"])

        # if verbose:
        #     print(f"[ITER {cur_iteration} Evaluation {config['name']}] L1: {l1_test}, PSNR: {psnr_test}")

        tb_writer.add_scalar(f"eval_{config['name']}/frame_{cur_time_index:03d} - l1", l1_test, cur_iteration)
        tb_writer.add_scalar(f"eval_{config['name']}/frame_{cur_time_index:03d} - psnr", psnr_test, cur_iteration)
        tb_writer.add_scalar(
            f"eval_{config['name']}/frame_{cur_time_index:03d} - l1_real", l1_test_real, cur_iteration
        )
        tb_writer.add_scalar(
            f"eval_{config['name']}/frame_{cur_time_index:03d} - psnr_real", psnr_test_real, cur_iteration
        )


if __name__ == "__main__":
    lt.monkey_patch()
    args, mp_extract, op_extract, pp_extract = get_parser()
    train(args, mp_extract, op_extract, pp_extract)

    # All done
    print("Training complete.")
