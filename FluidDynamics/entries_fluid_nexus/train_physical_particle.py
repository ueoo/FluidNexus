import os
import random
import sys

from argparse import Namespace

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
from utils.loss_utils import distance_loss, l1_loss, l2_loss, ssim


def train(args: Namespace, model_args: ModelParams, optim_args: OptimizationParams, pipe_args: PipelineParams):

    write_args_to_file(args, model_args, optim_args, pipe_args, "training")

    tb_writer = prepare_output_and_logger(model_args)
    render_func, GRsetting, GRzer = get_render_pipe(pipe_args.rd_pipe)

    print(f"Model: {model_args.model}")
    Gaussian = get_model(model_args.model)

    gaussians: GaussianModel = Gaussian(model_args.sh_degree)

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

    print("Start training")
    print(f"Num of unique training timestamps: {len(unique_timestamps)}")

    checkpoint_path = os.path.join(scene.model_path, "checkpoint")
    quantities_path = os.path.join(scene.model_path, "quantities")
    quantities_sim_path = os.path.join(scene.model_path, "quantities_sim")
    quantities_optim_path = os.path.join(scene.model_path, "quantities_optim")

    ##########################################################################################
    ####################### First frame, optimize visual particles' xyz ######################
    ##########################################################################################
    gaussians.create_particles_visual(model_args)
    gaussians.prepare_visual_particles_for_rendering()

    cur_time_index = 0

    cur_viewpoint_set = train_cam_dict[cur_time_index]
    cur_test_viewpoint_set = test_cam_dict[cur_time_index]
    current_time_iterations = optim_args.iterations_per_time_first

    gaussians.save_particles_optimization_first(quantities_optim_path, cur_time_index, 0)

    # Use 1 based index for saving and testing
    # testing_iterations = [1, current_time_iterations // 2, current_time_iterations]
    testing_iterations = [1, current_time_iterations]
    # testing_iterations = [current_time_iterations]

    gaussians.training_setup_first_visual(optim_args)

    desc_str = f"Optimizing first frame visual"
    postfix = {"Visual": gaussians._visual_xyz.shape[0]}

    for itr in trange(1, current_time_iterations + 1, desc=desc_str, postfix=postfix, leave=True):
        gaussians.total_iterations += 1

        gaussians.update_learning_rate_first_visual(itr)

        gaussians.zero_gradient_cache_first_visual()

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

            gt_image_gray = torch.mean(gt_image, dim=0, keepdim=True)
            gt_image = torch.cat([gt_image_gray] * 3, dim=0)

            image_gray = torch.mean(image, dim=0, keepdim=True)
            image = torch.cat([image_gray] * 3, dim=0)

            view_name = viewpoint_cam.image_name

            l1_value = l1_loss(image, gt_image)
            ssim_value = 1.0 - ssim(image, gt_image)

            if optim_args.lambda_first_distance > 0:
                dist_value = distance_loss(gaussians.get_visual_xyz, optim_args.distance_threshold_visual)
            else:
                dist_value = torch.tensor(0.0, device="cuda")

            weight_loss = 0.0
            weight_loss += (1.0 - optim_args.lambda_dssim) * l1_value
            weight_loss += optim_args.lambda_dssim * ssim_value
            weight_loss += optim_args.lambda_first_distance * dist_value
            loss = weight_loss

            t_idx = cur_time_index
            loss_prefix_str = f"train_loss_frame_{t_idx:03d}"
            tb_writer.add_scalar(f"{loss_prefix_str}/l1_{view_name}", l1_value.item(), itr)
            tb_writer.add_scalar(f"{loss_prefix_str}/ssim_{view_name}", ssim_value.item(), itr)
            tb_writer.add_scalar(f"{loss_prefix_str}/dist_{view_name}", dist_value.item(), itr)
            tb_writer.add_scalar(f"{loss_prefix_str}/total_{view_name}", loss.item(), itr)

            loss.backward()
            gaussians.cache_gradient_first_visual()
            gaussians.optimizer.zero_grad()

        gaussians.set_batch_gradient_first_visual(optim_args.batch)

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad()

        if itr % 10 == 0:
            gaussians.save_particles_optimization_first(quantities_optim_path, cur_time_index, itr)

        if itr in testing_iterations:
            training_report(
                cur_time_index,
                cur_viewpoint_set,
                cur_test_viewpoint_set,
                tb_writer,
                gaussians.total_iterations,
                scene,
                render_func,
                pipe_args,
                background,
                GRsetting,
                GRzer,
                scale=False,
                save_gt=itr == testing_iterations[0],
                verbose=itr == testing_iterations[-1],
            )

    ####################################################################################################
    ####################### First frame, initialize hidden particles' xyz, velocity ####################
    ####################################################################################################

    gaussians.detach_visual_and_scale()

    gaussians.create_particles_hidden(model_args)

    num_hidden = gaussians._xyz.shape[0]
    num_visual = gaussians._visual_xyz.shape[0]
    tb_writer.add_scalar("num_particles/hidden", num_hidden, gaussians.total_sim_iterations)
    tb_writer.add_scalar("num_particles/visual", num_visual, gaussians.total_sim_iterations)
    tb_writer.add_scalar("num_particles/total", num_hidden + num_visual, gaussians.total_sim_iterations)

    gaussians.save_particles_simulation(quantities_sim_path, gaussians.total_sim_iterations)
    gaussians.total_sim_iterations += 1

    # In stable iterations, we don't update the visual particles
    for stable_iter in trange(optim_args.stable_iterations, desc="Stabilizing first frame", leave=True):
        gaussians.remove_invalid_particles()
        gaussians.guess_hidden_particles(stable=True)
        gaussians.save_particles_simulation_guess(quantities_sim_path, gaussians.total_sim_iterations)
        for _ in range(optim_args.solver_iterations):
            gaussians.update_solver_counts()
        for _ in range(optim_args.solver_iterations):
            ret_values = gaussians.project_gas_constraints()
            for ret_k, ret_v in ret_values.items():
                tb_writer.add_scalar(f"sim_stable/{ret_k}", ret_v, gaussians.total_tb_log_iterations)
            gaussians.total_tb_log_iterations += 1

        gaussians.confirm_guess_hidden_particles()

        num_hidden = gaussians._xyz.shape[0]
        num_visual = gaussians._visual_xyz.shape[0]
        tb_writer.add_scalar("num_particles/hidden", num_hidden, gaussians.total_sim_iterations)
        tb_writer.add_scalar("num_particles/visual", num_visual, gaussians.total_sim_iterations)
        tb_writer.add_scalar("num_particles/total", num_hidden + num_visual, gaussians.total_sim_iterations)

        gaussians.save_particles_simulation(quantities_sim_path, gaussians.total_sim_iterations)
        gaussians.total_sim_iterations += 1

    gaussians.save_particles_frame(quantities_path, 0)
    gaussians.save_all(checkpoint_path, 0)

    ####################################################################################################
    ####################### Current frame, simulation and fitting ######################################
    ####################################################################################################
    ## the first hidden stage is skipped, as the theory is not clear
    ## currently, we have the fitted visual_xyz, and initialized hidden xyz

    gaussians.prepare_emitter_points(model_args)

    simulation_ratio = optim_args.simulation_ratio
    assert simulation_ratio == 1
    wind_since = optim_args.wind_since
    data_2_since = model_args.data_2_since

    desc_str = "Simulating and optimizing current frame"
    for cur_time_index in trange(1, len(train_cam_dict), desc=desc_str, leave=True):
        # simulate the whole tick hidden/visual particles for simulation_ratio - 1 times
        # if simulation_ratio > 1:
        #     for sim_iter in trange(simulation_ratio - 1, desc=f"Simulating frame {cur_time_index}", leave=False):
        #         gaussians.remove_invalid_particles()
        #         gaussians.guess_hidden_particles()
        #         gaussians.save_particles_simulation_guess(quantities_sim_path, gaussians.total_sim_iterations)
        #         for _ in range(optim_args.solver_iterations):
        #             gaussians.update_solver_counts()
        #         for i in range(optim_args.solver_iterations):
        #             ret_values = gaussians.project_gas_constraints()
        #             for ret_k, ret_v in ret_values.items():
        #                 tb_writer.add_scalar(
        #                     f"sim_frame_{cur_time_index:03d}/{ret_k}", ret_v, gaussians.total_tb_log_iterations
        #                 )
        #             gaussians.total_tb_log_iterations += 1

        #         gaussians.confirm_guess_hidden_particles()
        #         gaussians.emit_new_particles()
        #         gaussians.update_visual_particles()

        #         num_hidden = gaussians._xyz.shape[0]
        #         num_visual = gaussians._visual_xyz.shape[0]
        #         tb_writer.add_scalar("num_particles/hidden", num_hidden, gaussians.total_sim_iterations)
        #         tb_writer.add_scalar("num_particles/visual", num_visual, gaussians.total_sim_iterations)
        #         tb_writer.add_scalar("num_particles/total", num_hidden + num_visual, gaussians.total_sim_iterations)

        #         gaussians.save_particles_simulation(quantities_sim_path, gaussians.total_sim_iterations)
        #         gaussians.total_sim_iterations += 1

        if data_2_since >= 0 and cur_time_index == data_2_since:
            gs_load_ply_path_2 = os.path.join(
                model_args.bg_2_load_path,
                "point_cloud",
                f"iteration_{model_args.bg_load_iteration:05d}",
                "point_cloud.ply",
            )
            gaussians.load_ply(gs_load_ply_path_2)

        gaussians.remove_invalid_particles()
        use_wind = wind_since >= 0 and cur_time_index >= wind_since

        gaussians.emit_new_particles()

        gaussians.guess_hidden_particles(use_wind=use_wind)

        gaussians.save_particles_simulation_guess(quantities_sim_path, gaussians.total_sim_iterations)

        for _ in range(optim_args.solver_iterations):
            gaussians.update_solver_counts()
        for _ in range(optim_args.solver_iterations):
            ret_values = gaussians.project_gas_constraints()
            for k, v in ret_values.items():
                tb_writer.add_scalar(f"sim_frame_{cur_time_index:03d}/{k}", v, gaussians.total_tb_log_iterations)
            gaussians.total_tb_log_iterations += 1

        # setup the visual particles for current frame
        gaussians.training_setup_current(optim_args)
        gaussians.prepare_visual_particles_for_rendering()

        cur_viewpoint_set = train_cam_dict[cur_time_index]
        cur_test_viewpoint_set = test_cam_dict[cur_time_index]
        iters_min = optim_args.iterations_per_time_current
        iters_max = optim_args.iterations_per_time_current_max
        current_time_iterations = iters_min + (iters_max - iters_min) * cur_time_index / len(train_cam_dict)
        current_time_iterations = int(current_time_iterations)

        if optim_args.sparse_views_from_time_index > 0 and cur_time_index >= optim_args.sparse_views_from_time_index:
            # sparse views
            sparse_viewpoint_set = []
            for viewpoint in cur_viewpoint_set:
                if viewpoint.image_name in optim_args.sparse_views:
                    sparse_viewpoint_set.append(viewpoint)
            cur_viewpoint_set = sparse_viewpoint_set
            current_time_iterations = optim_args.iterations_per_time_current_sparse

        # testing_iterations = [1, current_time_iterations // 2, current_time_iterations]
        testing_iterations = [1, current_time_iterations]
        # testing_iterations = [current_time_iterations]

        # Here we save the visual particles for the current frame before optimization
        gaussians.save_particles_optimization(quantities_optim_path, gaussians.get_visual_xyz, cur_time_index, 0)

        desc_str = f"Optimizing frame {cur_time_index}"
        postfix = {"Hidden": gaussians._xyz.shape[0], "Visual": gaussians._visual_xyz.shape[0]}
        for itr in (pbar := trange(1, current_time_iterations + 1, desc=desc_str, postfix=postfix, leave=False)):
            gaussians.total_iterations += 1

            gaussians.update_learning_rate_current(itr)

            gaussians.zero_gradient_cache_current()

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
                    pos_type="guess_visual_nn",
                    scale=True,
                )
                image = render_pkg["render"]
                visual_xyz = render_pkg["render_xyz"]

                gt_image = viewpoint_cam.original_image.float().cuda()
                view_name = viewpoint_cam.image_name

                gt_image_gray = torch.mean(gt_image, dim=0, keepdim=True)
                gt_image = torch.cat([gt_image_gray] * 3, dim=0)

                image_gray = torch.mean(image, dim=0, keepdim=True)
                image = torch.cat([image_gray] * 3, dim=0)

                l1_value = l1_loss(image, gt_image)
                ssim_value = 1.0 - ssim(image, gt_image)

                if optim_args.lambda_current_distance > 0:
                    dist_value = distance_loss(visual_xyz, optim_args.distance_threshold_visual)
                else:
                    dist_value = torch.tensor(0.0, device="cuda")

                if optim_args.lambda_exyz > 0:
                    fake_estimated_xyz = gaussians._estimate_xyz_nn * gaussians.scale_factor
                    exyz_loss_value = l2_loss(fake_estimated_xyz, gaussians._estimate_xyz)
                else:
                    fake_estimated_xyz = torch.tensor(0.0, device="cuda")
                    exyz_loss_value = torch.tensor(0.0, device="cuda")

                if optim_args.lambda_gas_constraints > 0:
                    gas_cs_p_ratio = gaussians.get_gas_constraints_from_exyz_nn()
                    gt_value = torch.ones_like(gas_cs_p_ratio)
                    gas_cs_loss_value = l2_loss(gas_cs_p_ratio, gt_value)
                else:
                    gas_cs_p_ratio = torch.tensor(0.0, device="cuda")
                    gt_value = torch.tensor(0.0, device="cuda")
                    gas_cs_loss_value = torch.tensor(0.0, device="cuda")

                if optim_args.lambda_next_gas_constraints > 0:
                    next_gas_cs_p_ratio = gaussians.get_gas_constraints_from_vel_nn_guess()
                    next_gt_value = torch.ones_like(next_gas_cs_p_ratio)
                    next_gas_cs_loss_value = l2_loss(next_gas_cs_p_ratio, next_gt_value)
                else:
                    next_gas_cs_p_ratio = torch.tensor(0.0, device="cuda")
                    next_gt_value = torch.tensor(0.0, device="cuda")
                    next_gas_cs_loss_value = torch.tensor(0.0, device="cuda")

                weight_loss = torch.tensor(0.0, device="cuda")
                weight_loss = weight_loss + (1.0 - optim_args.lambda_dssim) * l1_value * optim_args.lambda_image
                weight_loss = weight_loss + optim_args.lambda_dssim * ssim_value * optim_args.lambda_image

                weight_loss = weight_loss + optim_args.lambda_current_distance * dist_value

                weight_loss = weight_loss + optim_args.lambda_exyz * exyz_loss_value

                weight_loss = weight_loss + optim_args.lambda_gas_constraints * gas_cs_loss_value
                weight_loss = weight_loss + optim_args.lambda_next_gas_constraints * next_gas_cs_loss_value

                loss = weight_loss

                t_idx = cur_time_index
                loss_prefix_str = f"train_loss_frame_{t_idx:03d}"
                tb_writer.add_scalar(f"{loss_prefix_str}/l1_{view_name}", l1_value.item(), itr)
                tb_writer.add_scalar(f"{loss_prefix_str}/ssim_{view_name}", ssim_value.item(), itr)

                tb_writer.add_scalar(f"{loss_prefix_str}/dist_{view_name}", dist_value.item(), itr)

                tb_writer.add_scalar(f"{loss_prefix_str}/exyz_{view_name}", exyz_loss_value.item(), itr)

                tb_writer.add_scalar(f"{loss_prefix_str}/gas_cs_{view_name}", gas_cs_loss_value.item(), itr)
                tb_writer.add_scalar(f"{loss_prefix_str}/next_gas_cs_{view_name}", next_gas_cs_loss_value.item(), itr)

                tb_writer.add_scalar(f"{loss_prefix_str}/total_{view_name}", loss.item(), itr)

                optim_prefix_str = f"optim_frame_{t_idx:03d}"
                tb_writer.add_scalar(f"{optim_prefix_str}/gas_cs_p_ratio", gas_cs_p_ratio.mean().item(), itr)
                tb_writer.add_scalar(f"{optim_prefix_str}/next_gas_cs_p_ratio", next_gas_cs_p_ratio.mean().item(), itr)

                loss.backward()
                gaussians.cache_gradient_current()
                gaussians.optimizer.zero_grad()

            gaussians.set_batch_gradient_current(optim_args.batch)
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad()

            if itr % 10 == 0:
                gaussians.save_particles_optimization(quantities_optim_path, visual_xyz, cur_time_index, itr)

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
                    pos_type="guess_visual_nn",
                    save_gt=itr == testing_iterations[0],
                    scale=True,
                    verbose=itr == testing_iterations[-1],
                )

        gaussians.confirm_guess_hidden_particles_from_nn()
        gaussians.update_visual_xyz_from_nn()
        gaussians.confirm_guess_hidden_particles_wo_velocity()

        num_hidden = gaussians._xyz.shape[0]
        num_visual = gaussians._visual_xyz.shape[0]
        tb_writer.add_scalar("num_particles/hidden", num_hidden, gaussians.total_sim_iterations)
        tb_writer.add_scalar("num_particles/visual", num_visual, gaussians.total_sim_iterations)
        tb_writer.add_scalar("num_particles/total", num_hidden + num_visual, gaussians.total_sim_iterations)

        gaussians.save_particles_simulation(quantities_sim_path, gaussians.total_sim_iterations)
        gaussians.save_particles_frame(quantities_path, cur_time_index)
        gaussians.save_all(checkpoint_path, cur_time_index)
        gaussians.total_sim_iterations += 1

    ####################################################################################################
    ####################### Future prediction frame, simulation ######################################
    ####################################################################################################

    gaussians._estimate_xyz_nn_grad = None
    gaussians._estimate_xyz_nn = None
    gaussians._velocity_nn_grad = None
    gaussians._velocity_nn = None
    gaussians.optimizer = None

    cur_time_index += 1
    future_pred_frames = optim_args.future_pred_frames
    if future_pred_frames <= 0:
        print("No future prediction frames")
        return

    decay_frames_future_p0 = optim_args.decay_frames_future_p0
    p0_recon = gaussians.p0
    p0_future = optim_args.p0_future
    # reached_max_hidden_particles = False

    for future_time_index in (pbar := trange(future_pred_frames, desc=f"Predicting future frames", leave=True)):
        future_frame_index = cur_time_index + future_time_index
        gaussians.p0 = p0_future + (p0_recon - p0_future) * (1 - min(1, future_time_index / decay_frames_future_p0))
        gaussians.remove_invalid_particles()
        gaussians.emit_new_particles(future_time_index)
        gaussians.guess_hidden_particles()
        gaussians.save_particles_simulation_guess(quantities_sim_path, gaussians.total_sim_iterations)
        for _ in range(optim_args.solver_iterations_future):
            gaussians.update_solver_counts()
        for i in range(optim_args.solver_iterations_future):
            ret_values = gaussians.project_gas_constraints()
            for k, v in ret_values.items():
                tb_writer.add_scalar(f"fut_frame_{future_frame_index:03d}/{k}", v, gaussians.total_tb_log_iterations)
            gaussians.total_tb_log_iterations += 1

        gaussians.confirm_guess_hidden_particles()
        gaussians.update_visual_particles()

        tb_writer.add_scalar("p0", gaussians.p0, gaussians.total_sim_iterations)
        num_hidden = gaussians._xyz.shape[0]
        num_visual = gaussians._visual_xyz.shape[0]
        tb_writer.add_scalar("num_particles/hidden", num_hidden, gaussians.total_sim_iterations)
        tb_writer.add_scalar("num_particles/visual", num_visual, gaussians.total_sim_iterations)
        tb_writer.add_scalar("num_particles/total", num_hidden + num_visual, gaussians.total_sim_iterations)

        gaussians.prepare_visual_particles_for_rendering()

        # we just use the first camera for future prediction
        # we append _0000 to filename to make it compatible with frames to video
        time_index = 0
        viewpoint_set = train_cam_dict[time_index]
        for viewpoint_cam in viewpoint_set:
            render_pkg = render_func(
                viewpoint_cam,
                gaussians,
                pipe_args,
                background,
                GRsetting=GRsetting,
                GRzer=GRzer,
                pos_type="visual",
                scale=True,
            )

            image = render_pkg["render"]
            save_image(
                image,
                os.path.join(
                    scene.model_path,
                    "training_render",
                    f"render_frame{future_frame_index:03d}_{viewpoint_cam.image_name}_{0:08d}.png",
                ),
            )

        test_viewpoint_set = test_cam_dict[time_index]
        for viewpoint_cam in test_viewpoint_set:
            render_pkg = render_func(
                viewpoint_cam,
                gaussians,
                pipe_args,
                background,
                GRsetting=GRsetting,
                GRzer=GRzer,
                pos_type="visual",
                scale=True,
            )

            image = render_pkg["render"]
            save_image(
                image,
                os.path.join(
                    scene.model_path,
                    "training_render",
                    f"render_frame{future_frame_index:03d}_{viewpoint_cam.image_name}_{0:08d}.png",
                ),
            )

        gaussians.save_particles_simulation(quantities_sim_path, gaussians.total_sim_iterations)
        gaussians.save_particles_frame(quantities_path, future_frame_index)
        gaussians.save_all(checkpoint_path, future_frame_index)
        gaussians.total_sim_iterations += 1

        post_fix = {}
        post_fix["Hidden"] = gaussians._xyz.shape[0]
        post_fix["Visual"] = gaussians._visual_xyz.shape[0]
        post_fix["Total"] = post_fix["Hidden"] + post_fix["Visual"]
        pbar.set_postfix(post_fix)

        # if not reached_max_hidden_particles and gaussians._xyz.shape[0] > optim_args.max_hidden_particles:
        #     reached_max_hidden_particles = True
        #     msg = f"Reached max hidden particles {optim_args.max_hidden_particles}, will not add more hidden particles"
        #     print(msg)
        # if not reached_max_hidden_particles:
        #     gaussians.emit_new_particles(future_time_index)


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
    verbose=False,
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
            rendered_gpf_only = render_func(
                viewpoint,
                scene.gaussians,
                pipe_args,
                background,
                override_color=None,
                GRsetting=GRsetting,
                GRzer=GRzer,
                pos_type=pos_type,
                scale=scale,
                gpf_only=True,
            )
            # rendered_gs_only = render_func(
            #     viewpoint,
            #     scene.gaussians,
            #     pipe_args,
            #     background,
            #     override_color=None,
            #     GRsetting=GRsetting,
            #     GRzer=GRzer,
            #     pos_type=pos_type,
            #     scale=scale,
            #     gs_only=True,
            # )

            image = torch.clamp(rendered["render"], 0.0, 1.0)
            image_gpf_only = torch.clamp(rendered_gpf_only["render"], 0.0, 1.0)
            # image_gs_only = torch.clamp(rendered_gs_only["render"], 0.0, 1.0)
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
            save_image(
                image_gpf_only,
                os.path.join(
                    scene.model_path,
                    "training_render",
                    f"render_gpf_frame{cur_time_index:03d}_{viewpoint.image_name}_{cur_iteration:08d}.png",
                ),
            )
            # save_image(
            #     image_gs_only,
            #     os.path.join(
            #         scene.model_path,
            #         "training_render",
            #         f"render_gs_frame{cur_time_index:03d}_{viewpoint.image_name}_{cur_iteration:08d}.png",
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
            tb_writer.add_images(
                f"frame_gpf_{cur_time_index:03d}_view_{viewpoint.image_name}/render",
                image_gpf_only[None],
                global_step=cur_iteration,
            )
            # tb_writer.add_images(
            #     f"frame_gs_{cur_time_index:03d}_view_{viewpoint.image_name}/render",
            #     image_gs_only[None],
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
    # lt.monkey_patch()
    args, mp_extract, op_extract, pp_extract = get_parser()
    train(args, mp_extract, op_extract, pp_extract)

    # All done
    print("Training complete.")
