import os
import sys

from argparse import Namespace

import lovely_tensors as lt
import torch

from torchvision.utils import save_image
from tqdm import trange


sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_splatting.gm_fluid import GaussianModel
from helpers.helper_gaussian import get_model
from helpers.helper_parser import get_parser, write_args_to_file
from helpers.helper_pipe import get_render_pipe
from helpers.helper_train import prepare_output_and_logger
from scene import Scene


@torch.no_grad
def predict(args: Namespace, model_args: ModelParams, optim_args: OptimizationParams, pipe_args: PipelineParams):

    write_args_to_file(args, model_args, optim_args, pipe_args, "future_predicting")

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

    print("Start future prediction")
    print(f"Num of unique training timestamps: {len(unique_timestamps)}")

    total_sim_iterations = 0
    total_tb_log_iterations = 0

    assert os.path.exists(args.load_path), f"Load path {args.load_path} does not exist"
    checkpoint_load_path = os.path.join(args.load_path, "checkpoint")
    assert os.path.exists(checkpoint_load_path), f"Checkpoint load path {checkpoint_load_path} does not exist"
    visual_checkpoint_load_path = checkpoint_load_path

    if optim_args.use_level_two_in_future:
        print(f"Using level two for future prediction")
        msg = f"Level two load path {args.level_two_load_path} does not exist"
        assert os.path.exists(args.level_two_load_path), msg
        level_two_checkpoint_load_path = os.path.join(args.level_two_load_path, "checkpoint_level_two")
        msg = f"Level two checkpoint load path {level_two_checkpoint_load_path} does not exist"
        assert os.path.exists(level_two_checkpoint_load_path), msg
        visual_checkpoint_load_path = level_two_checkpoint_load_path

    checkpoint_path = os.path.join(scene.model_path, "checkpoint")
    quantities_path = os.path.join(scene.model_path, "quantities")
    quantities_sim_path = os.path.join(scene.model_path, "quantities_sim")

    cur_time_index = len(unique_timestamps) - 1

    gaussians.load_hidden(checkpoint_load_path, cur_time_index)
    if optim_args.use_level_two_smoothed_in_future:
        print(f"Using smoothed level two for future prediction")
        gaussians.load_visual_smoothed(visual_checkpoint_load_path, cur_time_index)
    else:
        gaussians.load_visual(visual_checkpoint_load_path, cur_time_index)
    gaussians.prepare_emitter_points()
    # gaussians.prepare_emitter_future_first_points()

    cur_time_index += 1
    future_pred_frames = optim_args.future_pred_frames
    if future_pred_frames <= 0:
        print("No future prediction frames")
        return

    SOLVER_ITERATIONS_FUTURE = optim_args.solver_iterations_future
    decay_frames_future_p0 = optim_args.decay_frames_future_p0
    p0_recon = gaussians.p0
    p0_future = optim_args.p0_future

    wind_since = optim_args.wind_since
    rigid_since = optim_args.rigid_since

    for future_time_index in (pbar := trange(future_pred_frames, desc=f"Predicting future frames")):
        future_frame_index = cur_time_index + future_time_index
        gaussians.p0 = p0_future + (p0_recon - p0_future) * (1 - min(1, future_time_index / decay_frames_future_p0))
        gaussians.remove_invalid_particles()

        use_wind = wind_since >= 0 and future_frame_index >= wind_since
        if rigid_since >= 0 and future_frame_index == rigid_since:
            gaussians.create_rigid_body()
            gaussians.save_particles_rigid_body(quantities_path, future_frame_index)
        gaussians.emit_new_particles()

        gaussians.guess_hidden_particles(use_wind=use_wind)

        gaussians.save_particles_simulation_guess(quantities_sim_path, total_sim_iterations)

        # for _ in range(SOLVER_ITERATIONS_FUTURE):
        #     gaussians.update_solver_counts()
        for i in range(SOLVER_ITERATIONS_FUTURE):
            if rigid_since >= 0 and future_frame_index >= rigid_since:
                rigid_ret_values = gaussians.project_rigid_body_constraints()
                for ret_k, ret_v in rigid_ret_values.items():
                    tb_writer.add_scalar(
                        f"fut_sim_rigid_{future_frame_index:03d}/{ret_k}", ret_v, total_tb_log_iterations
                    )

            ret_values = gaussians.project_gas_constraints()
            for ret_k, ret_v in ret_values.items():
                tb_writer.add_scalar(f"fut_sim_{future_frame_index:03d}/{ret_k}", ret_v, total_tb_log_iterations)
            total_tb_log_iterations += 1

        gaussians.confirm_guess_hidden_particles()

        gaussians.update_visual_particles()

        if rigid_since >= 0 and future_frame_index >= rigid_since:
            vis_rigid_ret_values = gaussians.project_rigid_body_constraints_for_visual_particles()
            for ret_k, ret_v in vis_rigid_ret_values.items():
                tb_writer.add_scalar(
                    f"fut_sim_rigid_vis_{future_frame_index:03d}/{ret_k}", ret_v, total_tb_log_iterations
                )

        tb_writer.add_scalar("p0", gaussians.p0, total_sim_iterations)
        tb_writer.add_scalar("num_hidden_particles", gaussians._xyz.shape[0], total_sim_iterations)
        tb_writer.add_scalar("num_visual_particles", gaussians._visual_xyz.shape[0], total_sim_iterations)

        gaussians.prepare_future_visual_particles_for_rendering(optim_args.use_level_two_in_future)

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
                    f"render_frame{future_frame_index:03d}_{viewpoint_cam.image_name}_0000.png",
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
                    f"render_frame{future_frame_index:03d}_{viewpoint_cam.image_name}_0000.png",
                ),
            )

        gaussians.save_particles_simulation(quantities_sim_path, total_sim_iterations)
        gaussians.save_particles_frame(quantities_path, future_frame_index)
        gaussians.save_all(checkpoint_path, future_frame_index)
        total_sim_iterations += 1

        post_fix = {}
        post_fix["Hidden"] = gaussians._xyz.shape[0]
        post_fix["Visual"] = gaussians._visual_xyz.shape[0]
        pbar.set_postfix(post_fix)


if __name__ == "__main__":
    lt.monkey_patch()
    args, mp_extract, op_extract, pp_extract = get_parser()
    predict(args, mp_extract, op_extract, pp_extract)

    # All done
    print("Training complete.")
