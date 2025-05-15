import json
import os

import cv2
import numpy as np

from PIL import Image
from tqdm import tqdm, trange
from utils.graphics_utils import focal2fov, fov2focal

from .camera_info import CameraInfo
from .functions import (
    rotate_camera_around_x_axis,
    rotate_camera_around_y_axis,
    rotate_camera_around_z_axis,
)


def read_cameras_from_transforms_real_capture(
    path,
    transforms_file,
    white_background,
    extension=".png",
    start_time=50,
    duration=50,
    time_step=1,
    max_timestamp=1.0,
    gray_image=False,
    train_views="0134",
    train_views_fake=None,
    test_views_fake=None,
    img_offset=False,
    is_bg=False,
    capture_part="black",
    use_refined_fake=False,
    refined_strength="0d26",
    gen_future_since=90,
    gen_prefixed_future="one",
    gen_future_strength="0d75",
    data_2_path="",
    data_2_since=-1,
    use_demo_cameras=False,
    is_wind=False,
    read_image=True,
    *args,
    **kwargs,
):
    msg = f"  transforms_file: {transforms_file}"
    msg += f", train_views: {train_views}"
    use_fake = False
    if train_views_fake is not None and len(train_views_fake) > 0:
        msg += f", train_views_fake: {train_views_fake}"
        use_fake = True
    if test_views_fake is not None and len(test_views_fake) > 0:
        msg += f", test_views_fake: {test_views_fake}"
        use_fake = True
    if use_fake and use_refined_fake:
        msg += f", use_refined_fake"
        msg += f", refined_strength: {refined_strength}"
    if gen_future_since >= 0:
        msg += f", gen_future_since: {gen_future_since}"
        if gen_prefixed_future == "i2v3":
            msg += f" with i2v_3samples_prefixed"
        else:
            raise ValueError(f"Unknown gen_prefixed_future: {gen_prefixed_future}")
        msg += f" with strength: {gen_future_strength}"
    if read_image:
        if is_bg:
            msg += f", read_image_bg"
        else:
            msg += f", read_image"
    if img_offset:
        msg += f" with img_offset"

    if data_2_path != "" and data_2_since >= 0:
        msg += f", data_2_path: {data_2_path}"
        msg += f", data_2_since: {data_2_since}"

    if use_demo_cameras:
        msg += f", use_demo_cameras"

    msg += f", capture: {capture_part}"

    print(msg)
    cam_infos = []
    # print(f"start_time {start_time} duration {duration} time_step {time_step}")

    with open(os.path.join(path, transforms_file)) as json_file:
        contents = json.load(json_file)

    near = float(contents["near"])
    far = float(contents["far"])

    frames = contents["frames"]
    camera_uid = 0

    desc_str = "  reading views"

    if use_demo_cameras:
        demo_camera_npy_path = os.path.join(path, "demo_cams_poses_extra.npy")
        demo_cameras_raw = np.load(demo_camera_npy_path)

        ## 2 -> 4
        demo_cameras_part1 = demo_cameras_raw[demo_cameras_raw.shape[0] // 2 :]
        ## 4 -> 0
        demo_cameras_part2 = demo_cameras_raw[::-1]
        ## 0 -> 2
        demo_cameras_part3 = demo_cameras_raw[: demo_cameras_raw.shape[0] // 2]
        demo_cameras = np.concatenate([demo_cameras_part1, demo_cameras_part2, demo_cameras_part3], axis=0)
        demo_cameras = demo_cameras[::2]
        print(f"demo_cameras.shape {demo_cameras.shape}")

        ### just 0 -> 4
        # demo_cameras = demo_cameras_raw.copy()
        # print(f"demo_cameras.shape {demo_cameras.shape}")

        demo_Rs = []
        demo_Ts = []
        for demo_c2w in demo_cameras:
            demo_c2w[:3, 1:3] *= -1
            demo_w2c = np.linalg.inv(demo_c2w)
            demo_R = np.transpose(demo_w2c[:3, :3])
            demo_T = demo_w2c[:3, 3]
            demo_Rs.append(demo_R)
            demo_Ts.append(demo_T)

    for idx, frame in tqdm(enumerate(frames), desc=desc_str, total=len(frames), leave=True):
        cam_name = frame["file_path"][-1:]  # train0x -> x used to determine with train_views

        # camera idx
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])

        if capture_part == "smoke":
            #### camera hacks, due to colmap is not perfect
            if cam_name == "0":
                theta_z = np.deg2rad(7)
                c2w = rotate_camera_around_z_axis(c2w, theta_z)
                theta_y = np.deg2rad(-7.3)
                c2w = rotate_camera_around_y_axis(c2w, theta_y)

            elif cam_name == "1":
                theta_z = np.deg2rad(4.8)
                c2w = rotate_camera_around_z_axis(c2w, theta_z)
                theta_y = np.deg2rad(-4.8)
                c2w = rotate_camera_around_y_axis(c2w, theta_y)
                theta_x = np.deg2rad(0.55)
                c2w = rotate_camera_around_x_axis(c2w, theta_x)

            elif cam_name == "2":
                theta_x = np.deg2rad(1.15)
                c2w = rotate_camera_around_x_axis(c2w, theta_x)

            elif cam_name == "3":
                theta_z = np.deg2rad(-2.2)
                c2w = rotate_camera_around_z_axis(c2w, theta_z)
                theta_y = np.deg2rad(5)
                c2w = rotate_camera_around_y_axis(c2w, theta_y)
                theta_x = np.deg2rad(0.5)
                c2w = rotate_camera_around_x_axis(c2w, theta_x)

            elif cam_name == "4":
                theta_z = np.deg2rad(-4.2)
                c2w = rotate_camera_around_z_axis(c2w, theta_z)
                theta_y = np.deg2rad(8)
                c2w = rotate_camera_around_y_axis(c2w, theta_y)

        elif capture_part == "ball":
            #### camera hacks, due to colmap is not perfect
            if cam_name == "0":
                theta_z = np.deg2rad(7)
                c2w = rotate_camera_around_z_axis(c2w, theta_z)
                theta_y = np.deg2rad(-7.3)
                c2w = rotate_camera_around_y_axis(c2w, theta_y)

            elif cam_name == "1":
                theta_z = np.deg2rad(4.8)
                c2w = rotate_camera_around_z_axis(c2w, theta_z)
                theta_y = np.deg2rad(-4.8)
                c2w = rotate_camera_around_y_axis(c2w, theta_y)

            elif cam_name == "2":
                theta_z = np.deg2rad(2)
                c2w = rotate_camera_around_z_axis(c2w, theta_z)
                theta_y = np.deg2rad(0.4)
                c2w = rotate_camera_around_y_axis(c2w, theta_y)

            elif cam_name == "3":
                theta_z = np.deg2rad(-2.1)
                c2w = rotate_camera_around_z_axis(c2w, theta_z)
                theta_y = np.deg2rad(4.8)
                c2w = rotate_camera_around_y_axis(c2w, theta_y)

            elif cam_name == "4":
                theta_z = np.deg2rad(-5.5)
                c2w = rotate_camera_around_z_axis(c2w, theta_z)
                theta_y = np.deg2rad(7.3)
                c2w = rotate_camera_around_y_axis(c2w, theta_y)

        elif capture_part == "smoke_and_ball_object":
            #### camera hacks, due to colmap is not perfect
            if cam_name == "0":
                theta_z = np.deg2rad(7)
                c2w_1 = rotate_camera_around_z_axis(c2w.copy(), theta_z)
                theta_y = np.deg2rad(-7.3)
                c2w_1 = rotate_camera_around_y_axis(c2w_1, theta_y)

                theta_z = np.deg2rad(7)
                c2w_2 = rotate_camera_around_z_axis(c2w.copy(), theta_z)
                theta_y = np.deg2rad(-7.3)
                c2w_2 = rotate_camera_around_y_axis(c2w_2, theta_y)

            elif cam_name == "1":
                theta_z = np.deg2rad(4.8)
                c2w_1 = rotate_camera_around_z_axis(c2w.copy(), theta_z)
                theta_y = np.deg2rad(-4.8)
                c2w_1 = rotate_camera_around_y_axis(c2w_1, theta_y)
                theta_x = np.deg2rad(0.55)
                c2w_1 = rotate_camera_around_x_axis(c2w_1, theta_x)

                theta_z = np.deg2rad(4.8)
                c2w_2 = rotate_camera_around_z_axis(c2w.copy(), theta_z)
                theta_y = np.deg2rad(-4.8)
                c2w_2 = rotate_camera_around_y_axis(c2w_2, theta_y)

            elif cam_name == "2":
                theta_x = np.deg2rad(1.15)
                c2w_1 = rotate_camera_around_x_axis(c2w.copy(), theta_x)

                theta_z = np.deg2rad(2)
                c2w_2 = rotate_camera_around_z_axis(c2w.copy(), theta_z)
                theta_y = np.deg2rad(0.4)
                c2w_2 = rotate_camera_around_y_axis(c2w_2, theta_y)

            elif cam_name == "3":
                theta_z = np.deg2rad(-2.2)
                c2w_1 = rotate_camera_around_z_axis(c2w.copy(), theta_z)
                theta_y = np.deg2rad(5)
                c2w_1 = rotate_camera_around_y_axis(c2w_1, theta_y)
                theta_x = np.deg2rad(0.5)
                c2w_1 = rotate_camera_around_x_axis(c2w_1, theta_x)

                theta_z = np.deg2rad(-2.1)
                c2w_2 = rotate_camera_around_z_axis(c2w.copy(), theta_z)
                theta_y = np.deg2rad(4.8)
                c2w_2 = rotate_camera_around_y_axis(c2w_2, theta_y)

            elif cam_name == "4":
                theta_z = np.deg2rad(-4.2)
                c2w_1 = rotate_camera_around_z_axis(c2w.copy(), theta_z)
                theta_y = np.deg2rad(8)
                c2w_1 = rotate_camera_around_y_axis(c2w_1, theta_y)

                theta_z = np.deg2rad(-5.5)
                c2w_2 = rotate_camera_around_z_axis(c2w.copy(), theta_z)
                theta_y = np.deg2rad(7.3)
                c2w_2 = rotate_camera_around_y_axis(c2w_2, theta_y)

            c2w = c2w_1

        elif capture_part == "black_blue_cloud_extra":
            pass

        else:
            raise ValueError(f"Unknown capture_part: {capture_part}")

        c2w_raw = c2w.copy()
        # change from OpenGL camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        if capture_part == "smoke_and_ball_object":
            # change from OpenGL camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w_2[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c_2 = np.linalg.inv(c2w_2)
            R_2 = np.transpose(w2c_2[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T_2 = w2c_2[:3, 3]

        camera_hw = frame["camera_hw"]
        h, w = camera_hw
        fov_x = frame["camera_angle_x"]
        focal_length = fov2focal(fov_x, w)
        fov_y = focal2fov(focal_length, h)
        FovY = fov_y
        FovX = fov_x
        # print(f"frame {frame['file_path']} focal_length {focal_length} FovX {FovX} FovY {FovY}")
        for time_idx in trange(
            start_time,
            start_time + duration * time_step,
            time_step,
            desc=f"    cam0{cam_name}",
            leave=False,
        ):
            timestamp = (time_idx - start_time) / (duration * time_step) * max_timestamp
            image_name = frame["file_path"].split("/")[-1]

            if capture_part == "smoke":
                tmp_time_idx = min(409, time_idx)
            else:
                tmp_time_idx = time_idx

            if read_image:
                if is_bg:
                    frame_name = os.path.join(frame["file_path"] + "_bg", f"{time_idx:03d}" + extension)
                else:
                    frame_name = os.path.join(frame["file_path"], f"{tmp_time_idx:03d}" + extension)

                # used to determine the loss type
                is_fake_view = False
                real_frame_name = frame_name

                if (train_views_fake is not None and len(train_views_fake) > 0 and cam_name in train_views_fake) or (
                    test_views_fake is not None and len(test_views_fake) > 0 and cam_name in test_views_fake
                ):
                    # print(f"FAKE VIEW: time_idx: {time_idx}, cam_name: {cam_name}, train_views_fake: {train_views_fake}")
                    is_fake_view = True
                    source_cam = train_views[:1]
                    fake_time_idx = time_idx - start_time
                    fake_time_idx = fake_time_idx // time_step

                    if capture_part == "smoke":
                        view_folder = f"zero123_finetune_52000_cam{source_cam}to{cam_name}_cogvxlora5b_strength{refined_strength}_rawsize"

                    elif capture_part == "ball":
                        view_folder = f"zero123_finetune_88000_cam{source_cam}to{cam_name}_cogvxlora5b_strength{refined_strength}_rawsize"

                    elif capture_part == "smoke_and_ball_object":
                        view_folder = f"zero123_finetune_52000_cam{source_cam}to{cam_name}_cogvxlora5b_strength{refined_strength}_start033_rawsize"
                        if data_2_since >= 0 and fake_time_idx >= data_2_since:
                            view_folder = f"zero123_finetune_88000_cam{source_cam}to{cam_name}_cogvxlora5b_strength{refined_strength}_rawsize"

                    frame_name = os.path.join(view_folder, f"frame_{fake_time_idx:06d}.png")

                if gen_future_since >= 0 and time_idx >= gen_future_since * time_step + start_time:
                    if capture_part == "smoke":
                        view_folder = f"camera0{cam_name}_cogvxlora5b_future_prefix9_i2v3_strength{gen_future_strength}_start{gen_future_since}_smoke_rawsize"
                        if is_wind:
                            view_folder = f"camera0{cam_name}_cogvxlora5b_prefix9_i2v3_strength{gen_future_strength}_start{gen_future_since}_wind_smoke_rawsize"
                    elif capture_part == "ball":
                        view_folder = f"camera0{cam_name}_cogvxlora5b_future_prefix9_i2v3_strength{gen_future_strength}_start{gen_future_since}_ball_rawsize"

                    future_time_idx = time_idx - gen_future_since * time_step - start_time
                    future_time_idx = future_time_idx // time_step
                    future_time_idx = gen_future_since + future_time_idx
                    frame_name = os.path.join(view_folder, f"frame_{future_time_idx:06d}.png")
                    # print(frame_name)

                if data_2_path != "" and data_2_since >= 0 and time_idx >= data_2_since * time_step + start_time:
                    cur_path = data_2_path
                else:
                    cur_path = path

                image_path = os.path.join(cur_path, frame_name)
                real_image_path = os.path.join(cur_path, real_frame_name)
                if not os.path.exists(real_image_path):
                    real_image_path = image_path
                # the image_name is used to index the camera so we all use the real name
                # there is no extension in the image_name

                assert os.path.exists(image_path), f"Image path {image_path} does not exist!"

                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                real_image = cv2.imread(real_image_path, cv2.IMREAD_COLOR)
                real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)

            else:
                image_path = ""
                is_fake_view = True
                image = np.zeros((h, w, 3), dtype=np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                real_image = np.zeros((h, w, 3), dtype=np.uint8)
                real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)

            image = Image.fromarray(image)
            real_image = Image.fromarray(real_image)

            if gray_image:
                image = image.convert("L")
                real_image = real_image.convert("L")

            pose = 1 if time_idx == start_time else None
            hp_directions = 1 if time_idx == start_time else None

            uid = camera_uid  # idx * duration//time_step + time_idx
            camera_uid += 1

            camera_time_idx = (time_idx - start_time) // time_step

            # print(f"frame_name {frame_name} timestamp {timestamp} camera uid {uid}")
            if data_2_path != "" and data_2_since >= 0 and time_idx >= data_2_since * time_step + start_time:
                cur_R, cur_T = R_2, T_2
            else:
                cur_R, cur_T = R, T

            if use_demo_cameras:
                cur_R, cur_T = demo_Rs[camera_time_idx], demo_Ts[camera_time_idx]
                image_name += f"_demo{camera_time_idx:03d}"

            cam_infos.append(
                CameraInfo(
                    uid=uid,
                    R=cur_R,
                    T=cur_T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    real_image=real_image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                    time_idx=camera_time_idx,
                    timestamp=timestamp,
                    near=near,
                    far=far,
                    pose=pose,
                    hp_directions=hp_directions,
                    cxr=0.0,
                    cyr=0.0,
                    is_fake_view=is_fake_view,
                )
            )

    return cam_infos
