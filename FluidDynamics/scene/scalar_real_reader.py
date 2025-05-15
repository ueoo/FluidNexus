import json
import os

import cv2
import numpy as np

from PIL import Image
from tqdm import tqdm, trange
from utils.graphics_utils import focal2fov, fov2focal

from .camera_info import CameraInfo
from .functions import shift_image


def read_cameras_from_transforms_scalar_real(
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
    use_refined_fake=False,
    refined_strength="0d26",
    gen_future_since=90,
    gen_prefixed_future="one",
    gen_future_strength="0d75",
    read_image=True,
    real_view_repeat=1,
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
    if real_view_repeat > 1:
        msg += f", real_view_repeat: {real_view_repeat}"
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
        msg += f", read_image"
    if img_offset:
        msg += f" with img_offset"

    print(msg)
    cam_infos = []
    # print(f"start_time {start_time} duration {duration} time_step {time_step}")

    with open(os.path.join(path, transforms_file)) as json_file:
        contents = json.load(json_file)

    near = float(contents["near"])
    far = float(contents["far"])

    voxel_scale = np.array(contents["voxel_scale"])
    voxel_scale = np.broadcast_to(voxel_scale, [3])

    voxel_matrix = np.array(contents["voxel_matrix"])
    voxel_matrix = np.stack([voxel_matrix[:, 2], voxel_matrix[:, 1], voxel_matrix[:, 0], voxel_matrix[:, 3]], axis=1)
    voxel_matrix_inv = np.linalg.inv(voxel_matrix)

    frames = contents["frames"]
    camera_uid = 0

    desc_str = "  reading views"

    for idx, frame in tqdm(enumerate(frames), desc=desc_str, total=len(frames), leave=True):
        # camera idx
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        camera_hw = frame["camera_hw"]
        h, w = camera_hw
        fov_x = frame["camera_angle_x"]
        focal_length = fov2focal(fov_x, w)
        fov_y = focal2fov(focal_length, h)
        FovY = fov_y
        FovX = fov_x
        cam_name = frame["file_path"][-1:]  # train0x -> x used to determine with train_views
        # print(f"frame {frame['file_path']} focal_length {focal_length} FovX {FovX} FovY {FovY}")
        for time_idx in trange(
            start_time, start_time + duration * time_step, time_step, desc=f"    cam0{cam_name}", leave=False
        ):
            timestamp = (time_idx - start_time) / (duration * time_step) * max_timestamp
            image_name = frame["file_path"].split("/")[-1]

            if read_image:
                frame_name = os.path.join("colmap_frames", f"colmap_{time_idx}", frame["file_path"] + extension)
                # used to determine the loss type
                is_fake_view = False
                real_frame_name = frame_name

                if (train_views_fake is not None and len(train_views_fake) > 0 and cam_name in train_views_fake) or (
                    test_views_fake is not None and len(test_views_fake) > 0 and cam_name in test_views_fake
                ):
                    # print(f"FAKE VIEW: time_idx: {time_idx}, cam_name: {cam_name}, train_views_fake: {train_views_fake}")
                    is_fake_view = True

                    source_cam = train_views[:1]
                    view_folder = f"zero123_finetune_15500_cam{source_cam}to{cam_name}_cogvxlora5b_strength{refined_strength}_rawsize"
                    frame_name = os.path.join(view_folder, f"frame_{time_idx:06d}.png")

                    fake_time_idx = time_idx - start_time
                    fake_time_idx = fake_time_idx // time_step
                    frame_name = os.path.join(view_folder, f"frame_{fake_time_idx:06d}.png")
                    # print(frame_name)

                if gen_future_since >= 0 and time_idx >= gen_future_since * time_step + start_time:
                    view_folder = f"train0{cam_name}_cogvxlora5b_future_prefix9_i2v3_strength{gen_future_strength}_start{gen_future_since}_scalar_rawsize"

                    future_time_idx = time_idx - gen_future_since * time_step - start_time
                    future_time_idx = future_time_idx // time_step
                    future_time_idx = gen_future_since + future_time_idx
                    frame_name = os.path.join(view_folder, f"frame_{future_time_idx:06d}.png")
                    # print(frame_name)

                image_path = os.path.join(path, frame_name)
                real_image_path = os.path.join(path, real_frame_name)
                if not os.path.exists(real_image_path):
                    print(f"Real image path {real_image_path} does not exist!")
                    real_image_path = image_path
                # the image_name is used to index the camera so we all use the real name
                # there is no extension in the image_name

                assert os.path.exists(image_path), f"Image path {image_path} does not exist!"

                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                real_image = cv2.imread(real_image_path, cv2.IMREAD_COLOR)
                real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)

                if img_offset:
                    # camera hack, as the camera poses are not perfect
                    if cam_name == "0":
                        image = shift_image(image, -12, 18)
                        real_image = shift_image(real_image, -12, 18)
                    if cam_name == "1":
                        image = shift_image(image, 52, 18)
                        real_image = shift_image(real_image, 52, 18)
                    if cam_name == "3":
                        image = shift_image(image, 11, -12)
                        real_image = shift_image(real_image, 11, -12)
                    if cam_name == "4":
                        image = shift_image(image, 11, -18)
                        real_image = shift_image(real_image, 11, -18)

            else:
                image_path = ""
                is_fake_view = True
                image = np.zeros((h, w, 3), dtype=np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                real_image = np.zeros((h, w, 3), dtype=np.uint8)
                real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)

            image[image < 10] = 0
            real_image[real_image < 10] = 0

            image = Image.fromarray(image)
            real_image = Image.fromarray(real_image)

            if gray_image:
                image = image.convert("L")
                real_image = real_image.convert("L")

            pose = 1 if time_idx == start_time else None
            hp_directions = 1 if time_idx == start_time else None

            uid = camera_uid  # idx * duration//time_step + time_idx
            camera_uid += 1

            # print(f"frame_name {frame_name} timestamp {timestamp} camera uid {uid}")

            cam_infos.append(
                CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    real_image=real_image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                    time_idx=time_idx - start_time,
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
            if real_view_repeat > 1 and not is_fake_view:
                for i in range(1, real_view_repeat):
                    uid = camera_uid  # idx * duration//time_step + time_idx
                    camera_uid += 1

                    cam_infos.append(
                        CameraInfo(
                            uid=uid,
                            R=R,
                            T=T,
                            FovY=FovY,
                            FovX=FovX,
                            image=image,
                            real_image=real_image,
                            image_path=image_path,
                            image_name=image_name,
                            width=image.size[0],
                            height=image.size[1],
                            time_idx=time_idx - start_time,
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
