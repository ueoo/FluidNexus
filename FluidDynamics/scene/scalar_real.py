import os

import numpy as np

from utils.sh_utils import sh2rgb

from .functions import fetch_ply, get_nerf_pp_norm, store_ply
from .scalar_real_reader import read_cameras_from_transforms_scalar_real
from .scene_info import SceneInfo


def read_scene_scalar_real(
    data_path,
    model_path,
    white_background,
    eval,
    extension=".png",
    start_time=50,
    duration=50,
    time_step=1,
    max_timestamp=1.0,
    gray_image=False,
    train_views="0134",
    train_views_fake=None,
    test_views_fake=None,
    test_all_views=False,
    no_init_pcd=False,
    source_init=False,
    init_region_type="large",
    img_offset=False,
    init_num_pts_per_time=1000,
    init_trbf_c_fix=False,
    init_color_fix_value: float = None,
    use_refined_fake=False,
    refined_strength="0d26",
    gen_future_since=90,
    gen_prefixed_future="one",
    gen_future_strength="0d75",
    real_view_repeat=1,
    *args,
    **kwargs,
):
    print("Reading Training ...")
    train_json = "transforms_train_scalar_real.json"
    if (
        set(train_views) != set("0134")
        and 0 < len(train_views) < 4
        and (train_views_fake is None or len(train_views_fake) == 0)
    ):
        # in this mode, just using some real views, no fake views for fitting
        train_views = "".join(sorted(train_views))
        train_json = f"transforms_train_{train_views}_scalar_real.json"
    if set(train_views) == set("01234"):  # use set to ignore the order
        train_json = f"transforms_train_test_scalar_real.json"
    train_cam_infos = read_cameras_from_transforms_scalar_real(
        data_path,
        train_json,
        white_background,
        extension,
        start_time,
        duration,
        time_step,
        max_timestamp,
        gray_image,
        train_views,
        train_views_fake,
        test_views_fake,
        img_offset,
        use_refined_fake,
        refined_strength,
        gen_future_since,
        gen_prefixed_future,
        gen_future_strength,
        real_view_repeat=real_view_repeat,
    )

    print("Reading Test ...")
    test_json = "transforms_test_scalar_real.json"
    if test_all_views:
        print("Using all views for testing")
        test_json = f"transforms_train_test_scalar_real.json"
    test_cam_infos = read_cameras_from_transforms_scalar_real(
        data_path,
        test_json,
        white_background,
        extension,
        start_time,
        duration,
        time_step,
        max_timestamp,
        gray_image,
        train_views,
        train_views_fake,
        test_views_fake,
        img_offset,
        use_refined_fake,
        refined_strength,
        gen_future_since,
        gen_prefixed_future,
        gen_future_strength,
        read_image=True,
        real_view_repeat=1,
    )

    nerf_normalization = get_nerf_pp_norm(train_cam_infos)

    total_ply_path = os.path.join(model_path, "initial_points3d_total.ply")
    if os.path.exists(total_ply_path):
        os.remove(total_ply_path)

    img_channel = 1 if gray_image else 3

    if no_init_pcd:
        pcd = None

    else:
        if init_region_type == "large":
            radius_max = 0.18  # default value 0.18  source region 0.026
            x_mid = 0.34  # default value 0.34 source region 0.34
            y_min = -0.01  # default value -0.01  source region -0.01
            y_max = 0.7  # default value 0.7  source region 0.05
            z_mid = -0.225  # default value -0.225  source region -0.225

        elif init_region_type == "small":
            radius_max = 0.026  # default value 0.18  source region 0.026
            x_mid = 0.34  # default value 0.34 source region 0.34
            y_min = -0.01  # default value -0.01  source region -0.01
            y_max = 0.03  # default value 0.7  source region 0.05
            z_mid = -0.225  # default value -0.225  source region -0.225

        elif init_region_type == "adaptive":
            radius_max_range = [0.026, 0.18]
            x_mid = 0.34
            z_mid = -0.225
            y_min = -0.01
            y_max_range = [0.03, 0.7]

        else:
            raise ValueError(f"Unknown init_region_type: {init_region_type}")

        if source_init:
            num_pts = init_num_pts_per_time
            print(f"Init {num_pts} points with {init_region_type} region type with source_init mode.")
            assert init_region_type in [
                "small",
                "large",
            ], f"In source_init mode, init_region_type must be small or large."
            print(f"Generating source_init random point cloud ({num_pts})...")
            y = np.random.uniform(y_min, y_max, (num_pts, 1))  # [-0.05, 0.15] [-0.05, 0.7]

            radius = np.random.random((num_pts, 1)) * radius_max  # * 0.03 # 0.18
            theta = np.random.random((num_pts, 1)) * 2 * np.pi
            x = radius * np.cos(theta) + x_mid
            z = radius * np.sin(theta) + z_mid

            xyz = np.concatenate((x, y, z), axis=1)

            shs = np.random.random((num_pts, img_channel)) / 255.0
            # rgb = np.random.random((num_pts, 3)) * 255.0
            rgb = sh2rgb(shs) * 255

            # print(f"init time {(i - start_time) / duration}")
            # when using our adding source, the time is not directly used
            time = np.zeros((xyz.shape[0], 1))

        else:
            # if the render pipeline is time-based activation and the init_region_type is large, the number of points should be larger
            num_pts = init_num_pts_per_time
            total_xyz = []
            total_rgb = []
            total_time = []
            print(f"Init {num_pts} points per time with {init_region_type} region type with time-based mode.")
            for i in range(start_time, start_time + duration, time_step):
                if init_region_type == "adaptive":
                    y_max = y_max_range[0] + (y_max_range[1] - y_max_range[0]) * (i - start_time) / duration
                    radius_max = (
                        radius_max_range[0] + (radius_max_range[1] - radius_max_range[0]) * (i - start_time) / duration
                    )

                y = np.random.uniform(y_min, y_max, (num_pts, 1))

                radius = np.random.random((num_pts, 1)) * radius_max
                theta = np.random.random((num_pts, 1)) * 2 * np.pi
                x = radius * np.cos(theta) + x_mid
                z = radius * np.sin(theta) + z_mid

                # print(f"Points init x: {x.min()}, {x.max()}")
                # print(f"Points init y: {y.min()}, {y.max()}")
                # print(f"Points init z: {z.min()}, {z.max()}")

                xyz = np.concatenate((x, y, z), axis=1)

                if init_color_fix_value is not None and isinstance(init_color_fix_value, float):
                    # 0.6 does not matter, the init value in Gaussian Model is used
                    rgb = np.ones((num_pts, img_channel)) * init_color_fix_value * 255.0
                else:
                    shs = np.random.random((num_pts, img_channel)) / 255.0
                    rgb = np.random.random((num_pts, 3)) * 255.0
                    rgb = sh2rgb(shs) * 255

                total_xyz.append(xyz)
                # rgb is not used for fixed color
                total_rgb.append(rgb)
                # print(f"init time {(i - start_time) / duration}")
                # when using our adding source, the time is not directly used
                if init_trbf_c_fix:
                    total_time.append(np.zeros((xyz.shape[0], 1)))
                else:
                    total_time.append(np.ones((xyz.shape[0], 1)) * (i - start_time) / duration * max_timestamp)

            xyz = np.concatenate(total_xyz, axis=0)
            rgb = np.concatenate(total_rgb, axis=0)
            time = np.concatenate(total_time, axis=0)

        assert xyz.shape[0] == rgb.shape[0]

        xyzt = np.concatenate((xyz, time), axis=1)
        store_ply(total_ply_path, xyzt, rgb, gray_image)

        pcd = fetch_ply(total_ply_path, gray_image)

        assert pcd is not None, "Point cloud could not be loaded!"

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=total_ply_path,
    )
    return scene_info


def read_scene_scalar_real_eval(
    data_path,
    model_path,
    white_background,
    eval,
    extension=".png",
    start_time=50,
    duration=50,
    time_step=1,
    max_timestamp=1.0,
    gray_image=False,
    train_views="0134",
    train_views_fake=None,
    test_all_views=False,
    img_offset=False,
    *args,
    **kwargs,
):

    print("Reading Test Transforms...")
    test_json = "transforms_test_scalar_real.json"
    if test_all_views:
        print("Using all views for testing")
        test_json = f"transforms_train_test_scalar_real.json"
    test_cam_infos = read_cameras_from_transforms_scalar_real(
        data_path,
        test_json,
        white_background,
        extension,
        start_time,
        duration,
        time_step,
        max_timestamp,
        gray_image,
        train_views,
        train_views_fake,
        img_offset,
    )

    nerf_normalization = get_nerf_pp_norm(test_cam_infos)

    total_ply_path = os.path.join(model_path, "initial_points3d_total.ply")
    pcd = fetch_ply(total_ply_path, gray_image)

    assert pcd is not None, "Point cloud could not be loaded!"

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=test_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=total_ply_path,
    )
    return scene_info
