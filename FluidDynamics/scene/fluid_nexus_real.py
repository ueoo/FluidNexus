import os

import numpy as np

from .fluid_nexus_real_reader import read_cameras_from_transforms_real_capture
from .functions import fetch_ply, get_nerf_pp_norm, store_ply
from .scene_info import SceneInfo


def read_scene_fluid_nexus_real(
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
    img_offset=False,
    is_bg=False,
    capture_part="black",
    init_pcd_bg=False,
    init_pcd_object=False,
    init_pcd_large_smoke=False,
    use_refined_fake=False,
    refined_strength="0d26",
    gen_future_since=90,
    gen_prefixed_future="one",
    gen_future_strength="0d75",
    real_view_repeat=1,
    data_2_path="",
    data_2_since=-1,
    use_demo_cameras=False,
    use_extra_transforms=False,
    is_wind=False,
    *args,
    **kwargs,
):
    print("Reading Training ...")
    train_json = "transforms_train.json"
    if (
        set(train_views) != set("0134")
        and 0 < len(train_views) < 4
        and (train_views_fake is None or len(train_views_fake) == 0)
    ):
        # in this mode, just using some real views, no fake views for fitting
        train_views = "".join(sorted(train_views))
        train_json = f"transforms_train_{train_views}.json"
    if set(train_views) == set("01234"):  # use set to ignore the order
        train_json = f"transforms.json"
    if use_extra_transforms:
        train_json = train_json.replace(".json", "_extra.json")
    train_cam_infos = read_cameras_from_transforms_real_capture(
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
        is_bg,
        capture_part,
        use_refined_fake,
        refined_strength,
        gen_future_since,
        gen_prefixed_future,
        gen_future_strength,
        data_2_path,
        data_2_since,
        use_demo_cameras,
        is_wind,
        real_view_repeat=real_view_repeat,
    )

    print("Reading Test ...")
    test_json = "transforms_test.json"
    if test_all_views:
        print("Using all views for testing")
        test_json = f"transforms.json"
    if use_extra_transforms:
        test_json = test_json.replace(".json", "_extra.json")
    test_cam_infos = read_cameras_from_transforms_real_capture(
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
        is_bg,
        capture_part,
        use_refined_fake,
        refined_strength,
        gen_future_since,
        gen_prefixed_future,
        gen_future_strength,
        data_2_path,
        data_2_since,
        use_demo_cameras,
        is_wind,
        read_image=True,
        real_view_repeat=1,
    )

    nerf_normalization = get_nerf_pp_norm(train_cam_infos)

    init_ply_path = os.path.join(model_path, "initial_points3d.ply")
    if os.path.exists(init_ply_path):
        os.remove(init_ply_path)

    if no_init_pcd:
        pcd = None
        print("No init pcd")
    else:
        total_xyz_list = []
        total_rgb_list = []
        total_time_list = []
        if init_pcd_bg:
            assert gray_image is False, "Gray image is not supported for real_capture background"
            # if not os.path.exists(ply_path):
            # Since this data set has no colmap data, we start with random points
            # hyfluid recreate the points every time
            num_pts = 100_000
            print(f"Generating random point cloud ({num_pts}) for background in real_capture...")

            x_min = -1.0
            x_max = 2.5
            y_min = -0.2
            y_max = 2.5
            z_min = -0.6
            z_max = -0.5

            x = np.random.uniform(x_min, x_max, (num_pts, 1))
            y = np.random.uniform(y_min, y_max, (num_pts, 1))
            z = np.random.uniform(z_min, z_max, (num_pts, 1))

            # print(f"Background points init x: {x.min():.5f}, {x.max():.5f}")
            # print(f"Background points init y: {y.min():.5f}, {y.max():.5f}")
            # print(f"Background points init z: {z.min():.5f}, {z.max():.5f}")

            xyz = np.concatenate((x, y, z), axis=1)

            rgb = np.zeros((num_pts, 3)) + 0.7
            time = np.zeros((xyz.shape[0], 1))

            total_xyz_list.append(xyz)
            total_rgb_list.append(rgb)
            total_time_list.append(time)

        if init_pcd_object:
            num_pts = 50_000
            print(f"Generating random point cloud ({num_pts}) for object in real_capture...")
            # change these values according to the object, smoke
            x_mid = 0.328
            y_mid = 0.378
            z_mid = -0.28
            radius = 0.11

            # Golden ratio to distribute points evenly
            golden_ratio = (1 + np.sqrt(5)) / 2

            # Arrays to store the spherical angles
            theta = 2 * np.pi * np.arange(num_pts) / golden_ratio
            phi = np.arccos(1 - 2 * (np.arange(num_pts) + 0.5) / num_pts)

            # Convert spherical coordinates to Cartesian coordinates
            x = x_mid + radius * np.sin(phi) * np.cos(theta)
            y = y_mid + radius * np.sin(phi) * np.sin(theta)
            z = z_mid + radius * np.cos(phi)

            xyz = np.stack((x, y, z), axis=1)

            rgb = np.zeros((num_pts, 3)) + 0.7
            time = np.zeros((xyz.shape[0], 1))

            total_xyz_list.append(xyz)
            total_rgb_list.append(rgb)
            total_time_list.append(time)

        if init_pcd_large_smoke:
            assert gray_image is False, "Gray image is not supported for real_capture background"
            # if not os.path.exists(ply_path):
            # Since this data set has no colmap data, we start with random points
            # hyfluid recreate the points every time
            num_pts = 100_000
            print(f"Generating random point cloud ({num_pts}) for large smoke in real_capture...")

            x_min = 0.0
            x_max = 0.5
            y_min = 0.0
            y_max = 0.7
            z_min = -0.5
            z_max = 0

            x = np.random.uniform(x_min, x_max, (num_pts, 1))
            y = np.random.uniform(y_min, y_max, (num_pts, 1))
            z = np.random.uniform(z_min, z_max, (num_pts, 1))

            xyz = np.concatenate((x, y, z), axis=1)

            rgb = np.zeros((num_pts, 3)) + 0.7
            time = np.zeros((xyz.shape[0], 1))

            total_xyz_list.append(xyz)
            total_rgb_list.append(rgb)
            total_time_list.append(time)

        assert len(total_xyz_list) == len(total_rgb_list) == len(total_time_list) > 0

        total_xyz = np.concatenate(total_xyz_list, axis=0)
        total_rgb = np.concatenate(total_rgb_list, axis=0)
        total_time = np.concatenate(total_time_list, axis=0)

        assert total_xyz.shape[0] == total_rgb.shape[0] == total_time.shape[0]

        total_xyzt = np.concatenate((total_xyz, total_time), axis=1)
        store_ply(init_ply_path, total_xyzt, total_rgb, gray_image)

        pcd = fetch_ply(init_ply_path, gray_image)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=init_ply_path,
        bbox_model=None,
    )
    return scene_info


def read_scene_fluid_nexus_real_eval(
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
    is_bg=False,
    capture_part="black",
    *args,
    **kwargs,
):

    print("Reading Test Transforms...")
    test_json = "transforms_test.json"
    if test_all_views:
        print("Using all views for testing")
        test_json = f"transforms.json"
    test_cam_infos = read_cameras_from_transforms_real_capture(
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
        is_bg=is_bg,
        capture_part=capture_part,
    )

    nerf_normalization = get_nerf_pp_norm(test_cam_infos)

    scene_info = SceneInfo(
        point_cloud=None,
        train_cameras=test_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=None,
        bbox_model=None,
    )
    return scene_info
