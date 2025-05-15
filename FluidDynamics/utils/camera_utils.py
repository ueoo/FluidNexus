import numpy as np

from tqdm import tqdm

from scene.camera import Camera
from utils.general_utils import pil_to_torch
from utils.graphics_utils import fov2focal


WARNED = False


def load_cam(args, id, cam_info, resolution_scale):

    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution_w = round(orig_w / (resolution_scale * args.resolution))
        resolution_h = round(orig_h / (resolution_scale * args.resolution))
        resolution = (resolution_w, resolution_h)
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = pil_to_torch(cam_info.image, resolution)
    resized_image_rgb_real = pil_to_torch(cam_info.real_image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    gt_image_real = resized_image_rgb_real[:3, ...]
    loaded_mask = None
    loaded_mask_real = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    if resized_image_rgb_real.shape[1] == 4:
        loaded_mask_real = resized_image_rgb_real[3:4, ...]

    camera_direct = cam_info.hp_directions
    camera_pose = cam_info.pose

    if camera_pose is not None:
        rays_o, rays_d = 1, camera_direct
    else:
        rays_o, rays_d = None, None

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device,
        near=cam_info.near,
        far=cam_info.far,
        time_idx=cam_info.time_idx,
        timestamp=cam_info.timestamp,
        rayo=rays_o,
        rayd=rays_d,
        cxr=cam_info.cxr,
        cyr=cam_info.cyr,
        is_fake_view=cam_info.is_fake_view,
        real_image=gt_image_real,
        gt_alpha_mask_real=loaded_mask_real,
    )


def camera_list_from_cam_infos(cam_infos, resolution_scale, args, split="Train"):
    camera_list = []

    for idx, c in tqdm(enumerate(cam_infos), total=len(cam_infos), desc=f"Loading {split} Cameras"):
        camera_list.append(load_cam(args, idx, c, resolution_scale))

    return camera_list


def camera_to_json(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry
