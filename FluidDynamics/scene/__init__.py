import os

from arguments import ModelParams
from gaussian_splatting.gm_fluid import GaussianModel
from helpers.helper_train import record_points_helper
from scene.dataset_readers import scene_load_type_callbacks
from utils.camera_utils import camera_list_from_cam_infos


class Scene:
    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        resolution_scales=[1.0],
        loader="fluid_nexus_real",
        **kwargs,
    ):

        self.model_path = args.model_path
        self.gaussians = gaussians

        all_loaders = scene_load_type_callbacks.keys()
        eval_loaders = [loader for loader in all_loaders if "eval" in loader]
        assert loader in all_loaders, f"Could not recognize loader type: {loader}"

        self.train_cameras = {}
        self.test_cameras = {}

        scene_info = scene_load_type_callbacks[loader](**args.__dict__)

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.bbox_model = scene_info.bbox_model

        for res_scale in resolution_scales:
            if loader in eval_loaders:
                self.train_cameras[res_scale] = []

            else:
                self.train_cameras[res_scale] = camera_list_from_cam_infos(
                    scene_info.train_cameras,
                    res_scale,
                    args,
                    "Train",
                )

            self.test_cameras[res_scale] = camera_list_from_cam_infos(
                scene_info.test_cameras,
                res_scale,
                args,
                "Test",
            )

        self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration, type=None, frame_idx=None):
        if type == "sim":
            assert frame_idx is not None
            point_cloud_path = os.path.join(
                self.model_path, f"point_cloud_sim/frame_{frame_idx:03d}_iteration_{iteration:05d}"
            )
        else:
            point_cloud_path = os.path.join(self.model_path, f"point_cloud/iteration_{iteration:05d}")
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def record_points(self, iteration, string, two_level=False, pbd=False):
        num_points = self.gaussians._xyz.shape[0]
        record_points_helper(self.model_path, num_points, iteration, string)

    def get_train_cameras(self, scale=1.0):
        return self.train_cameras[scale]

    def get_test_cameras(self, scale=1.0):
        return self.test_cameras[scale]
