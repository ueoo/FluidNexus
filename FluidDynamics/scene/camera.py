import numpy as np
import torch

from kornia import create_meshgrid
from torch import nn
from utils.graphics_utils import (
    get_projection_matrix,
    get_projection_matrix_cv,
    get_world_2_view2,
    pix2ndc,
)


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        gt_alpha_mask,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cpu",
        near=0.01,
        far=100.0,
        time_idx=0,
        timestamp=0.0,
        rayo=None,
        rayd=None,
        rays=None,
        cxr=0.0,
        cyr=0.0,
        is_fake_view=False,
        real_image=None,
        gt_alpha_mask_real=None,
    ):
        super().__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time_idx = time_idx
        self.timestamp = timestamp

        self.is_fake_view = is_fake_view

        # image is real image
        if not isinstance(image, tuple):
            if "camera_" not in image_name:
                self.original_image = image.clamp(0.0, 1.0)
                if real_image is not None:
                    self.original_image_real = real_image.clamp(0.0, 1.0)
            else:
                self.original_image = image.clamp(0.0, 1.0).float()
                if real_image is not None:
                    self.original_image_real = real_image.clamp(0.0, 1.0).float()
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]
            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width))

            if gt_alpha_mask_real is not None:
                self.original_image_real *= gt_alpha_mask_real
            else:
                self.original_image_real *= torch.ones((1, self.image_height, self.image_width))

        else:
            self.image_width = image[0]
            self.image_height = image[1]
            self.original_image = None
            self.original_image_real = None

        self.z_far = 100.0
        self.z_near = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(get_world_2_view2(R, T, trans, scale)).transpose(0, 1).cuda()
        if cyr != 0.0:
            self.cxr = cxr
            self.cyr = cyr
            self.projection_matrix = (
                get_projection_matrix_cv(
                    z_near=self.z_near, z_far=self.z_far, fovX=self.FoVx, fovY=self.FoVy, cx=cxr, cy=cyr
                )
                .transpose(0, 1)
                .cuda()
            )
        else:
            self.projection_matrix = (
                get_projection_matrix(z_near=self.z_near, z_far=self.z_far, fovX=self.FoVx, fovY=self.FoVy)
                .transpose(0, 1)
                .cuda()
            )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        if rayd is not None:
            project_inverse = self.projection_matrix.T.inverse()
            camera2wold = self.world_view_transform.T.inverse()
            pix_grid = create_meshgrid(
                self.image_height, self.image_width, normalized_coordinates=False, device="cuda"
            )[0]
            # pix_grid = pix_grid.cuda()  # H,W,

            pix_grid_x = pix_grid[:, :, 0]  # x
            pix_grid_y = pix_grid[:, :, 1]  # y

            ndc_y, ndc_x = pix2ndc(pix_grid_y, self.image_height), pix2ndc(pix_grid_x, self.image_width)
            ndc_x = ndc_x.unsqueeze(-1)
            ndc_y = ndc_y.unsqueeze(-1)  # * (-1.0)

            # N,4
            ndc_camera = torch.cat((ndc_x, ndc_y, torch.ones_like(ndc_y) * (1.0), torch.ones_like(ndc_y)), 2).float()

            projected = ndc_camera @ project_inverse.T
            direction_in_local = projected / projected[:, :, 3:]  # v

            direction = direction_in_local[:, :, :3] @ camera2wold[:3, :3].T
            rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)

            # rayo.permute(2, 0, 1).unsqueeze(0)
            self.rayo = self.camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0)
            self.rayd = rays_d.permute(2, 0, 1).unsqueeze(0)

        else:
            self.rayo = None
            self.rayd = None
