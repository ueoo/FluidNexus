import math
import time

import torch

from gaussian_splatting.gm_background import GaussianModel


def render_background(
    viewpoint_camera,
    gm: GaussianModel,
    pipe_args,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    GRsetting=None,
    GRzer=None,
    **kwargs,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    raw_render_xyz = gm.get_xyz
    render_xyz = raw_render_xyz

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screen_space_points = torch.zeros_like(render_xyz, dtype=render_xyz.dtype, requires_grad=True, device="cuda") + 0

    try:
        screen_space_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tan_fov_x = math.tan(viewpoint_camera.FoVx * 0.5)
    tan_fov_y = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tan_fov_x=tan_fov_x,
        tan_fov_y=tan_fov_y,
        bg=bg_color.float(),
        scale_modifier=scaling_modifier,
        view_matrix=viewpoint_camera.world_view_transform,
        proj_matrix=viewpoint_camera.full_proj_transform,
        sh_degree=gm.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
    )

    rasterizer = GRzer(raster_settings=raster_settings)

    means3D = render_xyz
    means2D = screen_space_points

    opacity = gm.get_opacity
    scales = gm.get_scaling
    rotations = gm.get_rotation
    colors_precomp = gm.get_color

    cov3D_precomp = None

    shs = None

    rendered_image, radii, depth = rasterizer(
        means3D=means3D.float(),
        means2D=means2D.float(),
        shs=shs,
        colors_precomp=colors_precomp.float(),
        opacities=opacity.float(),
        scales=scales.float(),
        rotations=rotations.float(),
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "render": rendered_image,
        "viewspace_points": screen_space_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "opacity": opacity,
        "depth": depth,
        "render_xyz": render_xyz,
        "raw_render_xyz": raw_render_xyz,
        "means3D": means3D,
        "means2D": means2D,
        "opacity": opacity,
        "rotations": rotations,
        "colors_precomp": colors_precomp,
        "scales": scales,
    }
