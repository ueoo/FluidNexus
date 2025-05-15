import math

import torch

from gaussian_splatting.gm_dynamics import GaussianModel


def render_dynamics(
    viewpoint_camera,
    gm: GaussianModel,
    pipe_args,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    GRsetting=None,
    GRzer=None,
    pos_type="visual",
    scale=False,
    prev_visual_xyz=None,
    gpf_only=False,
    gs_only=False,
    debug=False,
    **kwargs,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    if pos_type == "guess_visual_nn":
        raw_render_xyz = gm.get_visual_xyz_from_nn()
    elif pos_type == "guess_visual_hidden":
        raw_render_xyz = gm.get_visual_xyz_from_hidden_guess()
    elif pos_type == "visual":
        raw_render_xyz = gm.get_visual_xyz
    elif pos_type == "hidden":
        raw_render_xyz = gm.get_xyz
    elif pos_type == "rigid":
        raw_render_xyz = gm.get_rigid_xyz
    elif pos_type == "re_sim_visual":
        raw_render_xyz = gm.get_re_sim_visual_xyz
    else:
        raise ValueError(f"Unknown pos_type: {pos_type}")

    if scale:
        render_xyz = raw_render_xyz / gm.scale_factor
    else:
        render_xyz = raw_render_xyz

    if gpf_only:
        gpf_gs_render_xyz = render_xyz
    elif gs_only:
        gpf_gs_render_xyz = gm.get_gs_xyz
    else:
        gs_xyz = gm.get_gs_xyz
        gpf_gs_render_xyz = torch.cat([render_xyz, gs_xyz], dim=0)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screen_space_points = (
        torch.zeros_like(gpf_gs_render_xyz, dtype=gpf_gs_render_xyz.dtype, requires_grad=True, device="cuda") + 0
    )

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

    means3D = gpf_gs_render_xyz
    means2D = screen_space_points

    if pos_type == "hidden":
        gpf_opacity = gm.get_opacity_dummy
        gpf_scales = gm.get_scaling_dummy
        gpf_rotations = gm.get_rotation_dummy
        gpf_colors_precomp = gm.get_color_dummy
    elif pos_type == "rigid":
        gpf_opacity = gm.get_rigid_opacity
        gpf_scales = gm.get_rigid_scaling
        gpf_rotations = gm.get_rigid_rotation
        gpf_colors_precomp = gm.get_rigid_color
    elif pos_type == "high":
        gpf_opacity = gm.get_high_opacity
        gpf_scales = gm.get_high_scaling
        gpf_rotations = gm.get_high_rotation
        gpf_colors_precomp = gm.get_high_color
    elif pos_type == "dense":
        gpf_opacity = gm.get_dense_opacity
        gpf_scales = gm.get_dense_scaling
        gpf_rotations = gm.get_dense_rotation
        gpf_colors_precomp = gm.get_dense_color
    else:
        gpf_opacity = gm.get_visual_opacity
        gpf_scales = gm.get_visual_scaling
        gpf_rotations = gm.get_visual_rotation
        gpf_colors_precomp = gm.get_visual_color

    if gpf_colors_precomp.shape[1] == 1:
        # gpf particles are gray particles
        gpf_colors_precomp = gpf_colors_precomp.repeat(1, 3)

    if gpf_only:
        opacity = gpf_opacity
        scales = gpf_scales
        rotations = gpf_rotations
        colors_precomp = gpf_colors_precomp

    elif gs_only:
        gs_opacity = gm.get_gs_opacity
        gs_scales = gm.get_gs_scaling
        gs_rotations = gm.get_gs_rotation
        gs_colors_precomp = gm.get_gs_color

        opacity = gs_opacity
        scales = gs_scales
        rotations = gs_rotations
        colors_precomp = gs_colors_precomp

    else:
        gs_opacity = gm.get_gs_opacity
        gs_scales = gm.get_gs_scaling
        gs_rotations = gm.get_gs_rotation
        gs_colors_precomp = gm.get_gs_color

        opacity = torch.cat([gpf_opacity, gs_opacity], dim=0)
        scales = torch.cat([gpf_scales, gs_scales], dim=0)
        rotations = torch.cat([gpf_rotations, gs_rotations], dim=0)
        colors_precomp = torch.cat([gpf_colors_precomp, gs_colors_precomp], dim=0)

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
