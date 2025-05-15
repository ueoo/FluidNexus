import json
import os

import numpy as np
import torch

from simple_knn._C import distCUDA2
from torch import nn
from torch_cluster import radius, radius_graph
from torch_scatter import scatter_min

from utils.general_utils import (
    build_scaling_rotation,
    get_expon_lr_func,
    inv_sigmoid,
    strip_symmetric,
)
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.opacity_inverse_activation = inv_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, *args, **kwargs):
        self.active_sh_degree = 0
        # the fluid particles, used for PBD constraints
        self._xyz = torch.empty(0)
        self._estimate_xyz = torch.empty(0)
        self._force = torch.empty(0)
        self._velocity = torch.empty(0)
        self._imass = torch.empty(0)
        self._particle_id = torch.empty(0)
        self._particle_id_max = 0
        self.hidden_particles_created = False

        # the visual particles, used for rendering
        self._visual_xyz = torch.empty(0)

        # currently, these GS attributes are constant
        self._visual_color = torch.empty(0)
        self._visual_scales = torch.empty(0)
        self._visual_rotation = torch.empty(0)
        self._visual_opacity = torch.empty(0)
        # self._visual_omega = torch.empty(0)
        self.visual_particles_created = False

        # Maybe we need them for densification
        # currently, densification related tensors are not used
        self.visual_max_radii2D = torch.empty(0)
        self.visual_xyz_gradient_accum = torch.empty(0)
        self.visual_denom = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.setup_functions()

    def setup_constants(self, optim_args):
        self._gravity = torch.tensor([0.0, -9.8, 0.0], dtype=torch.float, device="cuda").reshape((1, 3))

        # a frame is 0.0333s
        # in flex seconds are 0.01, DONT CHANGE
        # to align, we tick 3 times per frame
        self._secs = optim_args.secs  # 0.01

        # Gravity scaling factor for gases
        self.alpha = optim_args.alpha  # -0.2
        self.buoyancy_decay_rate = optim_args.buoyancy_decay_rate
        self.buoyancy_max_y = optim_args.buoyancy_max_y
        self.beta = optim_args.beta

        # 0.00625 is the distance threshold
        self.H = optim_args.H
        self.H2 = self.H**2
        self.H6 = self.H**6
        self.H9 = self.H**9

        self.min_neighbors = optim_args.min_neighbors
        self.remove_out_boundary = optim_args.remove_out_boundary

        self.EPSILON = 1e-8

        self.RELAXATION = 0.01
        self.K_P = 0.2
        self.E_P = 4
        self.DQ_P = 0.25

        self.p0 = optim_args.p0
        self.k = optim_args.k

        self.new_hidden_particles_per_sec = optim_args.new_hidden_particles_per_sec
        self.new_visual_particles_per_sec = optim_args.new_visual_particles_per_sec
        self.visual_timer = 0.0
        self.hidden_timer = 0.0

        self.emitter_points_off_y0 = optim_args.emitter_points_off_y0

        # the emitter should emit particles uniformly
        self.emit_ratio_hidden = optim_args.emit_ratio_hidden
        self.emit_ratio_visual = optim_args.emit_ratio_visual

        self.emit_counter = 0

        self.scale_factor = 100.0

        self.poly6_term1 = 315.0 / (64.0 * np.pi * self.H9)
        self.spiky_grad_term1 = 45.0 / (np.pi * self.H6)
        self.lamb_corr_denom = self.poly6(self.DQ_P * self.DQ_P * self.H * self.H)

        self.KNN_K = optim_args.KNN_K

        # Only valid in level two optimization
        self.fit_xyz = optim_args.fit_xyz
        self.fit_color = optim_args.fit_color
        self.fit_opacity = optim_args.fit_opacity
        self.fit_scales = optim_args.fit_scales
        self.fit_rotation = optim_args.fit_rotation

        assert isinstance(optim_args.wind_force, list) and len(optim_args.wind_force) == 3
        self.wind_force = torch.tensor(optim_args.wind_force, dtype=torch.float, device="cuda").reshape((1, 3))
        self.wind_force_max = max(optim_args.wind_force)
        self.wind_power = optim_args.wind_power

        assert optim_args.rigid_body in ["cuboid", "sphere", "cylinder"]
        self.rigid_body = optim_args.rigid_body
        self.rigid_particle_radius = optim_args.rigid_particle_radius
        self.rigid_particle_diameter = 2 * self.rigid_particle_radius

        self.rigid_body_center = torch.tensor(optim_args.rigid_body_center, dtype=torch.float, device="cuda")
        self.rigid_body_center *= self.scale_factor

        # parameters for "cuboid" rigid body
        self.rigid_cuboid_num_one_side = optim_args.rigid_cuboid_num_one_side
        self.rigid_cuboid_num = optim_args.rigid_cuboid_num
        # parameters for "sphere" rigid body
        self.rigid_sphere_radius = optim_args.rigid_sphere_radius
        self.rigid_sphere_num = optim_args.rigid_sphere_num
        # parameters for "cylinder" rigid body
        self.rigid_cylinder_radius = optim_args.rigid_cylinder_radius
        self.rigid_cylinder_num = optim_args.rigid_cylinder_num

        self.total_iterations = 0
        self.total_sim_iterations = 0
        self.total_tb_log_iterations = 0

        self.record_time = optim_args.record_time

    def poly6(self, r2):
        term2 = self.H2 - r2
        mask = r2 < self.H2
        return mask * self.poly6_term1 * (term2**3)

    def spiky_grad(self, r, rlen):
        mask = (rlen < self.H) & (rlen > 0)
        r_norm = r / (rlen.unsqueeze(-1) + self.EPSILON)
        term2 = (self.H - rlen).unsqueeze(-1) ** 2
        grad = -r_norm * self.spiky_grad_term1 * term2
        grad[~mask] = 0.0
        return grad

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_estimate_xyz(self):
        return self._estimate_xyz

    @property
    def get_force(self):
        return self._force

    @property
    def get_velocity(self):
        return self._velocity

    @property
    def get_imass(self):
        return self._imass

    @property
    def get_color_dummy(self):
        return self._color_dummy

    @property
    def get_scaling_dummy(self):
        return self.scaling_activation(self._scales_dummy)

    @property
    def get_rotation_dummy(self):
        return self.rotation_activation(self._rotation_dummy)

    @property
    def get_opacity_dummy(self):
        return self.opacity_activation(self._opacity_dummy)

    @property
    def get_visual_xyz(self):
        return self._visual_xyz

    @property
    def get_visual_color(self):
        return self._visual_color

    @property
    def get_visual_scaling(self):
        return self.scaling_activation(self._visual_scales)

    @property
    def get_visual_rotation(self):
        return self.rotation_activation(self._visual_rotation)

    @property
    def get_visual_opacity(self):
        return self.opacity_activation(self._visual_opacity)

    @property
    def get_re_sim_visual_xyz(self):
        return self._re_sim_visual_xyz

    @property
    def get_rigid_xyz(self):
        return self._rigid_xyz

    @property
    def get_rigid_color(self):
        return self._rigid_color

    @property
    def get_rigid_scaling(self):
        return self.scaling_activation(self._rigid_scales)

    @property
    def get_rigid_rotation(self):
        return self.rotation_activation(self._rigid_rotation)

    @property
    def get_rigid_opacity(self):
        return self.opacity_activation(self._rigid_opacity)

    @property
    def get_high_xyz(self):
        return self._high_xyz

    @property
    def get_high_color(self):
        return self._high_color

    @property
    def get_high_scaling(self):
        return self.scaling_activation(self._high_scales)

    @property
    def get_high_rotation(self):
        return self.rotation_activation(self._high_rotation)

    @property
    def get_high_opacity(self):
        return self.opacity_activation(self._high_opacity)

    @property
    def get_dense_xyz(self):
        return self._dense_xyz + self._dense_delta

    @property
    def get_dense_color(self):
        return self._dense_color

    @property
    def get_dense_scaling(self):
        return self.scaling_activation(self._dense_scales)

    @property
    def get_dense_rotation(self):
        return self.rotation_activation(self._dense_rotation)

    @property
    def get_dense_opacity(self):
        return self.opacity_activation(self._dense_opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        # just for back compatibility
        self.spatial_lr_scale = spatial_lr_scale
        # print("spatial_lr_scale", self.spatial_lr_scale)

    def training_setup_first_visual(self, optim_args):
        self.percent_dense = optim_args.percent_dense
        self.visual_xyz_gradient_accum = torch.zeros(
            (self.get_visual_xyz.shape[0], 1), dtype=torch.float, device="cuda"
        )
        self.visual_denom = torch.zeros((self.get_visual_xyz.shape[0], 1), dtype=torch.float, device="cuda")
        self._visual_xyz = nn.Parameter(self._visual_xyz.requires_grad_(True))
        l = [
            {
                "params": [self._visual_xyz],
                "lr": optim_args.position_lr_init * self.spatial_lr_scale,
                "name": "visual_xyz",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=optim_args.position_lr_init * self.spatial_lr_scale,
            lr_final=optim_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=optim_args.position_lr_delay_mult,
            max_steps=optim_args.position_lr_max_steps,
        )

    def training_setup_current(self, optim_args):
        # in current stage, i.e., the rest frames
        # in optimizer, we scale down the tensor by scale_factor, for better optimization
        init_estimate_xyz_nn = self._estimate_xyz.detach().clone() / self.scale_factor
        self._estimate_xyz_nn = nn.Parameter(init_estimate_xyz_nn.detach().clone().requires_grad_(True))
        # init_velocity_nn = self._velocity.detach().clone() / self.scale_factor
        # self._velocity_nn = nn.Parameter(init_velocity_nn.detach().clone().requires_grad_(True))
        l = [
            {
                "params": [self._estimate_xyz_nn],
                "lr": optim_args.position_lr_init * self.spatial_lr_scale,
                "name": "estimate_xyz_nn",
            },
            # {
            #     "params": [self._velocity_nn],
            #     "lr": optim_args.position_lr_init * self.spatial_lr_scale,
            #     "name": "velocity_nn",
            # },
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=optim_args.position_lr_init * self.spatial_lr_scale,
            lr_final=optim_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=optim_args.position_lr_delay_mult,
            max_steps=optim_args.position_lr_max_steps,
        )

    def init_quantities_current_level_two(self, optim_args, prev_color, prev_opacity, prev_scales, prev_rotation):
        if self.fit_scales and optim_args.init_scales_w_xyz_dist:
            dist2 = torch.clamp_min(distCUDA2(self._visual_xyz.float().cuda()), 0.0000001)
            scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
            scales = torch.clamp(scales, -10, 1.0)

            self._visual_scales = scales

        if self.fit_color and prev_color is not None and optim_args.inherit_prev_color:
            self._visual_color[: prev_color.shape[0]] = prev_color.clone()
        if self.fit_opacity and prev_opacity is not None and optim_args.inherit_prev_opacity:
            self._visual_opacity[: prev_opacity.shape[0]] = prev_opacity.clone()
        if self.fit_scales and prev_scales is not None and optim_args.inherit_prev_scales:
            self._visual_scales[: prev_scales.shape[0]] = prev_scales.clone()
        if self.fit_rotation and prev_rotation is not None and optim_args.inherit_prev_rotation:
            self._visual_rotation[: prev_rotation.shape[0]] = prev_rotation.clone()

    def training_setup_current_level_two(self, optim_args):
        # in current stage, i.e., the rest frames
        # in optimizer, we scale down the tensor by scale_factor, for better optimization
        l = []
        if self.fit_color:
            self._visual_color = nn.Parameter(self._visual_color.detach().clone().requires_grad_(True))
            l += [{"params": [self._visual_color], "lr": optim_args.visual_color_lr, "name": "visual_color"}]
        if self.fit_opacity:
            self._visual_opacity = nn.Parameter(self._visual_opacity.detach().clone().requires_grad_(True))
            l += [{"params": [self._visual_opacity], "lr": optim_args.visual_opacity_lr, "name": "visual_opacity"}]
        if self.fit_scales:
            self._visual_scales = nn.Parameter(self._visual_scales.detach().clone().requires_grad_(True))
            l += [{"params": [self._visual_scales], "lr": optim_args.visual_scales_lr, "name": "visual_scales"}]
        if self.fit_rotation:
            self._visual_rotation = nn.Parameter(self._visual_rotation.detach().clone().requires_grad_(True))
            l += [{"params": [self._visual_rotation], "lr": optim_args.visual_rotation_lr, "name": "visual_rotation"}]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def update_learning_rate_first_visual(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "visual_xyz":
                lr = self.xyz_scheduler_args(iteration)
                # param_group["lr"] = 0.0
                return lr

    def update_learning_rate_current(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "estimate_xyz_nn":
                lr = self.xyz_scheduler_args(iteration)
                # param_group["lr"] = 0.0
                return lr

    def zero_gradient_cache_first_visual(self):
        self._visual_xyz_grad = torch.zeros_like(self._visual_xyz, device="cuda")

    def cache_gradient_first_visual(self):
        self._visual_xyz_grad += self._visual_xyz.grad

    def set_batch_gradient_first_visual(self, batch_size):
        ratio = 1.0 / batch_size
        self._visual_xyz.grad = self._visual_xyz_grad * ratio

    def zero_gradient_cache_current(self):
        self._estimate_xyz_nn_grad = torch.zeros_like(self._estimate_xyz_nn, device="cuda")
        # self._velocity_nn_grad = torch.zeros_like(self._velocity_nn, device="cuda")

    def cache_gradient_current(self):
        self._estimate_xyz_nn_grad += self._estimate_xyz_nn.grad
        # self._velocity_nn_grad += self._velocity_nn.grad

    def set_batch_gradient_current(self, batch_size):
        ratio = 1.0 / batch_size
        self._estimate_xyz_nn.grad = self._estimate_xyz_nn_grad * ratio
        # self._velocity_nn.grad = self._velocity_nn_grad * ratio

    def zero_gradient_cache_current_level_two(self):
        if self.fit_color:
            self._visual_color_grad = torch.zeros_like(self._visual_color, device="cuda")
        if self.fit_opacity:
            self._visual_opacity_grad = torch.zeros_like(self._visual_opacity, device="cuda")
        if self.fit_scales:
            self._visual_scales_grad = torch.zeros_like(self._visual_scales, device="cuda")
        if self.fit_rotation:
            self._visual_rotation_grad = torch.zeros_like(self._visual_rotation, device="cuda")

    def cache_gradient_current_level_two(self):
        if self.fit_color:
            self._visual_color_grad += self._visual_color.grad
        if self.fit_opacity:
            self._visual_opacity_grad += self._visual_opacity.grad
        if self.fit_scales:
            self._visual_scales_grad += self._visual_scales.grad
        if self.fit_rotation:
            self._visual_rotation_grad += self._visual_rotation.grad

    def set_batch_gradient_current_level_two(self, batch_size):
        ratio = 1.0 / batch_size
        if self.fit_color:
            self._visual_color.grad = self._visual_color_grad * ratio
        if self.fit_opacity:
            self._visual_opacity.grad = self._visual_opacity_grad * ratio
        if self.fit_scales:
            self._visual_scales.grad = self._visual_scales_grad * ratio
        if self.fit_rotation:
            self._visual_rotation.grad = self._visual_rotation_grad * ratio

    @torch.no_grad
    def detach_visual_and_scale(self):
        self._visual_xyz = self._visual_xyz.detach().clone().requires_grad_(False) * self.scale_factor

    @torch.no_grad
    def create_particles_visual(self):
        # initialize for first frame
        visual_num_pts = 600
        visual_radius_max = 0.03
        visual_x_mid = 0.34
        visual_y_min = -0.01
        visual_y_max = 0.04
        visual_z_mid = -0.225

        visual_y = np.random.uniform(visual_y_min, visual_y_max, (visual_num_pts, 1))

        radius = np.random.random((visual_num_pts, 1)) * visual_radius_max
        theta = np.random.random((visual_num_pts, 1)) * 2 * np.pi
        visual_x = radius * np.cos(theta) + visual_x_mid
        visual_z = radius * np.sin(theta) + visual_z_mid
        visual_xyz = np.concatenate((visual_x, visual_y, visual_z), axis=1)
        self._visual_xyz = torch.from_numpy(visual_xyz).float().cuda()
        print(f"Created {self._visual_xyz.shape[0]} visual particles")

        self.visual_particles_created = True

    @torch.no_grad
    def create_particles_hidden(self):
        ## uniform pillar region
        radius_max = 0.1
        delta = 0.009  # best 0.0105 for secs 0.01, 0.009 for secs 0.033
        x_mid = 0.34
        y_min = -0.02
        y_max = 0.08
        z_mid = -0.225

        # Generate points within the range
        x_range = np.arange(x_mid - radius_max, x_mid + radius_max + delta, delta)
        y_range = np.arange(y_min, y_max, delta)
        z_range = np.arange(z_mid - radius_max, z_mid + radius_max + delta, delta)

        points = []

        for x in x_range:
            for y in y_range:
                for z in z_range:
                    if (x - x_mid) ** 2 + (z - z_mid) ** 2 <= radius_max**2:
                        points.append([x, y, z])
        xyz = np.array(points) * self.scale_factor

        self._xyz = torch.from_numpy(xyz).float().cuda()

        self._estimate_xyz = torch.zeros(
            (self._xyz.shape[0], 3), requires_grad=False, dtype=torch.float, device="cuda"
        )
        self._buoyancy = torch.ones((self._xyz.shape[0], 3), requires_grad=False, dtype=torch.float, device="cuda")
        self._buoyancy *= self._gravity * self.alpha
        self._force = torch.zeros((self._xyz.shape[0], 3), requires_grad=False, dtype=torch.float, device="cuda")
        self._velocity = torch.zeros((self._xyz.shape[0], 3), requires_grad=False, dtype=torch.float, device="cuda")
        # the particles' mass is 1
        self._imass = torch.ones((self._xyz.shape[0], 1), requires_grad=False, dtype=torch.float, device="cuda")
        self._counts = torch.zeros((self._xyz.shape[0], 1), requires_grad=False, dtype=torch.float, device="cuda")
        self._particle_id = torch.arange(self._xyz.shape[0], device="cuda").unsqueeze(1)
        self._particle_id_max = self._xyz.shape[0]

        print(f"Created {self._xyz.shape[0]} hidden particles")
        self.hidden_particles_created = True

    @torch.no_grad
    def create_rigid_body(self):
        # the diameter of the rigid body is 0.5 in scaled space,

        diam = self.rigid_particle_diameter

        if self.rigid_body == "cuboid":
            x_num, y_num, z_num = self.rigid_cuboid_num
            rigid_points = []
            for i in range(x_num):
                for j in range(y_num):
                    for k in range(z_num):
                        x = i * diam - x_num // 2 * diam
                        y = j * diam - y_num // 2 * diam
                        z = k * diam - z_num // 2 * diam
                        if i != 0 and i != x_num - 1 and j != 0 and j != y_num - 1 and k != 0 and k != z_num - 1:
                            # only use the edge points
                            continue

                        rigid_points.append([x, y, z])
            rigid_xyz = np.array(rigid_points)

        elif self.rigid_body == "sphere":
            num_particles = self.rigid_sphere_num
            radius = self.rigid_sphere_radius
            phi = np.random.uniform(0, 2 * np.pi, num_particles)

            # Generate random cosines of the polar angles (theta) uniformly between -1 and 1
            cos_theta = np.random.uniform(-1, 1, num_particles)

            # Calculate the polar angles (theta)
            theta = np.arccos(cos_theta)

            # Calculate the x, y, z coordinates of the particles
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)

            # Combine x, y, z coordinates into an array
            rigid_xyz = np.vstack((x, y, z)).T

        elif self.rigid_body == "cylinder":
            diam = self.rigid_particle_diameter
            radius = self.rigid_cylinder_radius  # Radius of the cylinder
            num_cycle, num_height = self.rigid_cylinder_num  # Number of points to generate around the cylinder

            rigid_points = []
            for i in range(num_cycle):
                for j in range(num_height):
                    theta = i * 2 * np.pi / num_cycle
                    x = radius * np.cos(theta)
                    y = radius * np.sin(theta)
                    z = j - num_height / 2
                    z *= diam
                    rigid_points.append([x, y, z])

            rigid_xyz = np.array(rigid_points)

        n_rigid = rigid_xyz.shape[0]
        self._rigid_xyz = torch.from_numpy(rigid_xyz).float().cuda() + self.rigid_body_center
        self._rigid_imass = torch.zeros((n_rigid, 1), requires_grad=False, dtype=torch.float, device="cuda")

    @torch.no_grad
    def prepare_emitter_points(self):
        hidden_delta = 0.015
        visual_delta = 0.00625
        center_x, center_z = 0.34, -0.225
        center_y_hidden = -0.02
        center_y_visual = -0.01

        # Radius of the circle
        visual_radius = visual_delta * 4
        hidden_radius = hidden_delta * 6

        # Generate points within the range
        # stop is not included, so add delta
        visual_x_range = np.arange(center_x - visual_radius, center_x + visual_radius + visual_delta, visual_delta)
        visual_z_range = np.arange(center_z - visual_radius, center_z + visual_radius + visual_delta, visual_delta)

        visual_points = []

        for x in visual_x_range:
            for z in visual_z_range:
                if (x - center_x) ** 2 + (z - center_z) ** 2 <= visual_radius**2:
                    visual_points.append([x, center_y_visual, z])
        self.visual_emitter_points = torch.tensor(visual_points, dtype=torch.float, device="cuda")

        # Generate points within the range
        hidden_x_range = np.arange(center_x - hidden_radius, center_x + hidden_radius + hidden_delta, hidden_delta)
        hidden_z_range = np.arange(center_z - hidden_radius, center_z + hidden_radius + hidden_delta, hidden_delta)

        hidden_points = []

        for x in hidden_x_range:
            for z in hidden_z_range:
                if (x - center_x) ** 2 + (z - center_z) ** 2 <= hidden_radius**2:
                    hidden_points.append([x, center_y_hidden, z])
        self.hidden_emitter_points = torch.tensor(hidden_points, dtype=torch.float, device="cuda")
        num_visual_pts = self.visual_emitter_points.shape[0]
        num_hidden_pts = self.hidden_emitter_points.shape[0]
        print(f"Prepared {num_visual_pts} visual and {num_hidden_pts} hidden emitter points")
        print(f"with {self.emit_ratio_visual} visual and {self.emit_ratio_hidden} hidden emit ratio")

    @torch.no_grad
    def prepare_emitter_future_first_points(self):
        hidden_delta = 0.015
        visual_delta = 0.00625
        center_x, center_z = 0.34, -0.225
        center_y_hidden = -0.02
        center_y_visual = -0.01

        # Radius of the circle
        visual_radius = visual_delta * 4
        hidden_radius = hidden_delta * 6

        # Generate points within the range
        # stop is not included, so add delta
        visual_x_range = np.arange(center_x - visual_radius, center_x + visual_radius + visual_delta, visual_delta)
        visual_y_range = np.arange(center_y_visual, center_y_visual + visual_radius * 2 + visual_delta, visual_delta)
        visual_z_range = np.arange(center_z - visual_radius, center_z + visual_radius + visual_delta, visual_delta)

        visual_points = []

        for x in visual_x_range:
            for y in visual_y_range:
                for z in visual_z_range:
                    if (x - center_x) ** 2 + (z - center_z) ** 2 <= visual_radius**2:
                        visual_points.append([x, y, z])
        self.visual_emitter_first_points = torch.tensor(visual_points, dtype=torch.float, device="cuda")

        # Generate points within the range
        hidden_x_range = np.arange(center_x - hidden_radius, center_x + hidden_radius + hidden_delta, hidden_delta)
        hidden_y_range = np.arange(center_y_hidden, center_y_hidden + hidden_radius * 2 + hidden_delta, hidden_delta)
        hidden_z_range = np.arange(center_z - hidden_radius, center_z + hidden_radius + hidden_delta, hidden_delta)

        hidden_points = []

        for x in hidden_x_range:
            for y in hidden_y_range:
                for z in hidden_z_range:
                    if (x - center_x) ** 2 + (z - center_z) ** 2 <= hidden_radius**2:
                        hidden_points.append([x, y, z])
        self.hidden_emitter_first_points = torch.tensor(hidden_points, dtype=torch.float, device="cuda")
        num_visual_pts = self.visual_emitter_first_points.shape[0]
        num_hidden_pts = self.hidden_emitter_first_points.shape[0]
        print(f"Prepared {num_visual_pts} visual and {num_hidden_pts} hidden emitter future first points")

    @torch.no_grad
    def get_emitter_points_offset(self):
        hidden_delta = 0.015
        visual_delta = 0.015  # 0.00625
        offset_hidden = hidden_delta * (torch.rand_like(self.hidden_emitter_points) - 0.5)
        offset_visual = visual_delta * (torch.rand_like(self.visual_emitter_points) - 0.5)
        return offset_hidden, offset_visual

    @torch.no_grad
    def get_emitter_points_offset_hidden(self):
        hidden_delta = 0.015
        offset_hidden = hidden_delta * (torch.rand_like(self.hidden_emitter_points) - 0.5)
        if self.emitter_points_off_y0:
            offset_hidden[:, 1] = 0.0
        return offset_hidden

    @torch.no_grad
    def get_emitter_points_offset_visual(self):
        visual_delta = 0.015  # 0.00625
        offset_visual = visual_delta * (torch.rand_like(self.visual_emitter_points) - 0.5)
        if self.emitter_points_off_y0:
            offset_visual[:, 1] = 0.0
        return offset_visual

    @torch.no_grad
    def get_emitter_future_first_points_offset(self):
        hidden_delta = 0.015
        visual_delta = 0.015  # 0.00625
        offset_hidden = hidden_delta * (torch.rand_like(self.hidden_emitter_first_points) - 0.5)
        offset_visual = visual_delta * (torch.rand_like(self.visual_emitter_first_points) - 0.5)
        return offset_hidden, offset_visual

    @torch.no_grad
    def emit_new_particles(self, future_time_index=-1):
        # self.visual_timer += self._secs
        # self.hidden_timer += self._secs
        self.emit_counter += 1

        new_visual_pos = []
        new_hidden_pos = []

        if 0 <= future_time_index < 2:
            rand_offset_hidden, rand_offset_visual = self.get_emitter_future_first_points_offset()
            # if it's the first future frame, we add more particles
            cur_new_hidden_pos = (self.hidden_emitter_first_points.clone() + rand_offset_hidden) * self.scale_factor
            cur_new_visual_pos = (self.visual_emitter_first_points.clone() + rand_offset_visual) * self.scale_factor
            new_hidden_pos.append(cur_new_hidden_pos)
            new_visual_pos.append(cur_new_visual_pos)

        else:
            int_ratio_hidden = int(self.emit_ratio_hidden)
            float_ratio_hidden = self.emit_ratio_hidden - int_ratio_hidden
            int_ratio_visual = int(self.emit_ratio_visual)
            float_ratio_visual = self.emit_ratio_visual - int_ratio_visual
            for _ in range(int_ratio_hidden):
                rand_offset_hidden = self.get_emitter_points_offset_hidden()
                cur_new_hidden_pos = (self.hidden_emitter_points.clone() + rand_offset_hidden) * self.scale_factor
                new_hidden_pos.append(cur_new_hidden_pos)

            if float_ratio_hidden > 0:
                rand_offset_hidden = self.get_emitter_points_offset_hidden()
                potential_hidden = (self.hidden_emitter_points.clone() + rand_offset_hidden) * self.scale_factor
                num_hidden = potential_hidden.shape[0]
                selected_hidden = torch.randperm(num_hidden)[: int(float_ratio_hidden * num_hidden)]
                new_hidden_pos.append(potential_hidden[selected_hidden])

            for _ in range(int_ratio_visual):
                rand_offset_visual = self.get_emitter_points_offset_visual()
                cur_new_visual_pos = (self.visual_emitter_points.clone() + rand_offset_visual) * self.scale_factor
                new_visual_pos.append(cur_new_visual_pos)

            if float_ratio_visual > 0:
                rand_offset_visual = self.get_emitter_points_offset_visual()
                potential_visual = (self.visual_emitter_points.clone() + rand_offset_visual) * self.scale_factor
                num_visual = potential_visual.shape[0]
                selected_visual = torch.randperm(num_visual)[: int(float_ratio_visual * num_visual)]
                new_visual_pos.append(potential_visual[selected_visual])

        # else:
        #     rand_offset_hidden, rand_offset_visual = self.get_emitter_points_offset()
        #     while self.hidden_timer >= (1.0 / self.new_hidden_particles_per_sec):
        #         # new_hidden_pos.append(self.hidden_emitter_points.clone() * self.scale_factor)
        #         new_hidden_pos.append((self.hidden_emitter_points.clone() + rand_offset_hidden) * self.scale_factor)
        #         self.hidden_timer -= 1.0 / self.new_hidden_particles_per_sec

        #     while self.visual_timer >= (1.0 / self.new_visual_particles_per_sec):
        #         # new_visual_pos.append(self.visual_emitter_points.clone() * self.scale_factor)
        #         new_visual_pos.append((self.visual_emitter_points.clone() + rand_offset_visual) * self.scale_factor)
        #         self.visual_timer -= 1.0 / self.new_visual_particles_per_sec

        if len(new_hidden_pos) > 0:
            new_hidden_xyz = torch.cat(new_hidden_pos, dim=0)
            self._xyz = torch.cat((self._xyz, new_hidden_xyz), dim=0)
            new_hidden_estimate_xyz = torch.zeros(
                (new_hidden_xyz.shape[0], 3), requires_grad=False, dtype=torch.float, device="cuda"
            )
            self._estimate_xyz = torch.cat((self._estimate_xyz, new_hidden_estimate_xyz), dim=0)
            new_buoyancy = torch.ones(
                (new_hidden_xyz.shape[0], 3), requires_grad=False, dtype=torch.float, device="cuda"
            )
            new_buoyancy *= self._gravity * self.alpha
            self._buoyancy = torch.cat((self._buoyancy, new_buoyancy), dim=0)
            # new_vorticity = torch.zeros((new_hidden_xyz.shape[0], 3), requires_grad=False, dtype=torch.float, device="cuda")
            # self._vorticity = torch.cat((self._vorticity, new_vorticity), dim=0)
            new_hidden_force = torch.zeros(
                (new_hidden_xyz.shape[0], 3), requires_grad=False, dtype=torch.float, device="cuda"
            )
            self._force = torch.cat((self._force, new_hidden_force), dim=0)
            new_hidden_velocity = torch.zeros(
                (new_hidden_xyz.shape[0], 3), requires_grad=False, dtype=torch.float, device="cuda"
            )
            self._velocity = torch.cat((self._velocity, new_hidden_velocity), dim=0)
            new_hidden_imass = torch.ones(
                (new_hidden_xyz.shape[0], 1), requires_grad=False, dtype=torch.float, device="cuda"
            )
            self._imass = torch.cat((self._imass, new_hidden_imass), dim=0)
            self._counts = torch.zeros((self._xyz.shape[0], 1), requires_grad=False, dtype=torch.float, device="cuda")

            new_hidden_particle_id = torch.arange(
                self._particle_id_max, self._particle_id_max + new_hidden_xyz.shape[0], device="cuda"
            ).unsqueeze(1)
            self._particle_id = torch.cat((self._particle_id, new_hidden_particle_id), dim=0)
            self._particle_id_max += new_hidden_xyz.shape[0]

        if len(new_visual_pos) > 0:
            new_visual_xyz = torch.cat(new_visual_pos, dim=0)
            if self._visual_xyz.shape[0] == 0:
                self._visual_xyz = new_visual_xyz
            else:
                self._visual_xyz = torch.cat((self._visual_xyz, new_visual_xyz), dim=0)

    @torch.no_grad
    def guess_hidden_particles(self, stable=False, use_wind=False):
        # during stable iterations, we use smaller secs step and buoyancy
        if stable:
            cur_secs = 0.01
            cur_alpha = -1.0
        else:
            cur_secs = self._secs
            cur_alpha = self.alpha

        self._buoyancy = torch.ones_like(self._buoyancy, dtype=torch.float, device="cuda")
        self._buoyancy *= self._gravity * cur_alpha

        ## the higher particle, the lower buoyancy
        if self.buoyancy_max_y > 0.0:
            scale_max_y = self.buoyancy_max_y * self.scale_factor
            cur_buoyancy_coeff = 1.0 - (self._xyz[:, 1:2] / scale_max_y)
            # cur_buoyancy_coeff = torch.clamp(cur_buoyancy_coeff, 0.0, 1.0)
            cur_buoyancy = self._buoyancy * cur_buoyancy_coeff
        else:
            cur_buoyancy = self._buoyancy

        self._velocity += cur_buoyancy * cur_secs + cur_secs * self._force
        if use_wind:
            cur_particle_y = self._xyz[:, 1:2]
            cur_particle_y_scale = cur_particle_y / self.scale_factor
            cur_wind_force = (cur_particle_y_scale**self.wind_power) * self.wind_force
            cur_wind_force = torch.clamp(cur_wind_force, 0.0, self.wind_force_max)
            self._velocity += cur_wind_force * cur_secs

        ## decay the buoyancy
        if self.buoyancy_decay_rate > 0.0:
            self._buoyancy *= self.buoyancy_decay_rate
        self._force = torch.zeros_like(self._force, dtype=torch.float, device="cuda")
        self._estimate_xyz = self._xyz + cur_secs * self._velocity
        self._counts = torch.zeros_like(self._counts, dtype=torch.float, device="cuda")

    def get_guess_hidden_particles_from_nn(self):
        if self.buoyancy_max_y > 0.0:
            # we dont scale up, as self._estimate_xyz_nn is in smoke value range
            cur_buoyancy_coeff = 1.0 - (self._estimate_xyz_nn[:, 1:2] / self.buoyancy_max_y)
            cur_buoyancy = self._buoyancy * cur_buoyancy_coeff
        else:
            cur_buoyancy = self._buoyancy
        ### NOTE: here we use self._estimate_xyz_nn, not self._xyz
        #  since we simulate the next tick, to add constraints on both self._velocity_nn and self._estimate_xyz_nn
        # estimate_velocity = (
        #     self._velocity_nn * self.scale_factor + cur_buoyancy * self._secs + self._secs * self._force
        # )

        tmp_velocity = (self._estimate_xyz_nn * self.scale_factor - self._xyz) / self._secs
        estimate_velocity = tmp_velocity + cur_buoyancy * self._secs + self._secs * self._force
        estimate_xyz = self._estimate_xyz_nn * self.scale_factor + self._secs * estimate_velocity
        return estimate_xyz

    @torch.no_grad()
    def remove_invalid_particles(self):
        if self.min_neighbors < 0:
            return

        xyz = self._xyz
        N = xyz.shape[0]

        # Use radius_graph to find neighbors within radius H
        edge_index = radius_graph(x=xyz, r=self.H, loop=False)  # Exclude self-loops
        row, _ = edge_index  # We only need the source indices

        # Count the number of neighbors for each particle
        neighbor_counts = torch.bincount(row, minlength=N)

        # Create mask for particles that have at least min_neighbors
        mask = neighbor_counts >= self.min_neighbors

        # Remove particles that do not meet the neighbor count threshold
        if not mask.all():
            self._xyz = self._xyz[mask]
            self._estimate_xyz = self._estimate_xyz[mask]
            self._buoyancy = self._buoyancy[mask]
            self._force = self._force[mask]
            self._velocity = self._velocity[mask]
            self._imass = self._imass[mask]
            self._counts = self._counts[mask]
            self._particle_id = self._particle_id[mask]

    def update_solver_counts(self):
        self._counts += 1.0

    @torch.no_grad()
    def project_gas_constraints(self):
        if self.record_time:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            # Start timing
            start_event.record()

        N = self._estimate_xyz.shape[0]
        exyz = self._estimate_xyz

        # Compute edge_index including self-loops
        edge_index = radius_graph(exyz, r=self.H, loop=True, max_num_neighbors=self.KNN_K)
        row, col = edge_index  # row: source nodes, col: target nodes

        # Create masks to include or exclude self-loops
        self_loop_mask = row == col  # Mask for self-loops
        non_self_loop_mask = row != col  # Mask for non-self-loops

        # Compute differences and squared distances for all edges
        diff = exyz[row] - exyz[col]  # Shape: (E, 3)
        dist2 = torch.sum(diff**2, dim=1)  # Shape: (E,)

        # Compute poly6 values for all edges (including self-loops)
        poly6_values = self.poly6(dist2)  # Shape: (E,)

        # Compute pi per node (include self-loops)
        pi = torch.zeros(N, device=exyz.device)
        pi.index_add_(0, row, poly6_values)
        pi = pi.unsqueeze(1) / self._imass  # Shape: (N, 1)

        # Compute neighbor counts including self-loops
        neighbors_len = torch.bincount(row, minlength=N).unsqueeze(1).float()  # Shape: (N, 1)

        # Select edges excluding self-loops for further computations
        row_ns = row[non_self_loop_mask]
        col_ns = col[non_self_loop_mask]
        diff_ns = diff[non_self_loop_mask]
        dist2_ns = dist2[non_self_loop_mask]

        # Compute spiky gradients
        rlen_ns = torch.sqrt(dist2_ns + self.EPSILON)  # Shape: (E_ns,)
        spiky_grads = self.spiky_grad(diff_ns, rlen_ns)  # Shape: (E_ns, 3)

        # Compute gr per node
        gr = torch.zeros(N, 3, device=exyz.device)
        gr.index_add_(0, row_ns, spiky_grads)
        gr = gr / self.p0  # Shape: (N, 3)
        gr_dot = torch.sum(gr**2, dim=1)  # Shape: (N,)

        # Compute grad_dot per node
        grad_sq = (spiky_grads / self.p0) ** 2  # Shape: (E_ns, 3)
        grad_sq_sum = torch.sum(grad_sq, dim=1)  # Shape: (E_ns,)
        grad_dot = torch.zeros(N, device=exyz.device)
        grad_dot.index_add_(0, row_ns, grad_sq_sum)

        denom = (grad_dot + gr_dot).unsqueeze(1)  # Shape: (N, 1)

        # Compute pressure ratios and force corrections
        p_ratio = pi / self.p0  # Shape: (N, 1)
        force_delta = self._velocity * (1.0 - p_ratio) * -self.k  # Shape: (N, 3)
        self._force += force_delta

        # Compute lambdas
        lambdas = -(p_ratio - 1.0) / (denom + self.RELAXATION)  # Shape: (N, 1)

        # Compute lambda corrections
        poly6_values_ns = poly6_values[non_self_loop_mask]  # Shape: (E_ns,)
        lamb_corr = -self.K_P * (poly6_values_ns / self.lamb_corr_denom) ** self.E_P  # Shape: (E_ns,)

        # Gather lambdas of neighboring particles
        lambdas_row = lambdas[row_ns].squeeze(1)  # Shape: (E_ns,)
        lambdas_col = lambdas[col_ns].squeeze(1)  # Shape: (E_ns,)

        # Compute sum of lambdas
        lambdas_sum = lambdas_row + lambdas_col  # Shape: (E_ns,)

        # Compute position deltas
        deltas = ((lambdas_sum + lamb_corr).unsqueeze(-1)) * spiky_grads  # Shape: (E_ns, 3)

        # Compute deltas_sum per node
        deltas_sum = torch.zeros(N, 3, device=exyz.device)
        deltas_sum.index_add_(0, row_ns, deltas)
        deltas_sum = deltas_sum / self.p0  # Shape: (N, 3)

        # Compute estimate_xyz_delta_candidate
        estimate_xyz_delta_candidate = deltas_sum / (neighbors_len + self._counts)  # Shape: (N, 3)

        # Apply corrections
        self._estimate_xyz += estimate_xyz_delta_candidate

        if self.record_time:
            end_event.record()
            # Wait for everything to finish
            torch.cuda.synchronize()
            # Calculate elapsed time
            elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds
        else:
            elapsed_time = 0.0

        # Prepare return values for debugging or logging
        return_values = {
            "velocity": self._velocity.detach().clone().mean().item(),
            "xyz": self._xyz.detach().clone().mean().item(),
            "estimate_xyz": self._estimate_xyz.detach().clone().mean().item(),
            "diff": diff.detach().clone().mean().item(),
            "dist2": dist2.detach().clone().mean().item(),
            "poly6_values": poly6_values.detach().clone().mean().item(),
            "pi": pi.detach().clone().mean().item(),
            "rlen": rlen_ns.detach().clone().mean().item(),
            "spiky_grads": spiky_grads.detach().clone().mean().item(),
            "gr": gr.detach().clone().mean().item(),
            "gr_dot": gr_dot.detach().clone().mean().item(),
            "grad_dot": grad_dot.detach().clone().mean().item(),
            "denom": denom.detach().clone().mean().item(),
            "p_ratio": p_ratio.detach().clone().mean().item(),
            "force_delta": force_delta.detach().clone().mean().item(),
            "lambdas": lambdas.detach().clone().mean().item(),
            "lamb_corr": lamb_corr.detach().clone().mean().item(),
            "deltas": deltas.detach().clone().mean().item(),
            "estimate_xyz_delta": estimate_xyz_delta_candidate.detach().clone().mean().item(),
            "elapsed_time": elapsed_time,
        }

        return return_values

    @torch.no_grad
    def check_inside_rigid_body(self, xyz):
        if self.rigid_body == "cuboid":
            x_num, y_num, z_num = self.rigid_cuboid_num
            diam = self.rigid_particle_diameter
            edge_size = torch.tensor([x_num * diam, y_num * diam, z_num * diam]).float().cuda()
            half_edge = edge_size / 2.0
            lower = self.rigid_body_center - half_edge
            upper = self.rigid_body_center + half_edge
            inside_points_mask = torch.all((xyz >= lower) & (xyz <= upper), dim=1)
        elif self.rigid_body == "sphere":
            radius = self.rigid_sphere_radius
            center = self.rigid_body_center
            dist = torch.linalg.norm(xyz - center, dim=1)
            inside_points_mask = dist <= radius
        elif self.rigid_body == "cylinder":
            radius = self.rigid_cylinder_radius
            num_height = self.rigid_cylinder_num[1]
            diam = self.rigid_particle_diameter
            height = num_height * diam
            center = self.rigid_body_center
            x_points, y_points, z_points = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            # Extract the cylinder's center coordinates
            x_center, y_center, z_center = center[0], center[1], center[2]
            # Horizontal check: distance from the center in the xy-plane
            dist_xy_squared = (x_points - x_center) ** 2 + (y_points - y_center) ** 2
            # Condition 1: Points should be within the radius in the xy-plane
            within_radius = dist_xy_squared <= radius**2
            # Condition 2: Points should be within the height range in the z direction
            within_height = (z_points >= z_center - height / 2) & (z_points <= z_center + height / 2)
            # Combine both conditions
            inside_points_mask = within_radius & within_height
        return inside_points_mask

    @torch.no_grad()
    def project_rigid_body_constraints(self):
        N = self._estimate_xyz.shape[0]
        M = self._rigid_xyz.shape[0]
        xyz = self._estimate_xyz
        rigid_xyz = self._rigid_xyz

        # Check if particles are inside the rigid body
        mask_inside = self.check_inside_rigid_body(xyz).squeeze(1)  # Shape: (N,)
        if mask_inside.sum() == 0:
            return {}

        # Select particles inside the rigid body
        xyz_inside = xyz[mask_inside]  # Shape: (N_inside, 3)

        # Set a sufficiently large radius to ensure we find nearest rigid body points
        r = self.H  # You can adjust this radius as needed

        # Compute radius graph between particles inside the rigid body and rigid body points
        edge_index = radius(rigid_xyz, xyz_inside, r=r, max_num_neighbors=None)
        if edge_index.size(1) == 0:
            return {}

        row = edge_index[0]  # Indices in xyz_inside
        col = edge_index[1]  # Indices in rigid_xyz

        # Compute differences and squared distances
        diff = xyz_inside[row] - rigid_xyz[col]  # Shape: (E, 3)
        dist2 = torch.sum(diff**2, dim=1)  # Shape: (E,)

        # Find the nearest rigid body point for each particle inside
        min_dist2, argmin = scatter_min(dist2, row, dim=0)
        nearest_rigid_idx = col[argmin]  # Shape: (N_inside,)

        # Compute the difference vectors to the nearest rigid body points
        diff_min = xyz_inside - rigid_xyz[nearest_rigid_idx]  # Shape: (N_inside, 3)

        # Apply corrections
        dp = diff_min  # Shape: (N_inside, 3)
        dp1 = -dp
        self._estimate_xyz[mask_inside] += dp1

        return_values = {
            "mask": mask_inside.float().mean().item(),
            "dp": dp.mean().item(),
            "dp1": dp1.mean().item(),
        }

        return return_values

    def get_gas_constraints_from_exyz_nn(self):
        N = self._xyz.shape[0]

        # Scale positions
        scaled_estimate_xyz = self._estimate_xyz_nn * self.scale_factor  # Shape: (N, 3)

        # Use radius_graph to find neighbors within radius H
        edge_index = radius_graph(scaled_estimate_xyz, r=self.H, loop=True, max_num_neighbors=self.KNN_K)
        row, col = edge_index

        # Compute squared distances
        diff = scaled_estimate_xyz[row] - scaled_estimate_xyz[col]  # Shape: (E, 3)
        dist2 = torch.sum(diff**2, dim=1)  # Shape: (E,)

        # Compute poly6 values
        poly6_values = self.poly6(dist2)  # Shape: (E,)

        # Sum over neighbors for each particle, then divide by mass
        pi = torch.zeros(N, device=scaled_estimate_xyz.device)
        pi.index_add_(0, row, poly6_values)
        pi = pi.unsqueeze(1) / self._imass  # Shape: (N, 1)

        # Compute pressure ratio
        p_ratio = pi / self.p0  # Shape: (N, 1)

        return p_ratio

    def get_gas_constraints_from_vel_nn_guess(self):
        # Get the estimated positions of hidden particles
        vel_guess_estimate_xyz = self.get_guess_hidden_particles_from_nn()  # Shape: (N, 3)
        N = self._xyz.shape[0]

        # Use radius_graph to find neighbors within radius H
        edge_index = radius_graph(vel_guess_estimate_xyz, r=self.H, loop=True, max_num_neighbors=self.KNN_K)
        row, col = edge_index

        # Compute squared distances
        diff = vel_guess_estimate_xyz[row] - vel_guess_estimate_xyz[col]  # Shape: (E, 3)
        dist2 = torch.sum(diff**2, dim=1)  # Shape: (E,)

        # Compute poly6 values
        poly6_values = self.poly6(dist2)  # Shape: (E,)

        # Sum over neighbors for each particle, then divide by mass
        pi = torch.zeros(N, device=vel_guess_estimate_xyz.device)
        pi.index_add_(0, row, poly6_values)
        pi = pi.unsqueeze(1) / self._imass  # Shape: (N, 1)

        # Compute p_ratio
        p_ratio = pi / self.p0  # Shape: (N, 1)

        return p_ratio

    @torch.no_grad
    def confirm_guess_hidden_particles(self):
        self._velocity = (self._estimate_xyz - self._xyz) / self._secs

        # Calculate the lengths between corresponding points in ps and eps
        ep_lengths = torch.norm(self._estimate_xyz - self._xyz, dim=1)

        # Create a mask where the length is less than self.EPSILON
        mask = ep_lengths < self.EPSILON

        # Set vs elements to zeros where the mask is True
        self._velocity[mask] = torch.zeros(3).float().cuda()

        # Update ps elements where the mask is False
        self._xyz[~mask] = self._estimate_xyz[~mask]

    @torch.no_grad
    def confirm_guess_hidden_particles_wo_velocity(self):
        self._velocity = (self._estimate_xyz - self._xyz) / self._secs

        # Calculate the lengths between corresponding points in ps and eps
        ep_lengths = torch.norm(self._estimate_xyz - self._xyz, dim=1)

        # Create a mask where the length is less than self.EPSILON
        mask = ep_lengths < self.EPSILON

        # Set vs elements to zeros where the mask is True
        self._velocity[mask] = torch.zeros(3).float().cuda()

        # Update ps elements where the mask is False
        self._xyz[~mask] = self._estimate_xyz[~mask]

    @torch.no_grad
    def confirm_guess_hidden_particles_from_nn(self):
        self._estimate_xyz = self._estimate_xyz_nn.detach().clone().requires_grad_(False) * self.scale_factor
        # self._velocity = self._velocity_nn.detach().clone().requires_grad_(False) * self.scale_factor

    @torch.no_grad()
    def update_visual_particles(self):
        if self._visual_xyz.shape[0] == 0:
            return

        V = self._visual_xyz.shape[0]
        N = self._estimate_xyz.shape[0]

        # Compute edges between visual particles and estimated particles within radius H
        edge_index = radius(
            x=self._estimate_xyz,
            y=self._visual_xyz,
            r=self.H,
            max_num_neighbors=self.KNN_K,
        )
        row = edge_index[0]  # Indices in visual particles
        col = edge_index[1]  # Indices in estimated particles

        # Compute squared distances
        diff = self._visual_xyz[row] - self._estimate_xyz[col]
        dist2 = torch.sum(diff**2, dim=1)

        # Compute poly6 values
        p6 = self.poly6(dist2)

        # Get velocity of the estimated particles at the indices
        velocity = self._velocity[col]

        # Compute weighted sum of velocity
        weighted_velocity = velocity * p6.unsqueeze(-1)
        visual_velocity = torch.zeros(V, 3, device=self._visual_xyz.device)
        visual_velocity.index_add_(0, row, weighted_velocity)

        # Sum of p6 values
        sum_p6 = torch.zeros(V, device=self._visual_xyz.device)
        sum_p6.index_add_(0, row, p6)
        sum_p6 = sum_p6.clamp_min(self.EPSILON)

        # Compute delta for updating visual particle positions
        estimate_visual_xyz_delta = visual_velocity * self._secs / sum_p6.unsqueeze(-1)

        # Update visual positions
        self._visual_xyz += estimate_visual_xyz_delta

    @torch.no_grad()
    def project_rigid_body_constraints_for_visual_particles(self):
        N = self._visual_xyz.shape[0]
        if N == 0:
            return {}

        # Check which visual particles are inside the rigid body
        mask_inside = self.check_inside_rigid_body(self._visual_xyz).squeeze(1)
        if mask_inside.sum() == 0:
            return {}

        # Select particles inside the rigid body
        visual_xyz_inside = self._visual_xyz[mask_inside]

        # Compute radius graph between visual particles inside and rigid body particles
        edge_index = radius(
            x=self._rigid_xyz,
            y=visual_xyz_inside,
            r=self.H,
        )
        if edge_index.size(1) == 0:
            return {}

        row = edge_index[0]
        col = edge_index[1]

        # Compute differences and squared distances
        diff = visual_xyz_inside[row] - self._rigid_xyz[col]
        dist2 = torch.sum(diff**2, dim=1)

        # Find the nearest rigid body point for each particle inside
        min_dist2, argmin = scatter_min(dist2, row, dim=0)
        nearest_rigid_idx = col[argmin]

        # Compute the difference vectors to the nearest rigid body points
        diff_min = visual_xyz_inside - self._rigid_xyz[nearest_rigid_idx]

        # Apply corrections
        dp = diff_min
        dp1 = -dp
        self._visual_xyz[mask_inside] += dp1

        return_values = {
            "mask": mask_inside.float().mean().item(),
            "dp": dp.detach().mean().item(),
            "dp1": dp1.detach().mean().item(),
        }

        return return_values

    def get_visual_xyz_from_nn(self):
        visual_xyz = self._visual_xyz.detach()  # Shape: (V, 3)
        estimate_xyz_nn = self._estimate_xyz_nn * self.scale_factor  # Shape: (N, 3)
        # velocity_nn = self._velocity_nn * self.scale_factor  # Shape: (N, 3)
        velocity_nn = (estimate_xyz_nn - self._xyz) / self._secs

        V = visual_xyz.shape[0]
        N = estimate_xyz_nn.shape[0]

        # Use radius_graph to find neighbors within radius H
        edge_index = radius(
            x=estimate_xyz_nn,
            y=visual_xyz,
            r=self.H,
            max_num_neighbors=self.KNN_K,
        )
        row = edge_index[0]  # Indices in visual particles
        col = edge_index[1]  # Indices in estimated particles

        # Compute squared distances
        diff = visual_xyz[row] - estimate_xyz_nn[col]  # Shape: (E, 3)
        dist2 = torch.sum(diff**2, dim=1)  # Shape: (E,)

        # Compute poly6 values
        p6 = self.poly6(dist2)  # Shape: (E,)

        # Get velocity of the estimated particles at the indices
        velocity = velocity_nn[col]  # Shape: (E, 3)

        # Compute weighted sum of velocity
        weighted_velocity = velocity * p6.unsqueeze(-1)  # Shape: (E, 3)
        visual_velocity = torch.zeros(V, 3, device=visual_xyz.device)
        visual_velocity.index_add_(0, row, weighted_velocity)  # Sum over neighbors

        # Sum of p6 values
        sum_p6 = torch.zeros(V, device=visual_xyz.device)
        sum_p6.index_add_(0, row, p6)
        sum_p6 = sum_p6.clamp_min(self.EPSILON)  # Avoid division by zero

        # Compute delta for updating visual particle positions
        estimate_visual_xyz_delta = visual_velocity * self._secs / sum_p6.unsqueeze(-1)  # Shape: (V, 3)

        # Update visual positions
        estimate_visual_xyz = visual_xyz + estimate_visual_xyz_delta

        return estimate_visual_xyz

    @torch.no_grad
    def update_visual_xyz_from_nn(self):
        self._visual_xyz = self.get_visual_xyz_from_nn().detach().clone().requires_grad_(False)

    def get_visual_xyz_from_hidden_guess(self):
        visual_xyz = self._visual_xyz  # Shape: (V, 3)
        estimate_xyz = self._estimate_xyz  # Shape: (N, 3)
        velocity = self._velocity  # Shape: (N, 3)
        V = visual_xyz.shape[0]

        # Use radius graph to find neighbors within radius H
        edge_index = radius(
            x=estimate_xyz,
            y=visual_xyz,
            r=self.H,
            max_num_neighbors=self.KNN_K,
        )
        row = edge_index[0]  # Indices in visual particles
        col = edge_index[1]  # Indices in estimated particles

        # Compute squared distances
        diff = visual_xyz[row] - estimate_xyz[col]
        dist2 = torch.sum(diff**2, dim=1)

        # Compute poly6 values
        p6 = self.poly6(dist2)

        # Retrieve velocity of the estimated particles at the indices
        velocity_knn = velocity[col]

        # Compute weighted sum of velocity
        weighted_velocity = velocity_knn * p6.unsqueeze(-1)
        visual_velocity = torch.zeros(V, 3, device=visual_xyz.device)
        visual_velocity.index_add_(0, row, weighted_velocity)

        # Sum of p6 values
        sum_p6 = torch.zeros(V, device=visual_xyz.device)
        sum_p6.index_add_(0, row, p6)
        sum_p6 = sum_p6.clamp_min(self.EPSILON)

        # Compute delta for updating visual particle positions
        estimate_visual_xyz_delta = visual_velocity * self._secs / sum_p6.unsqueeze(-1)

        # Update visual positions
        estimate_visual_xyz = visual_xyz + estimate_visual_xyz_delta

        return estimate_visual_xyz

    @torch.no_grad
    def re_simulation_setup(self):
        self._re_sim_xyz = torch.empty(0)
        self._re_sim_velocity = torch.empty(0)
        self._re_sim_visual_xyz = torch.empty(0)
        self._re_sim_particle_id = torch.empty(0)

    @torch.no_grad
    def re_simulation_advect_particles(self):
        self._re_sim_velocity = self._velocity.clone().cuda()
        if self._re_sim_xyz.shape[0] == 0:
            self._re_sim_xyz = self._xyz.clone().cuda()
            self._re_sim_particle_id = self._particle_id.clone().cuda()
            self._re_sim_visual_xyz = self._visual_xyz.clone().cuda()
            self._re_sim_visual_velocity = torch.zeros_like(self._re_sim_visual_xyz, dtype=torch.float, device="cuda")

            re_sim_xyz_diff = 0.0
            re_sim_visual_xyz_diff = 0.0

        else:
            # Remove the particles that are not in the current simulation
            mask_good = torch.isin(self._re_sim_particle_id, self._particle_id).squeeze(1)
            mask_previous = torch.isin(self._particle_id, self._re_sim_particle_id).squeeze(1)
            mask_new = ~mask_previous

            self._re_sim_xyz = self._re_sim_xyz[mask_good]
            self._re_sim_particle_id = self._re_sim_particle_id[mask_good]

            _velocity_previous = self._velocity[mask_previous]
            self._re_sim_xyz += self._secs * _velocity_previous

            new_xyz = self._xyz[mask_new]
            new_particle_id = self._particle_id[mask_new]
            self._re_sim_xyz = torch.cat((self._re_sim_xyz, new_xyz), dim=0)
            self._re_sim_particle_id = torch.cat((self._re_sim_particle_id, new_particle_id), dim=0)

            re_sim_xyz_diff = torch.abs(self._re_sim_xyz - self._xyz).mean().item()

            self.re_simulation_update_visual_xyz()

            # directly append the new particles to the visual particles
            new_visual_xyz_num = self._visual_xyz.shape[0] - self._re_sim_visual_xyz.shape[0]
            if new_visual_xyz_num > 0:
                new_visual_xyz = self._visual_xyz[-new_visual_xyz_num:]
                self._re_sim_visual_xyz = torch.cat((self._re_sim_visual_xyz, new_visual_xyz), dim=0)

            re_sim_visual_xyz_diff = torch.abs(self._re_sim_visual_xyz - self._visual_xyz).mean().item()

        return re_sim_xyz_diff, re_sim_visual_xyz_diff

    @torch.no_grad
    def re_simulation_update_visual_xyz(self):
        visual_xyz = self._re_sim_visual_xyz  # Shape: (V, 3)
        estimate_xyz = self._re_sim_xyz  # Shape: (N, 3)
        velocity = self._re_sim_velocity  # Shape: (N, 3)
        V = visual_xyz.shape[0]

        # Use radius graph to find neighbors within radius H
        edge_index = radius(
            x=estimate_xyz,
            y=visual_xyz,
            r=self.H,
            max_num_neighbors=self.KNN_K,
        )
        row = edge_index[0]  # Indices in visual particles
        col = edge_index[1]  # Indices in estimated particles

        # Compute squared distances
        diff = visual_xyz[row] - estimate_xyz[col]
        dist2 = torch.sum(diff**2, dim=1)

        # Compute poly6 values
        p6 = self.poly6(dist2)

        # Retrieve velocity of the estimated particles at the indices
        velocity_knn = velocity[col]

        # Compute weighted sum of velocity
        weighted_velocity = velocity_knn * p6.unsqueeze(-1)
        visual_velocity = torch.zeros(V, 3, device=visual_xyz.device)
        visual_velocity.index_add_(0, row, weighted_velocity)

        # Sum of p6 values
        sum_p6 = torch.zeros(V, device=visual_xyz.device)
        sum_p6.index_add_(0, row, p6)
        sum_p6 = sum_p6.clamp_min(self.EPSILON)

        # Compute delta for updating visual particle positions
        estimate_visual_xyz_delta = visual_velocity * self._secs / sum_p6.unsqueeze(-1)

        # Update visual positions
        self._re_sim_visual_velocity = visual_velocity
        self._re_sim_visual_xyz = visual_xyz + estimate_visual_xyz_delta

    @torch.no_grad
    def re_simulation_get_visual_xyz_delta(self, xyz, visual_xyz, velocity):
        V = visual_xyz.shape[0]

        # Use radius graph to find neighbors within radius H
        edge_index = radius(
            x=xyz,
            y=visual_xyz,
            r=self.H,
            max_num_neighbors=self.KNN_K,
        )
        row = edge_index[0]  # Indices in visual particles
        col = edge_index[1]  # Indices in estimated particles

        # Compute squared distances
        diff = visual_xyz[row] - xyz[col]
        dist2 = torch.sum(diff**2, dim=1)

        # Compute poly6 values
        p6 = self.poly6(dist2)

        # Retrieve velocity of the estimated particles at the indices
        velocity_knn = velocity[col]

        # Compute weighted sum of velocity
        weighted_velocity = velocity_knn * p6.unsqueeze(-1)
        visual_velocity = torch.zeros(V, 3, device=visual_xyz.device)
        visual_velocity.index_add_(0, row, weighted_velocity)

        # Sum of p6 values
        sum_p6 = torch.zeros(V, device=visual_xyz.device)
        sum_p6.index_add_(0, row, p6)
        sum_p6 = sum_p6.clamp_min(self.EPSILON)

        # Compute delta for updating visual particle positions
        estimate_visual_xyz_delta = visual_velocity * self._secs / sum_p6.unsqueeze(-1)

        # Update visual positions
        new_visual_xyz = visual_xyz + estimate_visual_xyz_delta
        return new_visual_xyz

    def prepare_hidden_particles_for_rendering(self):
        assert self._xyz.shape[0] > 0, "No hidden particles to render"
        self._color_dummy = torch.zeros((self._xyz.shape[0], 1), dtype=torch.float, device="cuda") + 0.7
        self._scales_dummy = torch.zeros((self._xyz.shape[0], 3), dtype=torch.float, device="cuda") - 5.9
        self._rotation_dummy = torch.zeros((self._xyz.shape[0], 4), dtype=torch.float, device="cuda")
        self._rotation_dummy[:, 0] = 1.0
        self._opacity_dummy = inv_sigmoid(0.1 * torch.ones((self._xyz.shape[0], 1), dtype=torch.float, device="cuda"))

    def prepare_visual_particles_for_rendering(self):
        assert self._visual_xyz.shape[0] > 0, "No visual particles to render"
        self._visual_color = torch.zeros((self._visual_xyz.shape[0], 1), dtype=torch.float, device="cuda") + 0.7
        self._visual_scales = torch.zeros((self._visual_xyz.shape[0], 3), dtype=torch.float, device="cuda") - 5.9
        self._visual_rotation = torch.zeros((self._visual_xyz.shape[0], 4), dtype=torch.float, device="cuda")
        self._visual_rotation[:, 0] = 1.0
        opacity = inv_sigmoid(0.1 * torch.ones((self._visual_xyz.shape[0], 1), dtype=torch.float, device="cuda"))
        self._visual_opacity = opacity

    def prepare_future_visual_particles_for_rendering(self, use_level_two_future=False):
        if use_level_two_future:
            prev_num_visual = self._visual_color.shape[0]
            cur_num_visual = self._visual_xyz.shape[0]
            new_num_visual = cur_num_visual - prev_num_visual
            new_visual_color = torch.zeros((new_num_visual, 1), dtype=torch.float, device="cuda") + 0.7
            new_visual_scales = torch.zeros((new_num_visual, 3), dtype=torch.float, device="cuda") - 5.9
            new_visual_rotation = torch.zeros((new_num_visual, 4), dtype=torch.float, device="cuda")
            new_visual_rotation[:, 0] = 1.0
            new_opacity = inv_sigmoid(0.1 * torch.ones((new_num_visual, 1), dtype=torch.float, device="cuda"))
            self._visual_color = torch.cat((self._visual_color, new_visual_color), dim=0)
            self._visual_scales = torch.cat((self._visual_scales, new_visual_scales), dim=0)
            self._visual_rotation = torch.cat((self._visual_rotation, new_visual_rotation), dim=0)
            self._visual_opacity = torch.cat((self._visual_opacity, new_opacity), dim=0)

        else:
            self.prepare_visual_particles_for_rendering()

    def prepare_rigid_body_particles_for_rendering(self):
        assert self._rigid_xyz.shape[0] > 0, "No rigid body particles to render"
        self._rigid_color = torch.zeros((self._rigid_xyz.shape[0], 1), dtype=torch.float, device="cuda") + 0.9
        self._rigid_scales = torch.zeros((self._rigid_xyz.shape[0], 3), dtype=torch.float, device="cuda") - 5.5
        self._rigid_rotation = torch.zeros((self._rigid_xyz.shape[0], 4), dtype=torch.float, device="cuda")
        self._rigid_rotation[:, 0] = 1.0
        opacity = inv_sigmoid(0.3 * torch.ones((self._rigid_xyz.shape[0], 1), dtype=torch.float, device="cuda"))
        self._rigid_opacity = opacity

    @torch.no_grad
    def save_particles_rigid_body(self, quantities_path, frame_idx):
        mkdir_p(quantities_path)

        rigid_xyz_path = os.path.join(quantities_path, f"frame_{frame_idx:03d}_rigid_xyz.npy")
        rigid_xyz_numpy = self._rigid_xyz.detach().clone().cpu().numpy() / self.scale_factor
        np.save(rigid_xyz_path, rigid_xyz_numpy)

    @torch.no_grad
    def save_particles_frame(self, quantities_path, frame_idx):
        mkdir_p(quantities_path)

        xyz_path = os.path.join(quantities_path, f"frame_{frame_idx:03d}_xyz.npy")
        xyz_numpy = self._xyz.detach().clone().cpu().numpy() / self.scale_factor
        np.save(xyz_path, xyz_numpy)

        if self._visual_xyz.shape[0] > 0:
            visual_xyz_path = os.path.join(quantities_path, f"frame_{frame_idx:03d}_visual_xyz.npy")
            visual_xyz_numpy = self._visual_xyz.detach().clone().cpu().numpy() / self.scale_factor
            np.save(visual_xyz_path, visual_xyz_numpy)

    @torch.no_grad
    def save_particles_simulation(self, quantities_path, index):
        mkdir_p(quantities_path)

        xyz_path = os.path.join(quantities_path, f"{index:03d}_xyz.npy")
        xyz_numpy = self._xyz.detach().clone().cpu().numpy() / self.scale_factor
        np.save(xyz_path, xyz_numpy)

        estimated_xyz_path = os.path.join(quantities_path, f"{index:03d}_estimated_xyz.npy")
        estimated_xyz_numpy = self._estimate_xyz.detach().clone().cpu().numpy() / self.scale_factor
        np.save(estimated_xyz_path, estimated_xyz_numpy)

        if self._visual_xyz.shape[0] > 0:
            visual_xyz_path = os.path.join(quantities_path, f"{index:03d}_visual_xyz.npy")
            visual_xyz_numpy = self._visual_xyz.detach().clone().cpu().numpy() / self.scale_factor
            np.save(visual_xyz_path, visual_xyz_numpy)

    @torch.no_grad
    def save_particles_simulation_guess(self, quantities_path, index):
        mkdir_p(quantities_path)

        estimated_xyz_path = os.path.join(quantities_path, f"{index:03d}_guess_estimated_xyz.npy")
        estimated_xyz_numpy = self._estimate_xyz.detach().clone().cpu().numpy() / self.scale_factor
        np.save(estimated_xyz_path, estimated_xyz_numpy)

    @torch.no_grad
    def save_particles_optimization_first(self, quantities_path, frame_idx, iteration):
        # frame_idx == 0
        mkdir_p(quantities_path)

        visual_xyz_path = os.path.join(quantities_path, f"{frame_idx:03d}_{iteration:05d}_visual_xyz.npy")
        visual_xyz_numpy = self._visual_xyz.detach().clone().cpu().numpy()
        np.save(visual_xyz_path, visual_xyz_numpy)

    @torch.no_grad
    def save_particles_optimization(self, quantities_path, visual_xyz, frame_idx, iteration):
        mkdir_p(quantities_path)

        estimate_xyz_nn_path = os.path.join(quantities_path, f"{frame_idx:03d}_{iteration:05d}_estimate_xyz_nn.npy")
        estimate_xyz_nn_numpy = self._estimate_xyz_nn.detach().clone().cpu().numpy()  # estimate_xyz_nn is not scaled
        np.save(estimate_xyz_nn_path, estimate_xyz_nn_numpy)

        if visual_xyz.shape[0] > 0:
            visual_xyz_path = os.path.join(quantities_path, f"{frame_idx:03d}_{iteration:05d}_visual_xyz.npy")
            visual_xyz_numpy = visual_xyz.detach().clone().cpu().numpy()
            np.save(visual_xyz_path, visual_xyz_numpy)

    @torch.no_grad
    def save_particles_optimization_level_two(self, quantities_path, frame_idx, iteration):
        mkdir_p(quantities_path)

        visual_color_path = os.path.join(quantities_path, f"{frame_idx:03d}_{iteration:05d}_visual_color.npy")
        visual_color_numpy = self._visual_color.detach().clone().cpu().numpy()
        np.save(visual_color_path, visual_color_numpy)

        visual_scales_path = os.path.join(quantities_path, f"{frame_idx:03d}_{iteration:05d}_visual_scales.npy")
        visual_scales_numpy = self._visual_scales.detach().clone().cpu().numpy()
        np.save(visual_scales_path, visual_scales_numpy)

        visual_rotation_path = os.path.join(quantities_path, f"{frame_idx:03d}_{iteration:05d}_visual_rotation.npy")
        visual_rotation_numpy = self._visual_rotation.detach().clone().cpu().numpy()
        np.save(visual_rotation_path, visual_rotation_numpy)

        visual_opacity_path = os.path.join(quantities_path, f"{frame_idx:03d}_{iteration:05d}_visual_opacity.npy")
        visual_opacity_numpy = self._visual_opacity.detach().clone().cpu().numpy()
        np.save(visual_opacity_path, visual_opacity_numpy)

    @torch.no_grad
    def save_hidden(self, checkpoint_path, frame_idx):
        mkdir_p(checkpoint_path)

        # we only scale down the particles' xyz for saving
        xyz_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_xyz.npy")
        xyz_numpy = self._xyz.detach().clone().cpu().numpy() / self.scale_factor
        np.save(xyz_path, xyz_numpy)

        estimate_xyz_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_estimate_xyz.npy")
        estimate_xyz_numpy = self._estimate_xyz.detach().clone().cpu().numpy() / self.scale_factor
        np.save(estimate_xyz_path, estimate_xyz_numpy)

        buoyancy_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_buoyancy.npy")
        buoyancy_numpy = self._buoyancy.detach().clone().cpu().numpy()
        np.save(buoyancy_path, buoyancy_numpy)

        force_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_force.npy")
        force_numpy = self._force.detach().clone().cpu().numpy()
        np.save(force_path, force_numpy)

        velocity_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_velocity.npy")
        velocity_numpy = self._velocity.detach().clone().cpu().numpy()
        np.save(velocity_path, velocity_numpy)

        i_mass_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_imass.npy")
        i_mass_numpy = self._imass.detach().clone().cpu().numpy()
        np.save(i_mass_path, i_mass_numpy)

        counts_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_counts.npy")
        counts_numpy = self._counts.detach().clone().cpu().numpy()
        np.save(counts_path, counts_numpy)

        gravity_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_gravity.npy")
        gravity_numpy = self._gravity.detach().clone().cpu().numpy()
        np.save(gravity_path, gravity_numpy)

        particle_id_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_particle_id.npy")
        particle_id_numpy = self._particle_id.detach().clone().cpu().numpy()
        np.save(particle_id_path, particle_id_numpy)

        scalar_values_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_scalar_values.json")
        scalar_values = {
            "scale_factor": self.scale_factor,
            "secs": self._secs,
            "alpha": self.alpha,
            "k": self.k,
            "p0": self.p0,
            "buoyancy_decay_rate": self.buoyancy_decay_rate,
            "buoyancy_max_y": self.buoyancy_max_y,
            "min_neighbors": self.min_neighbors,
            "remove_out_boundary": self.remove_out_boundary,
            # "new_visual_particles_per_sec": self.new_visual_particles_per_sec,
            # "new_hidden_particles_per_sec": self.new_hidden_particles_per_sec,
            # "visual_timer": self.visual_timer,
            # "hidden_timer": self.hidden_timer,
            "emit_ratio_hidden": self.emit_ratio_hidden,
            "emit_ratio_visual": self.emit_ratio_visual,
            "emit_counter": self.emit_counter,
            "total_iterations": self.total_iterations,
            "total_sim_iterations": self.total_sim_iterations,
            "total_tb_log_iterations": self.total_tb_log_iterations,
            "particle_id_max": self._particle_id_max,
        }
        with open(scalar_values_path, "w") as f:
            json.dump(scalar_values, f)

    @torch.no_grad
    def save_visual(self, checkpoint_path, frame_idx, scale=True):
        mkdir_p(checkpoint_path)

        visual_xyz_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_visual_xyz.npy")
        visual_xyz_numpy = self._visual_xyz.detach().clone().cpu().numpy()
        if scale:
            visual_xyz_numpy = visual_xyz_numpy / self.scale_factor
        np.save(visual_xyz_path, visual_xyz_numpy)

        visual_color_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_visual_color.npy")
        visual_color_numpy = self._visual_color.detach().clone().cpu().numpy()
        np.save(visual_color_path, visual_color_numpy)

        visual_scales_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_visual_scales.npy")
        visual_scales_numpy = self._visual_scales.detach().clone().cpu().numpy()
        np.save(visual_scales_path, visual_scales_numpy)

        visual_rotation_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_visual_rotation.npy")
        visual_rotation_numpy = self._visual_rotation.detach().clone().cpu().numpy()
        np.save(visual_rotation_path, visual_rotation_numpy)

        visual_opacity_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_visual_opacity.npy")
        visual_opacity_numpy = self._visual_opacity.detach().clone().cpu().numpy()
        np.save(visual_opacity_path, visual_opacity_numpy)

    @torch.no_grad
    def save_dense(self, checkpoint_path, frame_idx, scale=True):
        mkdir_p(checkpoint_path)

        dense_xyz_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_dense_xyz.npy")
        dense_xyz_numpy = self._dense_xyz.detach().clone().cpu().numpy()
        np.save(dense_xyz_path, dense_xyz_numpy)

        dense_parent_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_dense_parent.npy")
        dense_parent_numpy = self._dense_parent_idx.detach().clone().cpu().numpy()
        np.save(dense_parent_path, dense_parent_numpy)

        dense_delta_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_dense_delta.npy")
        dense_delta_numpy = self._dense_delta.detach().clone().cpu().numpy()
        np.save(dense_delta_path, dense_delta_numpy)

        dense_color_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_dense_color.npy")
        dense_color_numpy = self._dense_color.detach().clone().cpu().numpy()
        np.save(dense_color_path, dense_color_numpy)

        dense_scales_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_dense_scales.npy")
        dense_scales_numpy = self._dense_scales.detach().clone().cpu().numpy()
        np.save(dense_scales_path, dense_scales_numpy)

        dense_rotation_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_dense_rotation.npy")
        dense_rotation_numpy = self._dense_rotation.detach().clone().cpu().numpy()
        np.save(dense_rotation_path, dense_rotation_numpy)

        dense_opacity_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_dense_opacity.npy")
        dense_opacity_numpy = self._dense_opacity.detach().clone().cpu().numpy()
        np.save(dense_opacity_path, dense_opacity_numpy)

    @torch.no_grad
    def save_re_sim(self, checkpoint_path, frame_idx):
        mkdir_p(checkpoint_path)

        re_sim_xyz_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_re_sim_xyz.npy")
        re_sim_xyz_numpy = self._re_sim_xyz.detach().clone().cpu().numpy() / self.scale_factor
        np.save(re_sim_xyz_path, re_sim_xyz_numpy)

        re_sim_velocity_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_re_sim_velocity.npy")
        re_sim_velocity_numpy = self._re_sim_velocity.detach().clone().cpu().numpy()
        np.save(re_sim_velocity_path, re_sim_velocity_numpy)

        re_sim_particle_id_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_re_sim_particle_id.npy")
        re_sim_particle_id_numpy = self._re_sim_particle_id.detach().clone().cpu().numpy()
        np.save(re_sim_particle_id_path, re_sim_particle_id_numpy)

        re_sim_visual_xyz_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_re_sim_visual_xyz.npy")
        re_sim_visual_xyz_numpy = self._re_sim_visual_xyz.detach().clone().cpu().numpy() / self.scale_factor
        np.save(re_sim_visual_xyz_path, re_sim_visual_xyz_numpy)

        re_sim_visual_velocity_path = os.path.join(
            checkpoint_path, f"frame_{frame_idx:03d}_re_sim_visual_velocity.npy"
        )
        re_sim_visual_velocity_numpy = self._re_sim_visual_velocity.detach().clone().cpu().numpy()
        np.save(re_sim_visual_velocity_path, re_sim_visual_velocity_numpy)

    @torch.no_grad
    def save_all(self, checkpoint_path, frame_idx):
        self.save_hidden(checkpoint_path, frame_idx)
        if self._visual_xyz.shape[0] > 0:
            self.save_visual(checkpoint_path, frame_idx)

    @torch.no_grad
    def load_hidden(self, checkpoint_path, frame_idx):
        xyz_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_xyz.npy")
        assert os.path.exists(xyz_path), f"File not found: {xyz_path}"
        self._xyz = torch.tensor(np.load(xyz_path), dtype=torch.float, device="cuda") * self.scale_factor

        e_xyz_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_estimate_xyz.npy")
        assert os.path.exists(e_xyz_path), f"File not found: {e_xyz_path}"
        self._estimate_xyz = torch.tensor(np.load(e_xyz_path), dtype=torch.float, device="cuda") * self.scale_factor

        buoyancy_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_buoyancy.npy")
        assert os.path.exists(buoyancy_path), f"File not found: {buoyancy_path}"
        self._buoyancy = torch.tensor(np.load(buoyancy_path), dtype=torch.float, device="cuda")

        force_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_force.npy")
        assert os.path.exists(force_path), f"File not found: {force_path}"
        self._force = torch.tensor(np.load(force_path), dtype=torch.float, device="cuda")

        velocity_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_velocity.npy")
        assert os.path.exists(velocity_path), f"File not found: {velocity_path}"
        self._velocity = torch.tensor(np.load(velocity_path), dtype=torch.float, device="cuda")

        i_mass_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_imass.npy")
        assert os.path.exists(i_mass_path), f"File not found: {i_mass_path}"
        self._imass = torch.tensor(np.load(i_mass_path), dtype=torch.float, device="cuda")

        counts_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_counts.npy")
        assert os.path.exists(counts_path), f"File not found: {counts_path}"
        self._counts = torch.tensor(np.load(counts_path), dtype=torch.float, device="cuda")

        gravity_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_gravity.npy")
        assert os.path.exists(gravity_path), f"File not found: {gravity_path}"
        self._gravity = torch.tensor(np.load(gravity_path), dtype=torch.float, device="cuda")

        particle_id_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_particle_id.npy")
        if not os.path.exists(particle_id_path):
            print(f"File not found: {particle_id_path}")
            print(f"Using dummy particle_id")
            self._particle_id = torch.arange(self._xyz.shape[0], device="cuda")
        else:
            self._particle_id = torch.tensor(np.load(particle_id_path), dtype=torch.int, device="cuda")

        scalar_values_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_scalar_values.json")
        assert os.path.exists(scalar_values_path), f"File not found: {scalar_values_path}"
        with open(scalar_values_path, "r") as f:
            scalar_values = json.load(f)
        self.scale_factor = scalar_values["scale_factor"]
        self._secs = scalar_values["secs"]
        self.alpha = scalar_values["alpha"]
        self.k = scalar_values["k"]
        self.p0 = scalar_values["p0"]
        self.buoyancy_decay_rate = scalar_values["buoyancy_decay_rate"]
        self.buoyancy_max_y = scalar_values["buoyancy_max_y"]
        self.min_neighbors = scalar_values["min_neighbors"]
        self.remove_out_boundary = scalar_values["remove_out_boundary"]
        # self.new_visual_particles_per_sec = scalar_values["new_visual_particles_per_sec"]
        # self.new_hidden_particles_per_sec = scalar_values["new_hidden_particles_per_sec"]
        # self.visual_timer = scalar_values["visual_timer"]
        # self.hidden_timer = scalar_values["hidden_timer"]
        self.emit_ratio_hidden = (
            scalar_values["emit_ratio_hidden"] if "emit_ratio_hidden" in scalar_values else self.emit_ratio_hidden
        )
        self.emit_ratio_visual = (
            scalar_values["emit_ratio_visual"] if "emit_ratio_visual" in scalar_values else self.emit_ratio_visual
        )
        self.emit_counter = scalar_values["emit_counter"] if "emit_counter" in scalar_values else self.emit_counter
        self.total_iterations = scalar_values["total_iterations"] if "total_iterations" in scalar_values else 0
        self.total_sim_iterations = (
            scalar_values["total_sim_iterations"] if "total_sim_iterations" in scalar_values else 0
        )
        self.total_tb_log_iterations = (
            scalar_values["total_tb_log_iterations"] if "total_tb_log_iterations" in scalar_values else 0
        )
        self.particle_id_max = scalar_values["particle_id_max"] if "particle_id_max" in scalar_values else 0
        return True

    @torch.no_grad
    def load_visual(self, checkpoint_path, frame_idx, scale=True):
        v_xyz_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_visual_xyz.npy")
        assert os.path.exists(v_xyz_path), f"File not found: {v_xyz_path}"
        self._visual_xyz = torch.tensor(np.load(v_xyz_path), dtype=torch.float, device="cuda")
        if scale:
            self._visual_xyz = self._visual_xyz * self.scale_factor

        visual_color_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_visual_color.npy")
        assert os.path.exists(visual_color_path), f"File not found: {visual_color_path}"
        self._visual_color = torch.tensor(np.load(visual_color_path), dtype=torch.float, device="cuda")

        visual_scales_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_visual_scales.npy")
        assert os.path.exists(visual_scales_path), f"File not found: {visual_scales_path}"
        self._visual_scales = torch.tensor(np.load(visual_scales_path), dtype=torch.float, device="cuda")

        visual_rotation_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_visual_rotation.npy")
        assert os.path.exists(visual_rotation_path), f"File not found: {visual_rotation_path}"
        self._visual_rotation = torch.tensor(np.load(visual_rotation_path), dtype=torch.float, device="cuda")

        visual_opacity_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_visual_opacity.npy")
        assert os.path.exists(visual_opacity_path), f"File not found: {visual_opacity_path}"
        self._visual_opacity = torch.tensor(np.load(visual_opacity_path), dtype=torch.float, device="cuda")

        return self._visual_xyz.shape[0]

    @torch.no_grad
    def load_visual_smoothed(
        self,
        checkpoint_path,
        frame_idx,
        scale=True,
        window_size=5,
        smoothed_color=True,
        smoothed_scales=True,
        smoothed_rotation=True,
        smoothed_opacity=True,
    ):
        v_xyz_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_visual_xyz.npy")
        assert os.path.exists(v_xyz_path), f"File not found: {v_xyz_path}"
        self._visual_xyz = torch.tensor(np.load(v_xyz_path), dtype=torch.float, device="cuda")
        if scale:
            self._visual_xyz = self._visual_xyz * self.scale_factor

        if smoothed_color:
            visual_color_path = os.path.join(
                checkpoint_path, f"frame_{frame_idx:03d}_visual_color_smoothed_ws{window_size}.npy"
            )
        else:
            visual_color_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_visual_color.npy")
        assert os.path.exists(visual_color_path), f"File not found: {visual_color_path}"
        self._visual_color = torch.tensor(np.load(visual_color_path), dtype=torch.float, device="cuda")

        if smoothed_scales:
            visual_scales_path = os.path.join(
                checkpoint_path, f"frame_{frame_idx:03d}_visual_scales_smoothed_ws{window_size}.npy"
            )
        else:
            visual_scales_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_visual_scales.npy")
        assert os.path.exists(visual_scales_path), f"File not found: {visual_scales_path}"
        self._visual_scales = torch.tensor(np.load(visual_scales_path), dtype=torch.float, device="cuda")

        if smoothed_rotation:
            visual_rotation_path = os.path.join(
                checkpoint_path, f"frame_{frame_idx:03d}_visual_rotation_smoothed_ws{window_size}.npy"
            )
        else:
            visual_rotation_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_visual_rotation.npy")
        assert os.path.exists(visual_rotation_path), f"File not found: {visual_rotation_path}"
        self._visual_rotation = torch.tensor(np.load(visual_rotation_path), dtype=torch.float, device="cuda")

        if smoothed_opacity:
            visual_opacity_path = os.path.join(
                checkpoint_path, f"frame_{frame_idx:03d}_visual_opacity_smoothed_ws{window_size}.npy"
            )
        else:
            visual_opacity_path = os.path.join(checkpoint_path, f"frame_{frame_idx:03d}_visual_opacity.npy")
        assert os.path.exists(visual_opacity_path), f"File not found: {visual_opacity_path}"
        self._visual_opacity = torch.tensor(np.load(visual_opacity_path), dtype=torch.float, device="cuda")

        return self._visual_xyz.shape[0]

    @torch.no_grad
    def load_all(self, checkpoint_path, frame_idx):
        ret_hid = self.load_hidden(checkpoint_path, frame_idx)
        ret_vis = self.load_visual(checkpoint_path, frame_idx)
        return ret_hid > 0 and ret_vis > 0
