import os

from argparse import ArgumentParser


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):

    def export_changed_args_to_json(self, args):
        defaults = {}
        for arg in vars(args).items():
            try:
                if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                    default_value = getattr(self, arg[0])
                    # defaults[ arg[0] ] = default_value
                    if default_value != arg[1]:
                        defaults[arg[0]] = arg[1]
            except:
                pass

        return defaults

    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._data_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self.white_background = False
        self.random_background = False
        self.data_device = "cuda"
        self.verify_llff = 0
        self.eval = False
        self.model = "g_model"
        self.loader = "colmap"

        self.basic_function = ""
        self.densify = 0

        self.rgb_function = "none"

        self.start_time = 0
        self.duration = 50
        self.time_step = 1
        self.max_timestamp = 1.0

        self.is_wind = False

        self.gen_future_since = -1
        self.gen_prefixed_future = "one"
        self.gen_future_strength = "0d75"

        self.future_since = -1
        self.data_2_path = ""
        self.data_2_since = -1

        self.is_circle = False
        self.circle_cameras_around = ""
        self.circle_cameras_num = 0
        self.circle_cameras_radius = 0
        self.use_demo_cameras = False

        # GaussianFluid parameters
        self.gray_image = False
        self.test_all_views = False
        # four training cameras, 00, 01, 03, 04
        self.train_views = "0134"
        # same format as train views # the views that produced from zero123 finetuned model
        self.train_views_fake = None
        self.test_views_fake = None
        self.use_refined_fake = False
        self.refined_strength = "0d26"

        self.real_view_repeat = 1

        self.use_extra_transforms = False

        self.source_init = False
        self.new_pts = 10_000
        self.img_offset = False
        self.is_bg = False
        self.capture_part = "smoke"

        self.init_region_type = "large"

        self.no_init_pcd = False
        self.init_pcd_bg = False
        self.init_pcd_object = False
        self.init_pcd_large_smoke = False

        self.spherical_cam_start = 0
        self.spherical_cam_duration = 120
        self.spherical_cam_step = 1

        self.init_num_pts_per_time = 1000
        self.init_trbf_c_fix = False
        self.init_color_fix_value = None  # None for random color, float for fix value

        self.load_path = ""
        self.level_two_load_path = ""
        self.level_two_color_3ch = False

        self.bg_load_path = ""
        self.bg_2_load_path = ""
        self.bg_load_iteration = 30000

        self.load_low_path = ""
        self.load_high_path = ""

        self.init_visual_num_pts = 1000
        self.init_thick_visual_num_pts = 150
        # visual_radius_max: capture_black_gold_middle_one: 0.013, others 0.012
        self.init_visual_radius_small_max = 0.014
        self.init_visual_radius_max = 0.028
        self.init_x_mid = 0.326
        self.init_visual_y_min = -0.09
        self.init_visual_y_max = 0.32
        self.init_z_mid = -0.3

        self.init_rotation_degree = 0

        self.init_visual_y_thick_min = 0.16

        self.init_hidden_radius_max = 0.042
        self.init_hidden_delta = 0.009
        self.init_hidden_y_min = -0.11
        self.init_hidden_y_max = 0.35

        self.emitter_hidden_delta = 0.009
        self.emitter_visual_delta = 0.004
        self.emitter_center_y_hidden = -0.11
        self.emitter_center_y_visual = -0.09
        self.emitter_center_y_hidden_max = 0.25
        self.emitter_center_y_visual_max = 0.16

        # Radius of the circle
        self.emitter_visual_radius_ratio = 3
        self.emitter_hidden_radius_ratio = 5

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.data_path = os.path.abspath(g.data_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.rd_pipe = "v2"
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000

        self.batch = 2

        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.color_lr = 0.0025
        self.feature_lr = 0.0025
        self.feature_t_lr = 0.001
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005

        self.trbf_c_lr = 0.0001
        self.trbf_s_lr = 0.03
        self.trbf_scale_init = 0.0
        self.rgb_lr = 0.0001

        self.move_lr = 3.5

        self.omega_lr = 0.0001
        self.beta_lr = 0.0001
        self.rotation_lr = 0.001

        self.lambda_dssim = 0.2

        self.percent_dense = 0.01

        self.opacity_reset_interval = 3_000
        self.opacity_reset_at = 10000

        self.densify_cnt = 6
        self.reg = 0
        self.lambda_reg = 0.0001
        self.shrink_scale = 2.0
        self.random_feature = 0
        self.ems_type = 0
        self.radials = 10.0
        self.new_ray_step = 2
        self.ems_start = 1600  # small for debug
        self.loss_tart = 200
        self.save_emp_points = 0
        self.prune_by_size = 0
        self.ems_threshold = 0.6
        self.opacity_threshold = 0.005
        self.selective_view = 0
        self.preprocess_points = 0
        self.freeze_rotation_iteration = 8001
        self.add_sph_points_scale = 0.8
        self.g_num_limit = 330000
        self.ray_end = 7.5
        self.ray_start = 0.7
        self.shuffle_ems = 1
        self.prev_path = "1"
        self.load_all = 0
        self.remove_scale = 5
        self.gt_mask = 0  # 0 means not train with mask for undistorted gt image; 1 means

        # scalar_real
        self.cur_time_only_iterations = 10000
        self.iterations_per_time = 250
        self.iterations_per_time_post = 12

        self.lambda_velocity = 0.0
        self.lambda_opacity_vel = 0.0

        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002

        self.clone = True
        self.split = True
        self.split_prune = True
        self.prune = True

        self.valid_min_y = -0.035
        self.valid_max_z = -0.58

        self.prune_near_interval = 0
        self.prune_near_with_object = False
        self.prune_near_cam_interval = 0
        self.prune_large_interval = 0
        self.prune_bbox_interval = 0

        self.post_prune = False
        self.post_prune_interval = 100
        self.post_prune_from_iter = 25000
        self.post_prune_until_iter = 27000

        self.zero_grad_level = None

        self.act_level_1 = False

        self.transparent_level_0 = False

        # PBD simulation
        # the highest priority for densification
        self.no_densify_prune = False
        self.iterations_per_time_first = 1000
        self.iterations_per_time_current = 1000
        self.iterations_per_time_current_max = 1000
        self.iterations_per_time_current_sparse = 500
        self.iterations_per_time_current_level_two = 1000
        self.iterations_per_time_current_level_two_max = 1000

        self.record_time = False

        self.min_neighbors = -1
        self.remove_out_boundary = False
        self.secs = 0.01
        self.alpha = -1.5
        self.buoyancy_max_y = 0.0
        self.beta = 0.1
        self.buoyancy_decay_rate = 0.0

        self.H = 2.0
        self.p0 = 2.0
        self.p0_future = 1.5
        self.k = 10
        self.KNN_K = 100

        self.extra_visual_ratio = 0.0
        self.extra_visual_num = 0
        self.extra_visual_y_min = 0.16
        self.extra_visual_min_num = 0
        self.extra_visual_pilar_radius = 0.06
        self.extra_visual_pilar_radius_delta = 0.0015

        self.pos_lr_scale_factor = 1.0

        self.init_hidden_velocity = 0.0

        self.new_hidden_particles_per_sec = 15
        self.new_visual_particles_per_sec = 15
        self.stable_iterations = 20
        self.stable_iterations_future = 0
        self.solver_iterations = 3
        self.solver_iterations_future = 3

        self.decay_frames_future_p0 = 30

        self.sparse_views_from_time_index = -1
        self.sparse_views = ["train00"]

        self.max_hidden_particles = 28000
        self.future_pred_frames = 0

        self.simulation_ratio = 3
        self.emitter_points_off_y0 = False

        self.emit_ratio_hidden = 1.32  # 0.033 / (1/40)
        self.emit_ratio_visual = 1.32  # 0.033 / (1/40)

        self.lambda_first_distance = 0.0
        self.distance_threshold_hidden = 0.001
        self.distance_threshold_visual = 0.001

        self.lambda_current_distance = 0.0

        self.lambda_exyz = 0.0
        self.lambda_vel = 0.0

        self.lambda_image = 1.0
        self.lambda_gas_constraints = 0.0
        self.lambda_next_gas_constraints = 0.0

        self.velocity_lr_init = 0.00016
        self.xyz_lr = 1e-4

        self.fit_features = False
        self.visual_features_lr = 0.0025
        self.fit_color = False
        self.visual_color_lr = 0.0025
        self.high_color_lr = 0.0025
        self.dense_color_lr = 0.0025
        self.fit_opacity = False
        self.visual_opacity_lr = 0.05
        self.high_opacity_lr = 0.05
        self.dense_opacity_lr = 0.05
        self.fit_scales = False
        self.visual_scales_lr = 0.005
        self.high_scales_lr = 0.005
        self.dense_scales_lr = 0.005
        self.fit_rotation = False
        self.visual_rotation_lr = 0.001
        self.high_rotation_lr = 0.001
        self.dense_rotation_lr = 0.001

        self.fit_xyz = False
        self.high_xyz_lr = 0.00016 * 1.8

        self.lambda_consistency_color = 0.0
        self.consistency_color_threshold = 0.0
        self.lambda_consistency_opacity = 0.0
        self.consistency_opacity_threshold = 0.0
        self.lambda_consistency_scales = 0.0
        self.consistency_scales_threshold = 0.0
        self.lambda_consistency_rotation = 0.0
        self.consistency_rotation_threshold = 0.0
        self.lambda_consistency_xyz = 0.0
        self.lambda_min_update_xyz = 0.0

        self.init_scales_w_xyz_dist = False

        self.inherit_prev_features = False
        self.inherit_prev_color = False
        self.inherit_prev_opacity = False
        self.inherit_prev_scales = False
        self.inherit_prev_rotation = False

        self.lambda_reg_scaling = 0.0
        self.scaling_reg_ratio_threshold = 0

        self.high_frequency_per_visual = 1

        self.smoothed_window_size = 5
        self.use_smoothed_features = True
        self.use_smoothed_color = True
        self.use_smoothed_scales = True
        self.use_smoothed_opacity = True
        self.use_smoothed_rotation = True

        self.use_level_two_in_future = False
        self.use_level_two_smoothed_in_future = False

        self.wind_since = -1
        self.wind_force = [0.0, 0.0, 0.0]
        self.wind_power = 1.0

        self.rigid_since = -1
        self.rigid_body = "cuboid"
        self.rigid_body_center = [0.34, 0.5, -0.225]  # in rendering space
        self.rigid_particle_radius = 0.25  # in simulation space
        self.rigid_cuboid_num_one_side = 15
        self.rigid_cuboid_num = [5, 10, 55]
        self.rigid_sphere_radius = 5  # in simulation space
        self.rigid_sphere_num = 1000
        self.rigid_cylinder_radius = 4  # in simulation space
        self.rigid_cylinder_num = [50, 50]

        super().__init__(parser, "Optimization Parameters")
