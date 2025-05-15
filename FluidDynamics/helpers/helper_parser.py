import copy
import json
import os
import sys

from argparse import ArgumentParser, Namespace

import torch
import yaml

from arguments import ModelParams, OptimizationParams, PipelineParams
from utils.general_utils import safe_state


def get_parser():
    parser = ArgumentParser(description="Training script parameters")
    mp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6029)
    parser.add_argument("--debug_from", type=int, default=-2)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)

    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])

    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--start_checkpoint", type=str, default=None)

    parser.add_argument("--config_path", type=str, default="None")

    args = parser.parse_args(sys.argv[1:])

    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)

    print("Model path: " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # incase we provide config file not directly pass to the file
    if os.path.exists(args.config_path) and args.config_path != "None":
        print("Overload config from " + args.config_path)
        config = json.load(open(args.config_path))
        for k in config.keys():
            try:
                value = getattr(args, k)
                newvalue = config[k]
                setattr(args, k, newvalue)
            except:
                print("failed set config: " + k)
        print("Finish load config from " + args.config_path)
    else:
        raise ValueError("config file not exist or not provided")

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    return args, mp.extract(args), op.extract(args), pp.extract(args)


def get_test_parser():
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)

    parser.add_argument("--test_iteration", default=-1, type=int)

    parser.add_argument("--val_loader", type=str, default="colmap")
    parser.add_argument("--config_path", type=str, default="1")

    parser.add_argument("--test", action="store_true")
    parser.add_argument("--future", action="store_true")

    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    if os.path.exists(args.config_path) and args.config_path != "None":
        print("overload config from " + args.config_path)
        config = json.load(open(args.config_path))
        for k in config.keys():
            try:
                value = getattr(args, k)
                newvalue = config[k]
                setattr(args, k, newvalue)
            except:
                print("failed set config: " + k)
        print("finish load config from " + args.config_path)
        # print("args: " + str(args))

        return args, model.extract(args), optimization.extract(args), pipeline.extract(args)


def write_args_to_file(args, model_args, optim_args, pipe_args, name):
    cfg_output_path = os.path.join(model_args.model_path, f"{name}_cfg_args.yaml")
    idx = 0
    while os.path.exists(cfg_output_path):
        cfg_output_path = os.path.join(model_args.model_path, f"{name}_cfg_args_{idx}.yaml")
        idx += 1
    with open(cfg_output_path, "w") as cfg_log_f:
        # write args to yml file
        cfg_log_f.write(f"model_args:\n")
        for k, v in vars(model_args).items():
            cfg_log_f.write(f"  {k}: {v}\n")
        cfg_log_f.write(f"optim_args:\n")
        for k, v in vars(optim_args).items():
            cfg_log_f.write(f"  {k}: {v}\n")
        cfg_log_f.write(f"pipe_args:\n")
        for k, v in vars(pipe_args).items():
            cfg_log_f.write(f"  {k}: {v}\n")
        cfg_log_f.write(f"args:\n")
        for k, v in vars(args).items():
            cfg_log_f.write(f"  {k}: {v}\n")

    return cfg_output_path


def get_combined_args(parser: ArgumentParser):
    cmd_line_string = sys.argv[1:]
    args_cmdline = parser.parse_args(cmd_line_string)

    cfg_file_path = os.path.join(args_cmdline.model_path, "cfg_args")
    if os.path.exists(cfg_file_path):
        print("Looking for config file in", cfg_file_path)
        with open(cfg_file_path) as cfg_file:
            print(f"Config file found: {cfg_file_path}")
            cfg_file_string = cfg_file.read()
        args_cfg_file = eval(cfg_file_string)

        merged_dict = vars(args_cfg_file).copy()
    else:
        cfg_file_dir = os.listdir(args_cmdline.model_path)
        cfg_file_names = [os.path.join(args_cmdline.model_path, f) for f in cfg_file_dir if f.endswith(".yaml")]
        if len(cfg_file_names) == 0:
            raise FileNotFoundError(f"No config file found in {args_cmdline.model_path}")
        cfg_file_name = cfg_file_names[-1]
        cfg_file_path = os.path.join(args_cmdline.model_path, cfg_file_name)
        print("Looking for config file in", cfg_file_path)
        with open(cfg_file_path) as cfg_file:
            print(f"Config file found: {cfg_file_path}")
            cfg_data = yaml.load(cfg_file, Loader=yaml.FullLoader)

        merged_dict = copy.deepcopy(cfg_data["args"])

    for k, v in vars(args_cmdline).items():
        if k not in merged_dict:
            # print(f"New argument {k}: {v}")
            merged_dict[k] = v
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
