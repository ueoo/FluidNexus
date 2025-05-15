import math

from contextlib import nullcontext

import numpy as np
import torch

from einops import rearrange
from PIL import Image
from rich import print
from torch import autocast
from torchvision import transforms

from ldm.models.diffusion.ddim import DDIMSampler


@torch.no_grad()
def sample_model_simple(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale, ddim_eta, T):
    precision_scope = autocast if precision == "autocast" else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond["c_crossattn"] = [c]
            cond["c_concat"] = [
                model.encode_first_stage((input_im.to(c.device))).mode().detach().repeat(n_samples, 1, 1, 1)
            ]
            if scale != 1.0:
                uc = {}
                uc["c_concat"] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc["c_crossattn"] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(
                S=ddim_steps,
                conditioning=cond,
                batch_size=n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
                eta=ddim_eta,
                x_T=None,
            )
            # print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def main_run_simple(
    models,
    device,
    T,
    raw_im=None,
    scale=3.0,
    n_samples=4,
    ddim_steps=50,
    ddim_eta=1.0,
    precision="fp32",
    h=256,
    w=256,
    save_path=None,
):
    # print("running main_run_simple ...")
    input_im = raw_im

    sampler = DDIMSampler(models["turncam"])

    x_samples_ddim = sample_model_simple(
        input_im,
        models["turncam"],
        sampler,
        precision,
        h,
        w,
        ddim_steps,
        n_samples,
        scale,
        ddim_eta,
        T,
    )

    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))
    return output_ims


@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale, ddim_eta, x, y, z):
    precision_scope = autocast if precision == "autocast" else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([math.radians(x), math.sin(math.radians(y)), math.cos(math.radians(y)), z])
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond["c_crossattn"] = [c]
            cond["c_concat"] = [
                model.encode_first_stage((input_im.to(c.device))).mode().detach().repeat(n_samples, 1, 1, 1)
            ]
            if scale != 1.0:
                uc = {}
                uc["c_concat"] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc["c_crossattn"] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(
                S=ddim_steps,
                conditioning=cond,
                batch_size=n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
                eta=ddim_eta,
                x_T=None,
            )
            print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def main_run_smoke(
    models,
    device,
    cam_vis,
    return_what,
    x=0.0,
    y=0.0,
    z=0.0,
    raw_im=None,
    preprocess=True,
    scale=3.0,
    n_samples=4,
    ddim_steps=50,
    ddim_eta=1.0,
    precision="fp32",
    h=256,
    w=256,
    smoke=False,
    save_path=None,
):
    input_im = raw_im
    show_in_im1 = raw_im
    show_in_im2 = Image.fromarray(show_in_im1)

    if "rand" in return_what:
        x = int(np.round(np.arcsin(np.random.uniform(-1.0, 1.0)) * 160.0 / np.pi))  # [-80, 80].
        y = int(np.round(np.random.uniform(-150.0, 150.0)))
        z = 0.0

    if cam_vis._gradio_plot is not None:
        cam_vis.polar_change(x)
        cam_vis.azimuth_change(y)
        cam_vis.radius_change(z)
        cam_vis.encode_image(show_in_im1)
        new_fig = cam_vis.update_figure()
    else:
        new_fig = None

    if "vis" in return_what:
        description = (
            "The viewpoints are visualized on the top right. "
            "Click Run Generation to update the results on the bottom right."
        )

        if "angles" in return_what:
            return (x, y, z, description, new_fig, show_in_im2)
        else:
            return (description, new_fig, show_in_im2)

    elif "gen" in return_what:
        input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
        input_im = input_im * 2 - 1
        input_im = transforms.functional.resize(input_im, [h, w])

        sampler = DDIMSampler(models["turncam"])
        # used_x = -x  # NOTE: Polar makes more sense in Basile's opinion this way!
        used_x = x  # NOTE: Set this way for consistency.
        x_samples_ddim = sample_model(
            input_im, models["turncam"], sampler, precision, h, w, ddim_steps, n_samples, scale, ddim_eta, used_x, y, z
        )

        output_ims = []
        for x_sample in x_samples_ddim:
            x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
            output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))

        description = None

        if "angles" in return_what:
            return (x, y, z, description, new_fig, show_in_im2, output_ims)
        else:
            return (description, new_fig, show_in_im2, output_ims)
