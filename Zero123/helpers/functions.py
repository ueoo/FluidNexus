import torch

from pytorch_lightning.utilities.distributed import rank_zero_only

from ldm.util import instantiate_from_config


@rank_zero_only
def rank_zero_print(*args):
    print(*args)


def modify_weights(w, scale=1e-6, n=2):
    """Modify weights to accomodate concatenation to unet"""
    extra_w = scale * torch.randn_like(w)
    new_w = w.clone()
    for i in range(n):
        new_w = torch.cat((new_w, extra_w.clone()), dim=1)
    return new_w


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model
