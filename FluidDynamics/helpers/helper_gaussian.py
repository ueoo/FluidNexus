from gaussian_splatting.gaussian_model import GaussianModel


def get_model(model="gm_gs") -> GaussianModel:
    if model == "gm_gs":
        # gm_gs: the raw gaussian splatting model
        from gaussian_splatting.gaussian_model import GaussianModel

    elif model == "gm_fluid":
        # gm_fluid: the fluid gaussian splatting model
        # in this case, we don't separate the background and fluid
        from gaussian_splatting.gm_fluid import GaussianModel

    elif model == "gm_background":
        # gm_background: the background gaussian splatting model
        from gaussian_splatting.gm_background import GaussianModel

    elif model == "gm_dynamics":
        # gm_dynamics: the dynamics gaussian splatting model
        # in this case, we separate the background and fluid
        from gaussian_splatting.gm_dynamics import GaussianModel

    else:
        raise ValueError(f"Model {model} not found")

    return GaussianModel
