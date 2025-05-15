def get_render_pipe(option="render_gs"):
    print("Render pipeline option:", option)
    if option == "render_gs":
        # render_gs: the raw gaussian splatting renderer
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from renderer import render

        return render, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "render_fluid":
        # render_fluid: the fluid renderer, only the fluid is rendered, no background in this case/scene
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from renderer import render_fluid

        return render_fluid, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "render_background":
        # render_background: the background renderer
        from diff_gaussian_rasterization_ch3 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from renderer import render_background

        return render_background, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "render_dynamics":
        # render_dynamics: the dynamics renderer, background and dynamics are separated
        from diff_gaussian_rasterization_ch3 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from renderer import render_dynamics

        return render_dynamics, GaussianRasterizationSettings, GaussianRasterizer

    else:
        raise NotImplementedError(f"Render {option} not implemented")
