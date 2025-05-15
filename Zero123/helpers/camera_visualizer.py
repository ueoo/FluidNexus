import numpy as np
import plotly.graph_objects as go

from lovely_numpy import lo
from PIL import Image
from rich import print

from helpers.camera_utils import calc_cam_cone_pts_3d


class CameraVisualizer:
    def __init__(self, gradio_plot):
        self._gradio_plot = gradio_plot
        self._fig = None
        self._polar = 0.0
        self._azimuth = 0.0
        self._radius = 0.0
        self._raw_image = None
        self._8bit_image = None
        self._image_colorscale = None

    def polar_change(self, value):
        self._polar = value
        # return self.update_figure()

    def azimuth_change(self, value):
        self._azimuth = value
        # return self.update_figure()

    def radius_change(self, value):
        self._radius = value
        # return self.update_figure()

    def encode_image(self, raw_image):
        """
        :param raw_image (H, W, 3) array of uint8 in [0, 255].
        """
        # https://stackoverflow.com/questions/60685749/python-plotly-how-to-add-an-image-to-a-3d-scatter-plot

        dum_img = Image.fromarray(np.ones((3, 3, 3), dtype="uint8")).convert("P", palette="WEB")
        idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))

        self._raw_image = raw_image
        self._8bit_image = Image.fromarray(raw_image).convert("P", palette="WEB", dither=None)
        # self._8bit_image = Image.fromarray(raw_image.clip(0, 254)).convert(
        #     'P', palette='WEB', dither=None)
        self._image_colorscale = [[i / 255.0, "rgb({}, {}, {})".format(*rgb)] for i, rgb in enumerate(idx_to_color)]

        # return self.update_figure()

    def update_figure(self):
        fig = go.Figure()

        if self._raw_image is not None:
            (H, W, C) = self._raw_image.shape

            x = np.zeros((H, W))
            (y, z) = np.meshgrid(np.linspace(-1.0, 1.0, W), np.linspace(1.0, -1.0, H) * H / W)
            print("x:", lo(x))
            print("y:", lo(y))
            print("z:", lo(z))

            fig.add_trace(
                go.Surface(
                    x=x,
                    y=y,
                    z=z,
                    surfacecolor=self._8bit_image,
                    cmin=0,
                    cmax=255,
                    colorscale=self._image_colorscale,
                    showscale=False,
                    lighting_diffuse=1.0,
                    lighting_ambient=1.0,
                    lighting_fresnel=1.0,
                    lighting_roughness=1.0,
                    lighting_specular=0.3,
                )
            )

            scene_bounds = 3.5
            base_radius = 2.5
            zoom_scale = 1.5  # Note that input radius offset is in [-0.5, 0.5].
            fov_deg = 50.0
            edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]

            input_cone = calc_cam_cone_pts_3d(0.0, 0.0, base_radius, fov_deg)  # (5, 3).
            output_cone = calc_cam_cone_pts_3d(
                self._polar, self._azimuth, base_radius + self._radius * zoom_scale, fov_deg
            )  # (5, 3).
            # print('input_cone:', lo(input_cone).v)
            # print('output_cone:', lo(output_cone).v)

            for cone, clr, legend in [(input_cone, "green", "Input view"), (output_cone, "blue", "Target view")]:

                for i, edge in enumerate(edges):
                    (x1, x2) = (cone[edge[0], 0], cone[edge[1], 0])
                    (y1, y2) = (cone[edge[0], 1], cone[edge[1], 1])
                    (z1, z2) = (cone[edge[0], 2], cone[edge[1], 2])
                    fig.add_trace(
                        go.Scatter3d(
                            x=[x1, x2],
                            y=[y1, y2],
                            z=[z1, z2],
                            mode="lines",
                            line=dict(color=clr, width=3),
                            name=legend,
                            showlegend=(i == 0),
                        )
                    )
                    # text=(legend if i == 0 else None),
                    # textposition='bottom center'))
                    # hoverinfo='text',
                    # hovertext='hovertext'))

                # Add label.
                if cone[0, 2] <= base_radius / 2.0:
                    fig.add_trace(
                        go.Scatter3d(
                            x=[cone[0, 0]],
                            y=[cone[0, 1]],
                            z=[cone[0, 2] - 0.05],
                            showlegend=False,
                            mode="text",
                            text=legend,
                            textposition="bottom center",
                        )
                    )
                else:
                    fig.add_trace(
                        go.Scatter3d(
                            x=[cone[0, 0]],
                            y=[cone[0, 1]],
                            z=[cone[0, 2] + 0.05],
                            showlegend=False,
                            mode="text",
                            text=legend,
                            textposition="top center",
                        )
                    )

            # look at center of scene
            fig.update_layout(
                # width=640,
                # height=480,
                # height=400,
                height=360,
                autosize=True,
                hovermode=False,
                margin=go.layout.Margin(l=0, r=0, b=0, t=0),
                showlegend=True,
                legend=dict(
                    yanchor="bottom",
                    y=0.01,
                    xanchor="right",
                    x=0.99,
                ),
                scene=dict(
                    aspectmode="manual",
                    aspectratio=dict(x=1, y=1, z=1.0),
                    camera=dict(
                        eye=dict(x=base_radius - 1.6, y=0.0, z=0.6),
                        center=dict(x=0.0, y=0.0, z=0.0),
                        up=dict(x=0.0, y=0.0, z=1.0),
                    ),
                    xaxis_title="",
                    yaxis_title="",
                    zaxis_title="",
                    xaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks="",
                    ),
                    yaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks="",
                    ),
                    zaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks="",
                    ),
                ),
            )

        self._fig = fig
        return fig
