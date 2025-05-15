import math

from typing import NamedTuple

import numpy as np
import torch


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array
    times: np.array


def get_world_2_view(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def get_world_2_view2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def get_projection_matrix(z_near, z_far, fovX, fovY):  # ndc to
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * z_near
    bottom = -top
    right = tanHalfFovX * z_near
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * z_near / (right - left)
    P[1, 1] = 2.0 * z_near / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    # P[2, 2] = z_sign * z_far / (z_far - z_near)
    P[2, 2] = z_sign * (z_far + z_near) / (z_far - z_near)

    P[2, 3] = -(z_far * z_near) / (z_far - z_near)
    return P


# https://stackoverflow.com/a/22064917
# GLdouble perspMatrix[16]={2*fx/w,0,0,0,0,2*fy/h,0,0,2*(cx/w)-1,2*(cy/h)-1,-(far+near)/(far-near),-1,0,0,-2*far*near/(far-near),0};
# https://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/


# def get_projection_matrix_cv(z_near, z_far, fovX, fovY, cx=0.0, cy=0.0):
#     '''
#     cx and cy range is -1 1 which is the ratio of the image size - 0.5 *2!

#     '''
#     tanHalfFovY = math.tan(fovY / 2)
#     tanHalfFovX = math.tan(fovX / 2)

#     top = tanHalfFovY * z_near
#     bottom = -top
#     right = tanHalfFovX * z_near
#     left = -right

#     # Adjust for off-center projection
#     left += cx * z_near
#     right += cx * z_near
#     top += cy * z_near
#     bottom += cy * z_near

#     P = torch.zeros(4, 4)

#     z_sign = 1.0

#     P[0, 0] = 2.0 * z_near / (right - left)
#     P[1, 1] = 2.0 * z_near / (top - bottom)
#     P[0, 2] = (right + left) / (right - left)
#     P[1, 2] = (top + bottom) / (top - bottom)
#     P[3, 2] = z_sign
#     P[2, 2] = z_sign * z_far / (z_far - z_near)
#     P[2, 3] = -(z_far * z_near) / (z_far - z_near)
#     return P


def get_projection_matrix_cv(z_near, z_far, fovX, fovY, cx=0.0, cy=0.0):
    """
    cx and cy range is -0.5 to 0.5
    we use cx cy range -0.5 * 0.5

    """
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)

    top = tanHalfFovY * z_near
    bottom = -top
    right = tanHalfFovX * z_near
    left = -right

    # Adjust for off-center projection
    delta_x = (2 * tanHalfFovX * z_near) * cx  #
    delta_y = (2 * tanHalfFovY * z_near) * cy  #

    left += delta_x
    right += delta_x
    top += delta_y
    bottom += delta_y

    # left -= delta_x
    # right -= delta_x
    # top -= delta_y
    # bottom -= delta_y

    # left = left(1 +cx)* z_near
    # right += cx * z_near
    # top += cy * z_near
    # bottom += cy * z_near

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * z_near / (right - left)
    P[1, 1] = 2.0 * z_near / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    # P[2, 2] = z_sign * z_far / (z_far - z_near)
    P[2, 2] = z_sign * (z_far + z_near) / (z_far - z_near)

    P[2, 3] = -(z_far * z_near) / (z_far - z_near)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0


def ndc2pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5
