import random
import sys

from datetime import datetime

import numpy as np
import torch


def inv_sigmoid(x):
    return torch.log(x / (1 - x))


def sigmoid_v2(x):
    return 2 * torch.sigmoid(x)


def inv_sigmoid_v2(x):
    return torch.log((x / 2) / (1 - (x / 2)))


def sigmoid_k(x, k):
    return k * torch.sigmoid(x)


def inv_sigmoid_k(x, k):
    return torch.log((x / k) / (1 - (x / k)))


def sigmoid_a(x):
    return 0.7 * torch.sigmoid(x)


def inv_sigmoid_a(x):
    return torch.log((x / 0.7) / (1 - (x / 0.7)))


def sigmoid_c(x):
    return 0.8 * torch.sigmoid(x)


def inv_sigmoid_c(x):
    return torch.log((x / 0.8) / (1 - (x / 0.8)))


def sigmoid_v3(x):
    return 1.4 * torch.sigmoid(x)


def inv_sigmoid_v3(x):
    return torch.log((x / 1.4) / (1 - (x / 1.4)))


def pil_to_torch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lower_diag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lower_diag(sym)


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


# def update_quaternion(q, omega, delta_t):
#     magnitude_omega = torch.nn.functional.normalize(omega)
#     #norm = torch.sqrt([:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

#     if magnitude_omega < 1e-8:
#         return q

#     half_angle = magnitude_omega * delta_t / 2.0
#     delta_q = np.array([
#         np.cos(half_angle),
#         *(omega/magnitude_omega * np.sin(half_angle))
#     ])

#     # Quaternion multiplication
#     q_prime = np.array([
#         q[0] * delta_q[0] - np.dot(q[1:], delta_q[1:]),
#         q[0]*delta_q[1] + delta_q[0]*q[1] + q[2]*delta_q[3] - q[3]*delta_q[2],
#         q[0]*delta_q[2] + delta_q[0]*q[2] + q[3]*delta_q[1] - q[1]*delta_q[3],
#         q[0]*delta_q[3] + delta_q[0]*q[3] + q[1]*delta_q[2] - q[2]*delta_q[1]
#     ])

#     return q_prime


def update_quaternion(q, omega, delta_t):
    magnitude_omega = torch.norm(omega, dim=1, keepdim=True)
    half_angle = magnitude_omega * delta_t / 2.0
    delta_q_cos = torch.cos(half_angle)
    delta_q_sin = (
        torch.sin(half_angle) * omega / (magnitude_omega + torch.tensor([1e-8], dtype=torch.float, device="cuda"))
    )

    delta_q = torch.cat((delta_q_cos, delta_q_sin), dim=1)

    # Quaternion multiplication
    q0_delta_q0 = q[:, 0:1] * delta_q[:, 0:1]
    cross_product = torch.cross(q[:, 1:], delta_q[:, 1:], dim=1)
    dot_product = (q[:, 1:] * delta_q[:, 1:]).sum(dim=1, keepdim=True)
    q_prime = torch.cat(
        (q0_delta_q0 - dot_product, q[:, 0:1] * delta_q[:, 1:] + delta_q[:, 0:1] * q[:, 1:] + cross_product), dim=1
    )

    return q_prime


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def safe_state(silent):
    # old_f = sys.stdout

    # class F:
    #     def __init__(self, silent):
    #         self.silent = silent

    #     def write(self, x):
    #         if not self.silent:
    #             if x.endswith("\n"):
    #                 time_str = str(datetime.now().strftime("%d/%m %H:%M:%S"))
    #                 old_f.write(time_str + " " + x)
    #             else:
    #                 old_f.write(x)

    #     def flush(self):
    #         old_f.flush()

    # sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
