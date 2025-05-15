from math import exp

import torch
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def relative_loss(network_output, gt):
    return torch.abs((network_output - gt) / (gt + 0.001)).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).detach().clone().contiguous()
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim_map(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_map(img1, img2, window, window_size, channel, size_average)


def _ssim_map(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map


def distance_loss(positions, threshold):
    """
    Compute a loss to penalize distances between particles that are below a certain threshold.

    Args:
        positions (torch.Tensor): Tensor of shape (N, 3) representing the positions of N particles.
        threshold (float): The minimum allowed distance between particles.

    Returns:
        torch.Tensor: The computed loss value.
    """
    # Compute the pairwise distances
    distances = torch.cdist(positions, positions, p=2)  # Compute pairwise Euclidean distances

    # Create a mask for distances below the threshold
    mask = distances < threshold

    # Set diagonal to False to ignore self-distances
    mask.fill_diagonal_(False)

    # Compute the loss only for distances below the threshold
    loss = ((threshold - distances) * mask.float()).clamp(min=0).pow(2).sum()

    return loss


def l1_loss_optimal_matching(predictions, ground_truth):
    # Compute pairwise L1 distances
    pairwise_distances = torch.cdist(predictions, ground_truth, p=1)

    # Convert pairwise distances to numpy array for linear_sum_assignment
    pairwise_distances_np = pairwise_distances.cpu().detach().numpy()

    # Apply the Hungarian algorithm to find the optimal assignment
    row_indices, col_indices = linear_sum_assignment(pairwise_distances_np)

    # Compute the L1 loss based on the optimal assignment
    l1_loss = pairwise_distances[row_indices, col_indices].sum()

    return l1_loss


def l2_loss_consistency(predictions, prev_predictions, threshold=0.0):
    if prev_predictions is None:
        return torch.zeros(1, device=predictions.device)
    prev_num = prev_predictions.shape[0]
    cur_num = predictions.shape[0]
    assert cur_num >= prev_num, "Current number of particles must be greater than or equal to the previous"
    loss_value = F.mse_loss(predictions[:prev_num], prev_predictions)
    return loss_value
