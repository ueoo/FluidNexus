import torch


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def psnr_mask(img1, img2):
    valid_mask = torch.sum(img2[:, :, :], dim=1) > 0.01
    valid_mask = valid_mask.repeat(3, 1, 1)  # .float()
    valid_mask = valid_mask.view(img1.shape[0], -1)

    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1)[:, valid_mask[0, :]].mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
