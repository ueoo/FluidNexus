import os
import time

import cv2
import numpy as np

from lovely_numpy import lo
from PIL import Image
from rich import print

from ldm.util import load_and_preprocess


def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_image_gray(img_path, invert=False):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if invert:
        img = 255 - img
    return img


def preprocess_image(models, input_im, preprocess, smoke=False, save_path=None):
    """
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    """

    print("processing image...")
    print("old input_im:", input_im.size)
    start_time = time.time()

    if preprocess:
        input_im = load_and_preprocess(models["carvekit"], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        # (H, W, 3) array in [0, 1].

    elif smoke:
        # in smoke mode we use numpy array
        input_im = input_im[:, :, :3]
        input_im = 255 - input_im  # invert from black to white
        input_im = np.asarray(input_im, dtype=np.float32)
        input_im = input_im / 255.0
        larger_side = max(input_im.shape[0], input_im.shape[1])
        new_img = np.ones((larger_side, larger_side, 3), dtype=np.float32)
        pad_h = (larger_side - input_im.shape[0]) // 2
        pad_w = (larger_side - input_im.shape[1]) // 2
        new_img[pad_h : pad_h + input_im.shape[0], pad_w : pad_w + input_im.shape[1], :] = input_im
        input_im = new_img

        input_im_np = input_im.copy() * 255.0
        input_im_np = input_im_np.astype(np.uint8)
        input_im_np = cv2.cvtColor(input_im_np, cv2.COLOR_RGB2BGR)

        cv2.imwrite(f"{save_path}/input_im.png", input_im_np)

    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].

        # old method: thresholding background, very important
        # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

        # new method: apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].

    print(f"Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.")
    print("new input_im:", lo(input_im))

    return input_im
