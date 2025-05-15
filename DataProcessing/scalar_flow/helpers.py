import math
import os

import cv2
import numpy as np


def denoise(basename, folder_in, folder_out, perfect_denoise=False):
    in_path = os.path.join(folder_in, basename)
    image_gray = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    filename = os.path.join(folder_in, basename[:-8] + "%04d.png")
    curr_frame = int(basename[-8:-4])
    denoiseStrength = 3
    if (
        perfect_denoise
        and int(basename[-8:-4]) > 1
        and os.path.isfile(filename % (curr_frame + 1))
        and os.path.isfile(filename % (curr_frame + 2))
    ):
        images = [
            cv2.imread(filename % (curr_frame - 2), cv2.IMREAD_GRAYSCALE),
            cv2.imread(filename % (curr_frame - 1), cv2.IMREAD_GRAYSCALE),
            image_gray,
            cv2.imread(filename % (curr_frame + 1), cv2.IMREAD_GRAYSCALE),
            cv2.imread(filename % (curr_frame + 2), cv2.IMREAD_GRAYSCALE),
        ]
        image_gray = cv2.fastNlMeansDenoisingMulti(
            images, math.floor(len(images) / 2), len(images), None, denoiseStrength, 7, 21
        )
    else:
        image_gray = cv2.fastNlMeansDenoising(image_gray, None, denoiseStrength, 7, 21)
    cv2.imwrite(folder_out + basename, image_gray)


def separate_background(img_cur_name, img_first_name, folder_in, folder_out, threshold=8, scale=1.0):
    cur_img_path = os.path.join(folder_in, img_cur_name)

    if os.path.isfile(img_first_name):
        first_img_path = img_first_name
    else:
        first_img_path = os.path.join(folder_in, img_first_name)
    img_cur = cv2.imread(cur_img_path, cv2.IMREAD_GRAYSCALE)
    img_first = cv2.imread(first_img_path, cv2.IMREAD_GRAYSCALE)
    assert img_cur is not None, f"img_cur {cur_img_path} is None"
    assert img_first is not None, f"img_first {first_img_path} is None"

    img_sub = cv2.subtract(img_cur, img_first)
    ret, thres_out = cv2.threshold(img_sub, threshold, 255, cv2.THRESH_TOZERO)
    if scale > 1.0:
        scalar_out = thres_out.astype(np.float32) * 1.45
        scalar_out = scalar_out.astype(np.uint8)
    else:
        scalar_out = thres_out
    out_img_path = os.path.join(folder_out, img_cur_name)
    cv2.imwrite(out_img_path, scalar_out)
