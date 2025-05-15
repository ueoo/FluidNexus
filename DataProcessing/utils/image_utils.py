import math
import os
import re

import cv2
import numpy as np
import torch

from PIL import Image


def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = torch.from_numpy(img).float().cuda()
    img = img.permute(2, 0, 1)
    return img


def create_white_images(img_folder):
    src_images = os.listdir(img_folder)
    src_images = [src_image for src_image in src_images if "_src" in src_image]
    for src_image in src_images:
        src_image_path = os.path.join(img_folder, src_image.replace("_src", "_white"))
        white_img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        cv2.imwrite(src_image_path, white_img)


def denoise_gray(in_path, out_path, perfect_denoise=False, denoise_strength=3):
    image_gray = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)

    curr_frame = in_path.split("/")[-1].split(".")[0]
    curr_frame = int(curr_frame)
    prev_frame = curr_frame - 1
    next_frame = curr_frame + 1
    prev_frame_2 = curr_frame - 2
    next_frame_2 = curr_frame + 2
    prev_path = in_path.replace(f"{curr_frame:04d}", f"{prev_frame:04d}")
    next_path = in_path.replace(f"{curr_frame:04d}", f"{next_frame:04d}")
    prev_path_2 = in_path.replace(f"{curr_frame:04d}", f"{prev_frame_2:04d}")
    next_path_2 = in_path.replace(f"{curr_frame:04d}", f"{next_frame_2:04d}")
    if (
        perfect_denoise
        and os.path.isfile(prev_path)
        and os.path.isfile(next_path)
        and os.path.isfile(prev_path_2)
        and os.path.isfile(next_path_2)
    ):

        images = [
            cv2.imread(prev_path_2, cv2.IMREAD_GRAYSCALE),
            cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE),
            image_gray,
            cv2.imread(next_path, cv2.IMREAD_GRAYSCALE),
            cv2.imread(next_path_2, cv2.IMREAD_GRAYSCALE),
        ]
        image_gray = cv2.fastNlMeansDenoisingMulti(
            images, math.floor(len(images) / 2), len(images), None, denoise_strength, 7, 21
        )
    else:
        image_gray = cv2.fastNlMeansDenoising(image_gray, None, denoise_strength, 7, 21)
    cv2.imwrite(out_path, image_gray)


def denoise_color(img_cur_name, folder_in, folder_out, denoise_window=5):
    cur_frame_idx = int(img_cur_name.split(".")[0])
    images = []
    for frame_ix in range(cur_frame_idx - denoise_window, cur_frame_idx + denoise_window + 1):
        frame_path = f"{folder_in}/{frame_ix:06d}.png"
        if os.path.exists(frame_path):
            images.append(cv2.imread(frame_path))

    assert len(images) == denoise_window * 2 + 1, f"No images found for {img_cur_name}"
    denoised_image = cv2.fastNlMeansDenoisingColoredMulti(images, len(images) // 2, len(images), None, 3, 3, 7, 21)
    out_path = f"{folder_out}/{img_cur_name}"
    cv2.imwrite(out_path, denoised_image)


def denoise_color_another_name_pattern(img_cur_name, folder_in, folder_out, denoise_window=5, frame_nums=120):
    cur_frame_idx = int(re.search(r"frame(\d+)_", img_cur_name).group(1))
    images = []

    if cur_frame_idx - denoise_window < 0 or cur_frame_idx + denoise_window >= frame_nums:
        in_image_path = os.path.join(folder_in, img_cur_name)
        image = cv2.imread(in_image_path)
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        out_image_path = os.path.join(folder_out, img_cur_name)
        cv2.imwrite(out_image_path, image)

    else:
        for frame_ix in range(cur_frame_idx - denoise_window, cur_frame_idx + denoise_window + 1):
            target_frame_name = img_cur_name.replace(f"frame{cur_frame_idx:03d}", f"frame{frame_ix:03d}")
            frame_path = f"{folder_in}/{target_frame_name}"
            if os.path.exists(frame_path):
                images.append(cv2.imread(frame_path))

        assert len(images) == denoise_window * 2 + 1, f"No images found for {img_cur_name}"
        denoised_image = cv2.fastNlMeansDenoisingColoredMulti(images, len(images) // 2, len(images), None, 3, 3, 7, 21)
        out_path = f"{folder_out}/{img_cur_name}"
        cv2.imwrite(out_path, denoised_image)


def adjust_gamma_pil(image_path, output_path, gamma):
    # Open the image file
    img = Image.open(image_path).convert("L")  # Convert to grayscale

    # Apply gamma correction
    img = img.point(lambda x: 255 * ((x / 255) ** gamma))

    # Save the modified image
    img.save(output_path)


def adjust_gamma(image_path, output_path, gamma):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read directly as grayscale
    # print(f"{image_path} img", img.min(), img.max())
    img[img <= 2] = 0

    # # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # # Apply gamma correction using the lookup table
    adjusted_img = cv2.LUT(img, table)
    adjusted_img = np.clip(adjusted_img, 0, 255).astype(np.uint8)
    # Save the result
    cv2.imwrite(output_path, adjusted_img)


def pad_square(img):
    h, w = img.shape[:2]
    if h > w:
        pad = (h - w) // 2
        img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    elif h < w:
        pad = (w - h) // 2
        img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
    return img


def shift_image(image, offset_h, offset_w):
    if offset_h == 0 and offset_w == 0:
        return image
    shifted_image = np.zeros_like(image)

    # Perform the shift
    if offset_h > 0 and offset_w > 0:
        shifted_image[offset_h:, offset_w:, :] = image[:-offset_h, :-offset_w, :]
    elif offset_h > 0 and offset_w < 0:
        shifted_image[offset_h:, :offset_w, :] = image[:-offset_h, -offset_w:, :]
    elif offset_h < 0 and offset_w > 0:
        shifted_image[:offset_h, offset_w:, :] = image[-offset_h:, :-offset_w, :]
    elif offset_h < 0 and offset_w < 0:
        shifted_image[:offset_h, :offset_w, :] = image[-offset_h:, -offset_w:, :]
    elif offset_h > 0 and offset_w == 0:
        shifted_image[offset_h:, :, :] = image[:-offset_h, :, :]
    elif offset_h < 0 and offset_w == 0:
        shifted_image[:offset_h, :, :] = image[-offset_h:, :, :]
    elif offset_h == 0 and offset_w > 0:
        shifted_image[:, offset_w:, :] = image[:, :-offset_w, :]
    elif offset_h == 0 and offset_w < 0:
        shifted_image[:, :offset_w, :] = image[:, -offset_w:, :]

    return shifted_image


def unshift(in_path, out_path, view_name, hack_type="scalar"):
    # these offset is used in dataset_readers.py of FluidDynamics
    # if img_offset:
    #     if cam_name == "0":
    #         image = shift_image(image, -12, 18)
    #     if cam_name == "1":
    #         image = shift_image(image, 52, 18)
    #     if cam_name == "3":
    #         image = shift_image(image, 11, -12)
    #     if cam_name == "4":
    #         image = shift_image(image, 11, -18)

    image = cv2.imread(in_path)
    if hack_type == "scalar":
        ### scalar_real hack
        if view_name == "train00":
            offset_h = 12
            offset_w = -18
        elif view_name == "train01":
            offset_h = -52
            offset_w = -18
        elif view_name == "train02":
            offset_h = 0
            offset_w = 0
        elif view_name == "train03":
            offset_h = -11
            offset_w = 12
        elif view_name == "train04":
            offset_h = -11
            offset_w = 18
        else:
            raise ValueError(f"Unknown view name: {view_name}")
    else:
        raise ValueError(f"Unknown hack type: {hack_type}")

    unshifted_image = shift_image(image, offset_h, offset_w)
    cv2.imwrite(out_path, unshifted_image)


def crop_bbox(in_path, out_path, crop_back_path, crop_white_path, crop_box):
    image = cv2.imread(in_path)
    # image = cv2.rectangle(
    #     image, (crop_box[0], crop_box[1]), (crop_box[0] + crop_box[2], crop_box[1] + crop_box[3]), (0, 255, 0), 2
    # )
    image_cropped = image[crop_box[1] : crop_box[1] + crop_box[3], crop_box[0] : crop_box[0] + crop_box[2]]
    cv2.imwrite(out_path, image_cropped)

    back_image = np.zeros_like(image)
    back_image[crop_box[1] : crop_box[1] + crop_box[3], crop_box[0] : crop_box[0] + crop_box[2]] = image_cropped
    cv2.imwrite(crop_back_path, back_image)

    white_bg_image = 255 - image_cropped
    cv2.imwrite(crop_white_path, white_bg_image)


def crop_center(image):
    h, w = image.shape[:2]
    new_size = min(h, w)
    top = (h - new_size) // 2
    left = (w - new_size) // 2
    bottom = top + new_size
    right = left + new_size
    return image[top:bottom, left:right]


def crop_image_to_aspect_ratio(image, new_height, new_width):
    # Open an image file
    original_height, original_width = image.shape[:2]

    # Calculate the new dimensions keeping the aspect ratio
    target_ratio = new_width / new_height
    original_ratio = original_width / original_height

    if target_ratio > original_ratio:
        # Crop height to fit the new aspect ratio
        new_height = int(original_width / target_ratio)
        top = (original_height - new_height) // 2
        bottom = top + new_height
        left = 0
        right = original_width
    else:
        # Crop width to fit the new aspect ratio
        new_width = int(original_height * target_ratio)
        left = (original_width - new_width) // 2
        right = left + new_width
        top = 0
        bottom = original_height

    # Crop the image to the new dimensions
    cropped_image = image[top:bottom, left:right]

    return cropped_image


def preprocess_image(input_im, out_path):
    """
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    """

    # in smoke mode we use numpy array
    input_im = input_im[:, :, :3].copy()
    input_im = 255 - input_im

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

    cv2.imwrite(out_path, input_im_np)

    input_im_np_512 = cv2.resize(input_im_np, (512, 512))
    input_im_np_256 = cv2.resize(input_im_np, (256, 256))
    cv2.imwrite(out_path.replace(".png", "_256.png"), input_im_np_256)
    cv2.imwrite(out_path.replace(".png", "_512.png"), input_im_np_512)


def prepare_stable_image(in_path, out_path, width_new=1024, height_new=576, bg_color=[0, 0, 0], invert=False):
    # Original dimensions
    input_im = cv2.imread(in_path)
    height_original, width_original = input_im.shape[:2]

    # Calculating the ratio of new dimensions
    ratio_width = width_new / width_original
    ratio_height = height_new / height_original

    # Choosing the smallest ratio to maintain aspect ratio
    ratio = min(ratio_width, ratio_height)

    # Calculating new dimensions
    new_width = int(width_original * ratio)
    new_height = int(height_original * ratio)

    # Resizing the image
    resized_image = cv2.resize(input_im, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # To make sure the resized image fits exactly the desired dimensions (adding black border if necessary)
    final_image = cv2.copyMakeBorder(
        resized_image,
        top=int((height_new - new_height) / 2),
        bottom=int((height_new - new_height) / 2),
        left=int((width_new - new_width) / 2),
        right=int((width_new - new_width) / 2),
        borderType=cv2.BORDER_CONSTANT,
        value=bg_color,  # Black border
    )
    if invert:
        final_image = 255 - final_image
    cv2.imwrite(out_path, final_image)


def prepare_generative_image(
    in_path,
    out_path,
    white_out_path=None,
    width_new=1024,
    height_new=576,
    bg_color=[0, 0, 0],
    source_is_white=False,
):
    # Original dimensions
    input_im = cv2.imread(in_path)
    if source_is_white:
        # if the source image is white, invert the image
        input_im = 255 - input_im
    height_original, width_original = input_im.shape[:2]

    # Calculating the ratio of new dimensions
    ratio_width = width_new / width_original
    ratio_height = height_new / height_original

    # Choosing the smallest ratio to maintain aspect ratio
    ratio = min(ratio_width, ratio_height)

    # Calculating new dimensions
    new_width = int(width_original * ratio)
    new_height = int(height_original * ratio)

    # Resizing the image
    resized_image = cv2.resize(input_im, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # To make sure the resized image fits exactly the desired dimensions (adding black border if necessary)
    final_image = cv2.copyMakeBorder(
        resized_image,
        top=int((height_new - new_height) / 2),
        bottom=int((height_new - new_height) / 2),
        left=int((width_new - new_width) / 2),
        right=int((width_new - new_width) / 2),
        borderType=cv2.BORDER_CONSTANT,
        value=bg_color,  # Black border
    )
    cv2.imwrite(out_path, final_image)
    if white_out_path is None:
        return
    white_final_image = 255 - final_image
    cv2.imwrite(white_out_path, white_final_image)


def prepare_generative_image_crop_first(
    in_path,
    out_path,
    white_out_path=None,
    width_new=1024,
    height_new=576,
    bg_color=[0, 0, 0],
    source_is_white=False,
):
    # Original dimensions
    input_im = cv2.imread(in_path)
    if source_is_white:
        # if the source image is white, invert the image
        input_im = 255 - input_im
    height_original, width_original = input_im.shape[:2]
    crop_width = int(256 * (1080 / 1920))
    crop_left = (width_original - crop_width) // 2

    crop_input_im = input_im[:, crop_left : crop_left + crop_width, :]
    height_original, width_original = crop_input_im.shape[:2]

    # Calculating the ratio of new dimensions
    ratio_width = width_new / width_original
    ratio_height = height_new / height_original

    # Choosing the smallest ratio to maintain aspect ratio
    ratio = min(ratio_width, ratio_height)

    # Calculating new dimensions
    new_width = int(width_original * ratio)
    new_height = int(height_original * ratio)

    # Resizing the image
    resized_image = cv2.resize(crop_input_im, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # To make sure the resized image fits exactly the desired dimensions (adding black border if necessary)
    final_image = cv2.copyMakeBorder(
        resized_image,
        top=int((height_new - new_height) / 2),
        bottom=int((height_new - new_height) / 2),
        left=int((width_new - new_width) / 2),
        right=int((width_new - new_width) / 2),
        borderType=cv2.BORDER_CONSTANT,
        value=bg_color,  # Black border
    )
    cv2.imwrite(out_path, final_image)
    if white_out_path is None:
        return
    white_final_image = 255 - final_image
    cv2.imwrite(white_out_path, white_final_image)


def prepare_square_image(in_path, out_path, white_out_path=None):
    img = cv2.imread(in_path)
    height, width = img.shape[:2]
    size = max(height, width)
    new_img = cv2.copyMakeBorder(
        img,
        top=(size - height) // 2,
        bottom=(size - height) // 2,
        left=(size - width) // 2,
        right=(size - width) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],  # Black border
    )
    cv2.imwrite(out_path, new_img)
    if white_out_path is None:
        return
    white_final_image = 255 - new_img
    cv2.imwrite(white_out_path, white_final_image)


def crop_and_resize(in_path, out_path, new_width=1080, new_height=1920, to_gray=False):
    img = cv2.imread(in_path)
    ratio = new_width / new_height
    h, w = img.shape[:2]
    # Desired aspect ratio is 9:16, target size is 1080x1920
    target_width = h * ratio

    # Calculate the amount to crop
    crop_width = int(target_width)
    crop_x = (w - crop_width) // 2

    # Crop the image to the target aspect ratio
    cropped_image = img[:, crop_x : crop_x + crop_width]

    # Resize the cropped image to 1080x1920
    resized_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    if to_gray:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(out_path, resized_image)


def convert_zero123_out_to_raw_style(image, target_h=1920, target_w=1080):
    # 256x256 white bg to 1080x1920 (hxw) ratio black bg
    height, width = image.shape[:2]
    # Open the original image
    # Calculate new dimensions for the crop
    target_ratio = target_w / target_h
    new_width = int(height * target_ratio)
    new_height = height

    # Calculate the cropping area, centered on the new width
    left = (width - new_width) // 2
    top = 0
    right = left + new_width
    bottom = height

    # Crop the image
    img_cropped = image[top:bottom, left:right]

    # Resize the cropped image to the target size
    img_resized = cv2.resize(img_cropped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    img_resized = 255 - img_resized  # Invert the image
    # Save the resized image
    return img_resized


def separate_background(
    img_cur_name,
    img_first_name,
    folder_in,
    folder_out,
    emitter_bbox=None,
    threshold=8,
    post_threshold=8,
    scale=1.0,
):
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
    thres_out = cv2.threshold(img_sub, threshold, 255, cv2.THRESH_TOZERO)[1]
    if emitter_bbox is not None:
        x, y, w, h = emitter_bbox
        thres_out[y : y + h, x : x + w] = 0
    if scale > 0.0 and scale != 1.0:
        scalar_out = thres_out.astype(np.float32) * scale
        scalar_out = scalar_out.astype(np.uint8)
    else:
        scalar_out = thres_out
    post_out = cv2.threshold(scalar_out, post_threshold, 255, cv2.THRESH_TOZERO)[1]
    out_img_path = os.path.join(folder_out, img_cur_name)
    cv2.imwrite(out_img_path, post_out)


def scale_frame(image_basename, folder_in, folder_out, scale_value=1.0):
    img_path = os.path.join(folder_in, image_basename)
    img = cv2.imread(img_path)
    if scale_value != 1.0:
        img = img.astype(np.float32)
        img = img * scale_value
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
    out_img_path = os.path.join(folder_out, image_basename)
    cv2.imwrite(out_img_path, img)


def adjust_brightness_to_match(frame, target_brightness):
    # Convert the frame to HSV (hue, saturation, value) to easily adjust brightness
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Compute the current brightness
    current_brightness = np.mean(v)

    # Calculate the scaling factor to adjust to the target brightness
    scale_factor = target_brightness / current_brightness if current_brightness > 0 else 1

    # Adjust the brightness
    v = np.clip(v * scale_factor, 0, 255).astype(np.uint8)

    # Merge the channels back and convert to BGR
    adjusted_hsv = cv2.merge([h, s, v])
    adjusted_frame = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)

    return adjusted_frame


def normalize_brightness_across_videos(frames):
    # Compute the mean brightness across all frames
    brightness_values = []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        brightness_values.append(np.mean(v))

    # Compute target brightness as the mean of all brightness values
    target_brightness = np.mean(brightness_values)

    # Adjust each frame to the target brightness
    adjusted_frames = [adjust_brightness_to_match(frame, target_brightness) for frame in frames]

    return adjusted_frames


def normalize_brightness(cur_frame_name, folders_in, folders_out):
    frames = [cv2.imread(os.path.join(folder, cur_frame_name)) for folder in folders_in]
    adjusted_frames = normalize_brightness_across_videos(frames)

    for frame, folder_out in zip(adjusted_frames, folders_out):
        out_frame_path = os.path.join(folder_out, cur_frame_name)
        cv2.imwrite(out_frame_path, frame)


def crop_frame(cur_frame_name, folder_in, folder_out, crop_box=[1080, 1920]):
    img_path = os.path.join(folder_in, cur_frame_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image {img_path} is None")
        return
    h, w = img.shape[:2]
    if len(crop_box) == 2:
        # crop_box is the center of the crop box and the size of the crop box
        # crop_box = [w, h]
        target_w = crop_box[0]
        target_h = crop_box[1]
        top = (h - target_h) // 2
        left = (w - target_w) // 2
    elif len(crop_box) == 4:
        # crop_box is the top-left corner of the crop box and the size of the crop box
        # crop_box = [x, y, w, h]
        top = crop_box[1]
        left = crop_box[0]
        target_w = crop_box[2]
        target_h = crop_box[3]
    else:
        raise ValueError(f"Unknown crop_box format: {crop_box}")
    img_cropped = img[top : top + target_h, left : left + target_w]
    out_img_path = os.path.join(folder_out, cur_frame_name)
    cv2.imwrite(out_img_path, img_cropped)


def resize_frame(cur_frame_name, folder_in, folder_out, resize_to=[1080, 1920]):
    img_path = os.path.join(folder_in, cur_frame_name)
    assert os.path.exists(img_path), f"Image {img_path} does not exist"
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    target_w = resize_to[0]
    target_h = resize_to[1]
    if h != target_h or w != target_w:
        if h < target_h or w < target_w:
            img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        else:
            img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img

    out_img_path = os.path.join(folder_out, cur_frame_name)
    cv2.imwrite(out_img_path, img_resized)


def crop_resize_frame(
    cur_frame_name,
    folder_in,
    folder_crop,
    folder_resize,
    crop_box=[1080, 1920],
    resize_to=[1080, 1920],
):
    crop_frame(cur_frame_name, folder_in, folder_crop, crop_box)
    resize_frame(cur_frame_name, folder_crop, folder_resize, resize_to)


def adjust_white_balance(source_img, target_img):
    """
    Adjusts the white balance of the source image to match the target image.
    """
    # Convert images to LAB color space
    source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Compute mean and std deviation for each channel
    src_l_mean, src_l_std = cv2.meanStdDev(source_lab[:, :, 0])
    src_a_mean, src_a_std = cv2.meanStdDev(source_lab[:, :, 1])
    src_b_mean, src_b_std = cv2.meanStdDev(source_lab[:, :, 2])

    tgt_l_mean, tgt_l_std = cv2.meanStdDev(target_lab[:, :, 0])
    tgt_a_mean, tgt_a_std = cv2.meanStdDev(target_lab[:, :, 1])
    tgt_b_mean, tgt_b_std = cv2.meanStdDev(target_lab[:, :, 2])

    # Subtract the mean from source
    l, a, b = cv2.split(source_lab)
    l -= src_l_mean[0][0]
    a -= src_a_mean[0][0]
    b -= src_b_mean[0][0]

    # Scale by the standard deviations
    l = (tgt_l_std[0][0] / src_l_std[0][0]) * l
    a = (tgt_a_std[0][0] / src_a_std[0][0]) * a
    b = (tgt_b_std[0][0] / src_b_std[0][0]) * b

    # Add the mean of target
    l += tgt_l_mean[0][0]
    a += tgt_a_mean[0][0]
    b += tgt_b_mean[0][0]

    # Clip the values and merge back
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    adjusted_lab = cv2.merge([l, a, b])
    adjusted_img = cv2.cvtColor(adjusted_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return adjusted_img


def apply_color_to_grayscale(color_img, grayscale_img):
    # Check if dimensions match; if not, resize grayscale to match color image
    if color_img.shape[:2] != grayscale_img.shape[:2]:
        grayscale_img = cv2.resize(
            grayscale_img, (color_img.shape[1], color_img.shape[0]), interpolation=cv2.INTER_AREA
        )

    # Convert the color image to LAB color space
    color_lab = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)

    # Split LAB channels
    L_color, A_color, B_color = cv2.split(color_lab)

    # Normalize the grayscale image to match the L channel's range
    # OpenCV's L channel ranges from 0 to 255
    L_grayscale = grayscale_img.copy()

    # Optionally, you can adjust the contrast of the grayscale image to better match
    # # For example, using histogram equalization:
    # L_grayscale = cv2.equalizeHist(grayscale_img)

    # Merge the grayscale L channel with the color A and B channels
    merged_lab = cv2.merge((L_grayscale, A_color, B_color))

    # Convert back to BGR color space
    colorized_img = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    return colorized_img


def resize_frame_keep_ratio(frame, target_width):
    # Calculate the ratio of the new width to the original width
    ratio = target_width / frame.shape[1]

    # Resize the frame to the target width while maintaining the aspect ratio
    resized_frame = cv2.resize(frame, (target_width, int(frame.shape[0] * ratio)))

    return resized_frame
