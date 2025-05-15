import os
import subprocess

import cv2
import imageio
import numpy as np

from moviepy.editor import VideoFileClip


def cmd_wrapper(program):
    # print(program)
    os.system(program)


def images_to_video(img_folder, img_pre_fix, img_post_fix, output_vid_file, fps=30, verbose=False):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        f"/usr/bin/ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        f"{fps}",
        "-pattern_type",
        "glob",
        "-i",
        f"{img_folder}/{img_pre_fix}*{img_post_fix}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale='trunc(iw/2)*2:trunc(ih/2)*2'",
        "-y",
        f"{output_vid_file}",
    ]

    # print(f'Running "{" ".join(command)}"')
    subprocess.call(command)


def images_to_video_lex(img_folder, img_pre_fix, img_post_fix, output_vid_file, fps=30, verbose=False):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        f"/usr/bin/ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        f"{fps}",
        "-i",
        f"{img_folder}/{img_pre_fix}{img_post_fix}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale='trunc(iw/2)*2:trunc(ih/2)*2'",
        "-y",
        f"{output_vid_file}",
    ]

    # print(f'Running "{" ".join(command)}"')
    subprocess.call(command)


def create_white_images(img_folder):
    src_images = os.listdir(img_folder)
    src_images = [src_image for src_image in src_images if "_src" in src_image]
    for src_image in src_images:
        src_image_path = os.path.join(img_folder, src_image.replace("_src", "_white"))
        white_img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        cv2.imwrite(src_image_path, white_img)


def video_to_images(vid_file, img_folder, img_post_fix, fps=30, verbose=False):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        f"/usr/bin/ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        f"{vid_file}",
        "-vf",
        f"fps={fps}",
        "-qscale:v",
        "1",
        "-qmin",
        "1",
        "-qmax",
        "1",
        "-vsync",
        "0",
        f"{img_folder}/%06d{img_post_fix}",
    ]

    if verbose:
        print(f'Running "{" ".join(command)}"')
    subprocess.call(command)


def images_to_gif(folder_path, prefix, output_gif_path, duration=0.5, invert=False):
    images = []
    # List all files in the folder and sort them; assuming filenames have a sort-friendly format
    file_names = sorted(
        [f for f in os.listdir(folder_path) if f.endswith((".png", ".jpg", ".jpeg")) and f.startswith(prefix)]
    )

    for filename in file_names:
        file_path = os.path.join(folder_path, filename)
        # Read image using OpenCV
        image = cv2.imread(file_path)
        # Convert color from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image.shape[1] // 2 * 2, image.shape[0] // 2 * 2))
        if invert:
            image = 255 - image
        # Append the image to the list
        images.append(image)

    # Save the images as a gif using imageio
    imageio.mimsave(output_gif_path, images, duration=duration, loop=0)


def mp4_to_gif(input_path, output_path, start_time=0, end_time=None):
    # Load the video file
    clip = VideoFileClip(input_path)

    # If an end time is not specified, use the whole clip
    if end_time:
        clip = clip.subclip(start_time, end_time)
    else:
        clip = clip.subclip(start_time)

    # Write the result to a GIF file
    clip.write_gif(output_path)
