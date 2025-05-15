import os
import subprocess

from glob import glob
from multiprocessing import Pool

import cv2
import imageio
import numpy as np

from moviepy.editor import VideoFileClip
from rich import print


FFMPEG_BIN = "ffmpeg"


def cmd_wrapper(program):
    # print(program)
    os.system(program)


def images_to_video(img_folder, img_name_match_str, output_vid_file, fps=30, verbose=False):
    assert "*" in img_name_match_str, "img_name_match_str should contain '*' for glob sorting"
    command = [
        f"{FFMPEG_BIN}",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        f"{fps}",
        "-pattern_type",
        "glob",
        "-i",
        f"{img_folder}/{img_name_match_str}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale='trunc(iw/2)*2:trunc(ih/2)*2'",
        "-y",
        f"{output_vid_file}",
    ]

    if verbose:
        print(f'Running "{" ".join(command)}"')
    subprocess.call(command)


def images_to_video_lex(img_folder, img_name_match_str, output_vid_file, fps=30, verbose=False):
    assert "%" in img_name_match_str, "img_name_match_str should contain '%' for geographic sorting"
    command = [
        f"{FFMPEG_BIN}",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        f"{fps}",
        "-i",
        f"{img_folder}/{img_name_match_str}",
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


def video_to_images(vid_file, img_folder, img_post_fix, fps=30, thumb_scale=1.0, verbose=False):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        f"{FFMPEG_BIN}",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        f"{vid_file}",
    ]

    if fps is not None and fps > 0:
        command += ["-vf", f"fps={fps}"]
    if thumb_scale != 1.0:
        command += ["-vf", f"scale=iw*{thumb_scale}:ih*{thumb_scale}"]

    command += [
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


def images_to_gif(img_folder, img_name_match_str, output_gif_path, duration=0.5, invert=False):
    images = []
    # List all files in the folder and sort them; assuming filenames have a sort-friendly format

    images_files = sorted(glob(f"{img_folder}/{img_name_match_str}"))

    for filename in images_files:
        file_path = os.path.join(img_folder, filename)
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


def video_to_gif(input_path, output_path, fps, start_time=0, end_time=None):
    # Load the video file
    clip = VideoFileClip(input_path)

    # If an end time is not specified, use the whole clip
    if end_time:
        clip = clip.subclip(start_time, end_time)
    else:
        clip = clip.subclip(start_time)

    # Write the result to a GIF file
    clip.write_gif(output_path, fps=fps)


def images_to_video_gif(img_folder, img_name_match_str="*.png", output_vid_file="output.mp4", fps=30, verbose=False):
    images_files = sorted(glob(f"{img_folder}/{img_name_match_str}"))
    if len(images_files) == 0:
        print(f"No images found in {img_folder}/{img_name_match_str}")
        return
    images_to_video(img_folder, img_name_match_str, output_vid_file, fps=fps, verbose=verbose)
    output_gif_file = output_vid_file.replace(".mp4", ".gif").replace(".avi", ".gif")
    # video_to_gif(output_vid_file, output_gif_file, fps)
    images_to_gif(img_folder, img_name_match_str, output_gif_file, duration=1 / fps)
