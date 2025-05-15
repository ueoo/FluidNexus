import os

from shutil import copyfile

from tqdm import tqdm


project_root = "/path/to/FluidNexusRoot"

dataset_name = "FluidNexusSmoke"
# dataset_name = "FluidNexusBall"

output_dataset_root = f"{project_root}/{dataset_name}_cogvideox_dataset"


output_dataset_videos_folder = os.path.join(output_dataset_root, "videos")
output_dataset_labels_folder = os.path.join(output_dataset_root, "labels")

# start_frame_ids = [i for i in range(60, 111, 10)]
start_frame_ids = [235]

sub_dataset_root = f"{project_root}/{dataset_name}_cogvideox_dataset_sub_235"
sub_dataset_videos_folder = os.path.join(sub_dataset_root, "videos")
sub_dataset_labels_folder = os.path.join(sub_dataset_root, "labels")
os.makedirs(sub_dataset_videos_folder, exist_ok=True)
os.makedirs(sub_dataset_labels_folder, exist_ok=True)

all_label_names = os.listdir(output_dataset_labels_folder)

num_videos = 0
for label_name in tqdm(all_label_names, desc="Copy labels"):
    label_path = os.path.join(output_dataset_labels_folder, label_name)
    start_frame = int(label_name.split("_")[9])

    if start_frame in start_frame_ids:
        video_name = label_name.replace(".txt", ".mp4")
        video_path = os.path.join(output_dataset_videos_folder, video_name)
        copyfile(video_path, os.path.join(sub_dataset_videos_folder, video_name))
        copyfile(label_path, os.path.join(sub_dataset_labels_folder, label_name))
        num_videos += 1

print(f"Number of copied videos: {num_videos}")
