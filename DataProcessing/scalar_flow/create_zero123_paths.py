import json
import os


project_root = "/path/to/FluidNexusRoot"
scalar_flow_data_root = f"{project_root}/ScalarFlow_zero123_dataset"

num_total_sims = 104
num_val_sims = 10
paths_post = "10"
sim_frame_names = os.listdir(scalar_flow_data_root)
sim_frame_names = [name for name in sim_frame_names if "sim" in name and "frame" in name]

all_sim_names = [f"sim_{i:03d}" for i in range(num_total_sims)]

train_sim_names = all_sim_names[num_val_sims:]
val_sim_names = all_sim_names[:num_val_sims]


train_sim_frame_names = []
val_sim_frame_names = []
for sim_name in train_sim_names:
    train_sim_frame_names.extend([name for name in sim_frame_names if sim_name in name])

for sim_name in val_sim_names:
    val_sim_frame_names.extend([name for name in sim_frame_names if sim_name in name])

print(len(train_sim_names), len(train_sim_frame_names))
print(len(val_sim_names), len(val_sim_frame_names))
output_train_json = os.path.join(scalar_flow_data_root, f"train_paths{paths_post}.json")
output_val_json = os.path.join(scalar_flow_data_root, f"val_paths{paths_post}.json")

with open(output_train_json, "w") as f:
    json.dump(train_sim_frame_names, f)

with open(output_val_json, "w") as f:
    json.dump(val_sim_frame_names, f)
