echo "!! update dataset name and project_root first"
python fluid_nexus_real/create_cogvideox_dataset.py
python fluid_nexus_real/create_cogvideox_cams.py
python fluid_nexus_real/create_cogvideox_paths.py
python fluid_nexus_real/copy_cogvideox_val_dataset.py
