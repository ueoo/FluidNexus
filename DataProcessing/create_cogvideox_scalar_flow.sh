echo "!! update dataset name and project_root first"
python scalar_flow/create_cogvideox_dataset.py
python scalar_flow/create_cogvideox_cams.py
python scalar_flow/create_cogvideox_paths.py
python scalar_flow/copy_cogvideox_val_dataset.py
