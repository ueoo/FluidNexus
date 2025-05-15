echo "!! update dataset name and project_root first"
python scalar_flow/preprocess.py
python scalar_flow/create_zero123_dataset.py
python scalar_flow/create_zero123_cams.py
python scalar_flow/create_zero123_paths.py
