# FluidNexus: 3D Fluid Reconstruction and Prediction From a Single Video

[![arXiv](https://img.shields.io/badge/arXiv-2503.04720-b31b1b)](https://arxiv.org/abs/2503.04720)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-blue)](https://arxiv.org/pdf/2503.04720)
[![Project Website](https://img.shields.io/badge/Project-Website-g)](https://yuegao.me/FluidNexus/)
[![Source Code](https://img.shields.io/badge/Github-Code-08872B?logo=github)](https://github.com/ueoo/FluidNexus/)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-orange)](https://huggingface.co/datasets/yuegao/FluidNexusDatasets)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-gold)](https://huggingface.co/yuegao/FluidNexusModels)

[**Yue Gao**\*](https://yuegao.me/), [**Hong-Xing "Koven" Yu**\*](https://kovenyu.com/), [**Bo Zhu**](https://faculty.cc.gatech.edu/~bozhu/), [**Jiajun Wu**](https://jiajunwu.com/)

[Stanford University](https://svl.stanford.edu/); [Microsoft](https://microsoft.com/); [Georgia Institute of Technology](https://www.gatech.edu/)

\* denotes equal contribution

![FluidNexus Teaser](./assets/teaser_vel.gif)

## 🚀 Get Started

> Don’t forget to update all `/path/to/FluidNexusRoot` to your real path. Find & Replace is your friend!

### Set Up Root Folder and Python Environment

```shell
mkdir -p /path/to/FluidNexusRoot

cd /path/to/FluidNexusRoot
git clone https://github.com/ueoo/FluidNexus.git

cd FluidNexus
conda env create -f fluid_nexus.yml

conda activate fluid_nexus

# Install the 3D Gaussian Splatting submodules
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
pip install submodules/gaussian_rasterization_ch3
pip install submodules/gaussian_rasterization_ch1
pip install submodules/simple-knn
pip install git+https://github.com/openai/CLIP.git
pip install xformers --index-url https://download.pytorch.org/whl/cu124
```

### Download the Datasets

Our **FluidNexus-Smoke** and **FluidNexus-Ball** datasets each include 120 scenes. Every scene contains 5 synchronized multi-view videos, with cameras arranged along a horizontal arc of approximately 120°.

* **FluidNexusSmoke** and **FluidNexusBall**: Processed datasets containing one example sample used in our paper.
* **FluidNexusSmokeAll** and **FluidNexusBallAll**: All samples processed into frames, usable within the FluidNexus framework.
* **FluidNexusSmokeAllRaw** and **FluidNexusBallAllRaw**: Raw videos of all samples as originally captured.

For **ScalarFlow**, please refer to the original [website](https://ge.in.tum.de/publications/2019-scalarflow-eckert/).

```shell
cd /path/to/FluidNexusRoot

# Download FluidNexus-Smoke FluidNexus-Ball ScalarReal datasets from Hugging Face
git clone https://huggingface.co/datasets/yuegao/FluidNexusDatasets

cd FluidNexusDatasets

unzip FluidNexusBall.zip
unzip FluidNexusBallAll.zip
unzip FluidNexusBallAllRaw.zip
unzip FluidNexusSmoke.zip
unzip FluidNexusSmokeAll.zip
unzip FluidNexusSmokeAllRaw.zip
unzip ScalarReal.zip

mv FluidNexusBall /path/to/FluidNexusRoot
mv FluidNexusBallAll /path/to/FluidNexusRoot
mv FluidNexusBallAllRaw /path/to/FluidNexusRoot
mv FluidNexusSmoke /path/to/FluidNexusRoot
mv FluidNexusSmokeAll /path/to/FluidNexusRoot
mv FluidNexusSmokeAllRaw /path/to/FluidNexusRoot
mv ScalarReal /path/to/FluidNexusRoot
```

### Frame-wise Novel View Synthesis

#### 1. Convert the frames to Zero123 input frames

```shell
cd /path/to/FluidNexusRoot/FluidNexus/DataProcessing

python convert_original_to_zero123.py
```

#### 2. Download the pretrained Zero123 and CogVideoX models

```shell
cd /path/to/FluidNexusRoot

# Zero123 base models
mkdir -p zero123_weights
cd zero123_weights
wget https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt

# CogVideoX base models
mkdir -p cogvideox-sat
# Please refer to the CogVideoX repo, we use the 1.0 version
# https://github.com/THUDM/CogVideo/blob/main/sat/README.md

# Our finetuned models
git clone https://huggingface.co/yuegao/FluidNexusModels

cd FluidNexusModels

mv zero123_finetune_logs /path/to/FluidNexusRoot
mv cogvideox_lora_ckpts /path/to/FluidNexusRoot
```

#### 3. Inference the frame-wise novel view synthesis model

Take `FluidNexus-Smoke` as an example, we assume the camera 2 is the middle camera, which is used as input:

```shell
cd /path/to/FluidNexusRoot/FluidNexus/Zero123

python inference/infer_fluid_nexus_smoke.py --tgt_cam 0
python inference/infer_fluid_nexus_smoke.py --tgt_cam 1
python inference/infer_fluid_nexus_smoke.py --tgt_cam 3
python inference/infer_fluid_nexus_smoke.py --tgt_cam 4
```

### Generative Video Refinement

#### 1. Convert Zero123 output frames to CogVideoX input frames

```shell
cd /path/to/FluidNexusRoot/FluidNexus/DataProcessing
python convert_zero123_to_cogvideox.py
```

#### 2. Inference the video generative models

```shell
cd /path/to/FluidNexusRoot/FluidNexus/CogVideoX

bash tools_gen/gen_zero123_pi2v_long_fluid_nexus_smoke.sh

bash tools_gen/gen_zero123_pi2v_long_fluid_nexus_ball.sh

bash tools_gen/gen_zero123_pi2v_long_scalar_real.sh
```

#### 3. Convert the video gen output frames to original frame format

```shell
cd /path/to/FluidNexusRoot/FluidNexus/DataProcessing

python convert_cogvideox_to_original.py
```

### Fluid Dynamics Reconstruction

#### 1. Optimize the background

Skip this step for ScalarReal dataset

```shell
cd /path/to/FluidNexusRoot/FluidNexus/FluidDynamics

# For FluidNeuxs-Smoke
bash tools_fluid_nexus/smoke_train_background.sh

# For FluidNeuxs-Ball
bash tools_fluid_nexus/ball_train_background.sh
```

#### 2. Optimize the physical particles

```shell
cd /path/to/FluidNexusRoot/FluidNexus/FluidDynamics

# For FluidNeuxs-Smoke
bash tools_fluid_nexus/smoke_train_dynamics_physical.sh

# For FluidNeuxs-Ball
bash tools_fluid_nexus/ball_train_dynamics_physical.sh

# For ScalarReal
bash tools_scalar_real/train_physical_particle.sh
```

#### 3. Optimize the visual particles

```shell
cd /path/to/FluidNexusRoot/FluidNexus/FluidDynamics

# For FluidNeuxs-Smoke
bash tools_fluid_nexus/smoke_train_dynamics_visual.sh

# For FluidNeuxs-Ball
bash tools_fluid_nexus/ball_train_dynamics_visual.sh

# For ScalarReal
bash tools_scalar_real/train_visual_particle.sh
```

🎊🎊 The results are located in `training_render`! 🎊🎊

## 🕰️ Future Prediction

### Physics simulation

Physics simulation is used to render rough multi-view future prediction frames.

```shell
cd /path/to/FluidNexusRoot/FluidNexus/FluidDynamics

# For FluidNeuxs-Smoke
bash tools_fluid_nexus/smoke_future_simulation.sh

# For FluidNeuxs-Ball
bash tools_fluid_nexus/ball_future_simulation.sh

# For ScalarReal
bash tools_scalar_real/future_simulation.sh
```

### Convert the simulation results to CogVideoX input format

```shell
cd /path/to/FluidNexusRoot/FluidNexus/DataProcessing

# FluidNexus-Smoke
# update the experiment name first
python convert_simulation_original_to_cogvideox.py

# FluidNexus-Ball
# update the experiment name first
python convert_simulation_original_to_cogvideox.py

# ScalarReal
python convert_simulation_original_to_cogvideox_unshift.py
```

### Generative video refinement on future prediction

Refine the rough multi-view frames.

```shell
cd /path/to/FluidNexusRoot/FluidNexus/CogVideoX

bash tools_gen/gen_future_pi2v_fluid_nexus_smoke.sh

bash tools_gen/gen_future_pi2v_fluid_nexus_ball.sh

bash tools_gen/gen_future_pi2v_scalar_real.sh
```

### Fluid dynamics reconstruction with future prediction

#### 1. Optimize the physical particles with future prediction

```shell
cd /path/to/FluidNexusRoot/FluidNexus/FluidDynamics

# For FluidNeuxs-Smoke
bash tools_fluid_nexus/smoke_train_dynamics_physical_future.sh

# For FluidNeuxs-Ball
bash tools_fluid_nexus/ball_train_dynamics_physical_future.sh

# For ScalarReal
bash tools_scalar_real/train_physical_particle_future.sh
```

#### 2. Optimize the visual particles with future prediction

```shell
cd /path/to/FluidNexusRoot/FluidNexus/FluidDynamics

# For FluidNeuxs-Smoke
bash tools_fluid_nexus/smoke_train_dynamics_visual_future.sh

# For FluidNeuxs-Ball
bash tools_fluid_nexus/ball_train_dynamics_visual_future.sh

# For ScalarReal
bash tools_scalar_real/train_visual_particle_future.sh
```

## 💨 Counterfactual Interaction Simulation - Wind

### Physics simulation with wind

```shell
cd /path/to/FluidNexusRoot/FluidNexus/FluidDynamics
bash tools_fluid_nexus/smoke_wind_simulation.sh
```

### Convert the simulation results to CogVideoX format

```shell
cd /path/to/FluidNexusRoot/FluidNexus/DataProcessing

# FluidNexus-Smoke wind interaction
# update the experiment name first
python convert_simulation_original_to_cogvideox.py
```

### Generative video refinement with wind

```shell
cd /path/to/FluidNexusRoot/FluidNexus/CogVideoX

bash tools_gen/gen_future_pi2v_fluid_nexus_smoke_wind.sh
```

### Fluid dynamics reconstruction with wind

#### 1. Optimize the physical particles with wind

```shell
cd /path/to/FluidNexusRoot/FluidNexus/FluidDynamics

bash tools_fluid_nexus/smoke_train_dynamics_physical_wind.sh
```

#### 2. Optimize the visual particles with wind

```shell
cd /path/to/FluidNexusRoot/FluidNexus/FluidDynamics

bash fluid_dynamics/tools_fluid_nexus/smoke_train_dynamics_visual_wind.sh
```

## 🔮 Counterfactual Interaction Simulation - Object

### Fluid dynamics reconstruction with object

#### 1. Optimize the physical particles with object

```shell
cd /path/to/FluidNexusRoot/FluidNexus/FluidDynamics

bash tools_fluid_nexus/object_train_dynamics_physical.sh
```

#### 2. Optimize the visual particles with object

```shell
cd /path/to/FluidNexusRoot/FluidNexus/FluidDynamics

bash fluid_dynamics/tools_fluid_nexus/object_train_dynamics_visual.sh
```

## 🚞 Zero123 Finetuning

### Create Zero123 datasets

```shell
cd /path/to/FluidNexusRoot/FluidNexus/DataProcessing

# FluidNexus-Smoke
bash create_zero123_fluid_nexus_smoke.sh

# FluidNexus-Ball
bash create_zero123_fluid_nexus_ball.sh

# ScalarFlow
bash create_zero123_scalar_flow.sh
```

### Finetune Zero123 models

```shell
cd /path/to/FluidNexusRoot/FluidNexus/Zero123
# FluidNexus-Smoke
bash tools/train_fluid_nexus_smoke.sh

# FluidNexus-Ball
bash tools/train_fluid_nexus_ball.sh

# ScalarFlow
bash tools/train_scalar_flow.sh
```

## 🚂 CogVideoX LoRA Finetuning

### Create CogVideoX datasets

```shell
cd /path/to/FluidNexusRoot/FluidNexus/DataProcessing
# FluidNexus-Smoke
bash create_cogvideox_fluid_nexus_smoke.sh

# FluidNexus-Ball
bash create_cogvideox_fluid_nexus_ball.sh

# ScalarFlow
bash create_cogvideox_scalar_flow.sh
```

### Finetune CogVideoX models

```shell
cd /path/to/FluidNexusRoot/FluidNexus/CogVideoX
# FluidNexus-Smoke
bash tools_finetune/finetune_pi2v_fluid_nexus_smoke.sh

# FluidNexus-Ball
bash tools_finetune/finetune_pi2v_fluid_nexus_ball.sh

# ScalarFlow
bash tools_finetune/finetune_pi2v_scalar_flow.sh
```

## 🌴 Acknowledgements

Thanks to these great repositories: [SpacetimeGaussians](https://github.com/oppo-us-research/SpacetimeGaussians), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [HyFluid](https://github.com/y-zheng18/HyFluid), [CogVideo](https://github.com/THUDM/CogVideo), [Zero123](https://github.com/cvlab-columbia/zero123), [diffusers](https://github.com/huggingface/diffusers) and many other inspiring works in the community.

We sincerely thank the anonymous reviewers of CVPR 2025 for their helpful feedbacks.

## ⭐️ Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@inproceedings{gao2025fluidnexus,
    title     = {FluidNexus: 3D Fluid Reconstruction and Prediction from a Single Video},
    author    = {Gao, Yue and Yu, Hong-Xing and Zhu, Bo and Wu, Jiajun},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
}
```
