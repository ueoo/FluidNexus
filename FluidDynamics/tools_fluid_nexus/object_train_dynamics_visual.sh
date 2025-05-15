#!/bin/sh
{
python entries_fluid_nexus/train_visual_particle.py \
    --loader fluid_nexus_real \
    --data_path /path/to/FluidNexusRoot/RealCaptureBlackBlueCloudOneData \
    --data_2_path /path/to/FluidNexusRoot/RealCaptureBlackBlueCloudRedBallOneData \
    --config configs/fluid_nexus_object.json \
    --bg_load_path /path/to/FluidNexusRoot/fluid_nexus_dynamics_logs/fluid_nexus_smoke_background \
    --bg_2_load_path /path/to/FluidNexusRoot/fluid_nexus_dynamics_logs/fluid_nexus_ball_background \
    --load_path /path/to/FluidNexusRoot/fluid_nexus_dynamics_logs/fluid_nexus_object_physical_reconstruction \
    --model_path /path/to/FluidNexusRoot/fluid_nexus_dynamics_logs/fluid_nexus_object_visual_reconstruction \

exit
}
