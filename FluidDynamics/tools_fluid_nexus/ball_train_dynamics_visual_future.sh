#!/bin/sh
{
python entries_fluid_nexus/train_visual_particle.py \
    --loader fluid_nexus_real \
    --data_path /path/to/FluidNexusRoot/FluidNeuxs-Ball \
    --config configs/fluid_nexus_ball_visual_future.json \
    --bg_load_path /path/to/FluidNexusRoot/fluid_nexus_dynamics_logs/fluid_nexus_ball_background \
    --load_path /path/to/FluidNexusRoot/fluid_nexus_dynamics_logs/fluid_nexus_ball_physical_reconstruction_future \
    --model_path /path/to/FluidNexusRoot/fluid_nexus_dynamics_logs/fluid_nexus_ball_visual_reconstruction_future

exit
}
