#!/bin/sh
{
python entries_fluid_nexus/train_background.py \
    --data_path /path/to/FluidNexusRoot/FluidNeuxs-Ball \
    --config configs/fluid_nexus_ball_background.json \
    --loader fluid_nexus_real \
    --model_path /path/to/FluidNexusRoot/fluid_nexus_dynamics_logs/fluid_nexus_ball_background

exit
}
