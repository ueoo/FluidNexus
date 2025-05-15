#!/bin/sh
{
python entries_fluid_nexus/train_background.py \
    --data_path /path/to/FluidNexusRoot/FluidNeuxs-Smoke \
    --config configs/fluid_nexus_smoke_background.json \
    --loader fluid_nexus_real \
    --model_path /path/to/FluidNexusRoot/fluid_nexus_dynamics_logs/fluid_nexus_smoke_background

exit
}
