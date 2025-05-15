#!/bin/sh
{
python entries_fluid_nexus/train_visual_particle.py \
    --loader fluid_nexus_real \
    --data_path /path/to/FluidNexusRoot/FluidNeuxs-Smoke \
    --config configs/fluid_nexus_smoke_dynamics.json \
    --bg_load_path /path/to/FluidNexusRoot/fluid_nexus_dynamics_logs/fluid_nexus_smoke_background \
    --load_path /path/to/FluidNexusRoot/fluid_nexus_dynamics_logs/fluid_nexus_smoke_physical_reconstruction \
    --model_path /path/to/FluidNexusRoot/fluid_nexus_dynamics_logs/fluid_nexus_smoke_visual_reconstruction

exit
}
