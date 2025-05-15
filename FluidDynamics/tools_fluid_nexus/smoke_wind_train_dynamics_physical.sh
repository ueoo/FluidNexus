#!/bin/sh
{
python entries_fluid_nexus/train_physical_particle.py \
    --loader fluid_nexus_real \
    --data_path /path/to/FluidNexusRoot/FluidNeuxs-Smoke \
    --config configs/fluid_nexus_smoke_dynamics_wind.json \
    --bg_load_path /path/to/FluidNexusRoot/fluid_nexus_dynamics_logs/fluid_nexus_smoke_background \
    --model_path /path/to/FluidNexusRoot/fluid_nexus_dynamics_logs/fluid_nexus_smoke_wind_physical_reconstruction

exit
}
