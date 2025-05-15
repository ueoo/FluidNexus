#!/bin/sh
{
python entries_fluid_nexus/future_simulation.py \
    --loader fluid_nexus_real \
    --data_path /path/to/FluidNexusRoot/FluidNeuxs-Ball \
    --config configs/fluid_nexus_ball_future_simulation.json \
    --bg_load_path /path/to/FluidNexusRoot/fluid_nexus_dynamics_logs/fluid_nexus_ball_background \
    --load_path /path/to/FluidNexusRoot/fluid_nexus_dynamics_logs/fluid_nexus_ball_physical_reconstruction \
    --model_path /path/to/FluidNexusRoot/fluid_nexus_dynamics_logs/fluid_nexus_ball_physical_future_simulation

exit
}
