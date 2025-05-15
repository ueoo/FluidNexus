#!/bin/sh
{
python entries_scalar_real/future_simulation.py \
    --loader scalar_real \
    --data_path /path/to/FluidNexusRoot/ScalarRealAnother \
    --config configs/scalar_real_future_simulation.json \
    --load_path /path/to/FluidNexusRoot/fluid_nexus_dynamics_logs/scalar_real_fluid_recontruction \
    --model_path /path/to/FluidNexusRoot/fluid_nexus_dynamics_logs/scalar_real_fluid_future_simulation \

exit
}
