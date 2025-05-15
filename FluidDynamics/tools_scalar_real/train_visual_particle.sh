#!/bin/sh
{
python entries_scalar_real/train_visual_particle.py \
    --loader scalar_real \
    --data_path /path/to/FluidNexusRoot/ScalarRealAnother \
    --config configs/scalar_real.json \
    --load_path /path/to/FluidNexusRoot/fluid_nexus_dynamics_logs/scalar_real_physical_reconstruction \
    --model_path /path/to/FluidNexusRoot/fluid_nexus_dynamics_logs/scalar_real_visual_reconstruction \

exit
}
