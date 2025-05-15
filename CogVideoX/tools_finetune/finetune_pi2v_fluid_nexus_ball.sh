#! /bin/bash

echo "RUN on $(hostname)"

run_cmd="torchrun --standalone --nproc_per_node=4 train_image_to_video.py --base configs/cogvideox_5b_lora_prefixi2v.yaml configs/sft_pi2v_fluid_nexus_ball.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"
