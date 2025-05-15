#! /bin/bash
{
# echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

run_cmd="$environs python gen_future_pi2v.py --base configs/cogvideox_5b_lora_prefixi2v.yaml configs_gen/sdedit_future_pi2v_fluid_nexus_ball.yaml --seed $RANDOM"
echo ${run_cmd}
eval ${run_cmd}


echo "DONE on `hostname`"
exit
}
