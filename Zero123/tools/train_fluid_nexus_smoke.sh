{
python main.py \
    -t \
    --logdir /path/to/FluidNexusRoot/zero123_finetune_logs/ \
    --base configs/fluid_nexus_smoke.yaml \
    --gpus 0,1,2,3 \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 10 \
    --finetune_from /path/to/FluidNexusRoot/zero123-weights/zero123-xl.ckpt \

exit
}
