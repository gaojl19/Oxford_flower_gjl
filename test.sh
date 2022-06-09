python3.7 -m starter.analyze \
    --model_name VIT \
    --epoch_num 1 \
    --load_from_checkpoint ./log/ViT_vae_lr0.001/ \
    --test true \
    --save_result_dir ./output/vae/vit_102_ \
    --pretrain_type vae