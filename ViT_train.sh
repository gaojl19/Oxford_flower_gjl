python3.7 starter/run.py \
    --model_name VIT \
    --epoch_num 100 \
    --load_from_checkpoint ./log/ViT_simclr_lr0.001/model_best.pkl
    --save_dir ./log/ViT/simclr \
    --learning_rate 0.0001 \
    --save_model true