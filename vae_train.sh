# python3.7 starter/run_vae.py \
#     --model_name ConvMixer \
#     --epoch_num 20 \
#     --learning_rate 0.001 \
#     --save_model true \
#     --save_dir ./log/Conv_vae \
#     --data_dir ./5data

# python3.7 starter/run_vae.py \
#     --model_name VIT \
#     --epoch_num 20 \
#     --learning_rate 0.0001 \
#     --save_model true \
#     --save_dir ./log/ViT_vae \
#     --data_dir ./5data

python3.7 starter/run_vae.py \
    --model_name MLPMixer \
    --epoch_num 20 \
    --learning_rate 0.001 \
    --save_model true \
    --save_dir ./log/MLP_vae \
    --data_dir ./5data