# python3.7 starter/run_simclr.py \
#     --model_name VIT \
#     --epoch_num 100 \
#     --save_dir ./log/ViT \
#     --learning_rate 0.0001 \
#     --save_model true \
#     --save_dir ./log/ViT_simclr_new \
#     --data_dir ./5data

# python3.7 starter/run_simclr.py \
#     --model_name ConvMixer \
#     --epoch_num 20 \
#     --learning_rate 0.001 \
#     --save_model true \
#     --save_dir ./log/Conv_simclr


python3.7 starter/run_simclr.py \
    --model_name MLPMixer \
    --epoch_num 20 \
    --learning_rate 0.001 \
    --save_model true \
    --save_dir ./log/MLP_simclr \
    --data_dir ./5data \
    --kernel horizontal_gradient_kernel