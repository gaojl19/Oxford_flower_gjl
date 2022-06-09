CUDA_VISIBLE_DEVICES=1,2 python3.7 run.py \
    --model_name ConvMixer \
    --epoch_num 200 \
    --learning_rate 0.001 \
    --save_dir ./log/ConvMixer \
    --save_model true