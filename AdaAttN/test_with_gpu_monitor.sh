#!/bin/bash

conda activate adaattn

python test.py \
--content_path ../data/content_test_small \
--style_path ../data/style_test_small \
--name AdaAttN_retrain \
--model adaattn \
--dataset_mode unaligned \
--load_size 512 \
--crop_size 512 \
--image_encoder_path ./other/vgg_normalised.pth \
--gpu_ids 0 \
--skip_connection_3 \
--shallow_layer  \
--num_test 999999 &

TRAIN_PID=$!

echo "timestamp, name, index, memory.used [MiB], utilization.gpu [%]" > output.csv

while kill -0 $TRAIN_PID 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,name,index,memory.used,utilization.gpu --format=csv,noheader >> output.csv
    sleep 60
done

echo "Training completed. GPU monitoring stopped."
