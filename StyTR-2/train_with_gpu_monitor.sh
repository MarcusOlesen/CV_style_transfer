#!/bin/bash



# Clear modules and load correct versions
module purge
#module load anaconda-2020.11
module load cuda-10.1
module load cudnn-10.X-7.6.5


# Verify environment
echo "=== CUDA Info ==="
nvcc --version
nvidia-smi

python -u train.py \
  --style_dir ../data/style/train_unpacked_correctly \
  --content_dir ../data/content/train2017/train2017 \
  --save_dir models/ \
  --batch_size 2 &

TRAIN_PID=$!

echo "timestamp, name, index, memory.used [MiB], utilization.gpu [%]" > output.csv

while kill -0 $TRAIN_PID 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,name,index,memory.used,utilization.gpu --format=csv,noheader >> output.csv
    sleep 60
done

echo "Training completed. GPU monitoring stopped."
