#!/bin/bash



# Clear modules and load correct versions
conda activate cv_stytr
module purge
#module load anaconda-2020.11
module load cuda-10.1
module load cudnn-10.X-7.6.5


# Verify environment
echo "=== CUDA Info ==="
nvcc --version
nvidia-smi

python test.py --content_dir ../data/content_test_small/ --style_dir ../data/style_test_small/ --output out_selftrained &

TRAIN_PID=$!

echo "timestamp, name, index, memory.used [MiB], utilization.gpu [%]" > output.csv

while kill -0 $TRAIN_PID 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,name,index,memory.used,utilization.gpu --format=csv,noheader >> output.csv
    sleep 60
done

echo "Training completed. GPU monitoring stopped."
