#!/bin/bash

# Directories and commands
DIR1="./StyTR-2"
CMD1="python test.py --content_dir ../data/content_test_small/ --style_dir ../data/style_test_small/ --output out1"


# Run first test
conda activate stytr
echo "Changing to $DIR1 and running first test..."
cd "$DIR1" || { echo "Failed to cd into $DIR1"; exit 1; }
eval $CMD1
conda deactivate

cd ..
# what i need to run
DIR2="./AdaAttN"
cd "$DIR2" || { echo "Failed to cd into $DIR2"; exit 1; }

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

python test.py \
--content_path ../data/content_test_small \
--style_path ../data/style_test_small \
--name AdaAttN \
--model adaattn \
--dataset_mode unaligned \
--load_size 512 \
--crop_size 512 \
--image_encoder_path ./other/vgg_normalised.pth \
--gpu_ids 1 \
--skip_connection_3 \
--shallow_layer  \
--num_test 999999

wait

echo "All test jobs completed."
