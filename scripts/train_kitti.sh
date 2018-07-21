#!/bin/bash

set -e

DATASET=kitti
IMAGE_SIZE=256
NUM_CHANNELS=3
BATCH_SIZE=1
NUM_BATCHES=100000
LEARNING_RATE=1e-4
DATASET_DIR=$HOME/data/datasets/kitti
OUTPUT_DIR=$HOME/tf_output/cycle/kitti_$(date "+%Y%m%d%H%M%S")

mkdir -p $OUTPUT_DIR
cp $(readlink -f $0) $OUTPUT_DIR

git log --pretty=format:'%H' -n 1 > $OUTPUT_DIR/commit.git


python train_segment.py \
	--dataset $DATASET \
	--data_dir $DATASET_DIR \
	--split train \
	--num_channels $NUM_CHANNELS \
	--batch_size $BATCH_SIZE \
	--num_batches $NUM_BATCHES \
	--image_size $IMAGE_SIZE \
	--output_dir $OUTPUT_DIR \
	--learning_rate $LEARNING_RATE