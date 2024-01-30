#!/usr/bin/env bash

# Dataset URIs
# s3://argoverse/datasets/av2/sensor/ 
# s3://argoverse/datasets/av2/lidar/
# s3://argoverse/datasets/av2/motion-forecasting/
# s3://argoverse/datasets/av2/tbv/

export DATASET_NAME="motion-forecasting"  # sensor, lidar, motion_forecasting or tbv.
export TARGET_DIR="/data/jerome.zhou/prediction_dataset/av2"  # Target directory on your machine.

s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/$DATASET_NAME/*" $TARGET_DIR
