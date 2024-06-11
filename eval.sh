#! /bin/bash

# python3 eval.py \
#   model=model_forecast \
#   data_root="/data/jerome.zhou/prediction_dataset/av2" \
#   batch_size=64 \
#   checkpoint="./outputs/forecast-mae-forecast/2024-02-05/18-37-00/checkpoints/last.ckpt" \
#   test=True \

python3 eval.py \
  model=model_multiagent \
  data_root="/home/jerome.zhou/data/av2" \
  batch_size=4 \
  gpus=[0] \
  "checkpoint='/home/jerome.zhou/mae/outputs/qcnet_simpl/spa_6_propose_1_refine_1_scene_1_0.0002_4/use_N_change_mask_block/2024-06-04/04-37-31/checkpoints/last.ckpt'" \
  test=False \
