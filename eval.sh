#! /bin/bash

# python3 eval.py \
#   model=model_forecast \
#   data_root="/data/jerome.zhou/prediction_dataset/av2" \
#   batch_size=64 \
#   checkpoint="./outputs/forecast-mae-forecast/2024-02-05/18-37-00/checkpoints/last.ckpt" \
#   test=True \

python3 eval.py \
  model=model_multiagent \
  data_root="/data/jerome.zhou/prediction_dataset/av2" \
  batch_size=16 \
  "checkpoint='./best_result/baseline_05_1.ckpt'" \
  test=True \
