#! /bin/bash

# python3 eval.py \
#   model=model_forecast \
#   data_root="/data/jerome.zhou/prediction_dataset/av2" \
#   batch_size=64 \
#   checkpoint="./outputs/forecast-mae-forecast/2024-02-05/18-37-00/checkpoints/last.ckpt" \
#   test=True \

python3 eval.py \
  model=model_mae_sept \
  data_root="/data/jerome.zhou/prediction_dataset/av2" \
  batch_size=64 \
  checkpoint="./outputs/model_mae_sept-forecast/2024-02-06/00-43-04/checkpoints/last.ckpt" \
  test=True \
