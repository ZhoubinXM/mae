#! /bin/bash

# CUDA_VISIBLE_DEVICES=[1] \
python train.py \
       data_root=/data/jerome.zhou/prediction_dataset/av2 \
       model=model_sept \
       gpus=[7] \
       batch_size=96 \
       monitor=val_minFDE6 \
       road_embed_type=transform \
       tempo_embed_type=norm \
       epochs=50 \
       post_norm=False
