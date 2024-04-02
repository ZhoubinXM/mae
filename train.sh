#! /bin/bash

# CUDA_VISIBLE_DEVICES=[1] \
python train.py \
       data_root=/data/jerome.zhou/prediction_dataset/av2 \
       model=model_multiagent_mae \
       gpus=[1,2,3,4,5,6,7] \
       batch_size=14 \
       monitor=val_AvgMinFDE \
       road_embed_type=transform \
       tempo_embed_type=norm \
       epochs=60 \
       post_norm=False \
       output=baseline_scene_query_scene_rms_bias_false_gelu_vector_full
