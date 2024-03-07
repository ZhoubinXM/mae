#! /bin/bash

# CUDA_VISIBLE_DEVICES=[1] \
python train.py \
       data_root=/data/jerome.zhou/prediction_dataset/av2 \
       model=model_multiagent_mae \
       gpus=[0,1,2,3,4,5,6,7] \
       batch_size=16 \
       monitor=val_AvgMinFDE \
       road_embed_type=transform \
       tempo_embed_type=norm \
       epochs=60 \
       post_norm=False \
       output=multiagent_16_qcnetx_dec_temo_token_road_full \
