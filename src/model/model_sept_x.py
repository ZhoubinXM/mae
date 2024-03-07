"""This implemention of SEPT use X_TRNASFORMER in github
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.agent_embedding import AgentEmbeddingLayer
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.multimodal_decoder import MultimodalDecoder, SEPTMultimodalDecoder, SEPTProposeDecoder
from .layers.transformer_blocks import Block, CrossAttenderBlock
from .layers.relative_position_bias import RelativePositionBias

from x_transformers import Encoder, Decoder, CrossAttender


class ModelSEPTX(nn.Module):

    def __init__(self,
                 embed_dim=256,
                 tempo_depth=3,
                 roadnet_depth=3,
                 spa_depth=2,
                 num_heads=8,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 drop_path=0.2,
                 future_steps: int = 60,
                 history_steps: int = 50,
                 post_norm: bool = False,
                 road_embed_type: str = "transform",
                 tempo_embed_type: str = "T5",
                 viz: bool = True) -> None:
        super().__init__()
        self.viz = viz
        self.embed_dim = embed_dim
        self.road_embed_type = road_embed_type
        self.tempo_embed_type = tempo_embed_type
        self.project_agent_hist = nn.Sequential(nn.Linear(5, embed_dim),
                                                nn.ReLU(inplace=True))
        self.project_road = nn.Sequential(nn.Linear(8, embed_dim),
                                          nn.ReLU(inplace=True))
        self.agent_traj_query = nn.Embedding(6, embed_dim)
        # self.agent_traj_query = nn.Parameter(torch.randn(1, 6, embed_dim))

        self.tempo_net = Encoder(
            dim=embed_dim,
            heads=num_heads,
            dim_head=64,
            pre_norm=True,
            rel_pos_bias=
            True,  # adds relative positional bias to all attention layers, a la T5
            depth=3,
            ff_no_bias=True,  # set this to True
            layer_dropout=0.1,
        )

        self.road_net = Encoder(
            dim=embed_dim,
            heads=num_heads,
            dim_head=64,
            pre_norm=True,
            depth=3,
            ff_no_bias=True,  # set this to True
            layer_dropout=0.1,
        )

        self.spa_net = Encoder(
            dim=embed_dim,
            heads=num_heads,
            dim_head=64,
            pre_norm=True,
            depth=2,
            ff_no_bias=True,  # set this to True
            layer_dropout=0.1,
        )

        self.cross_attender = CrossAttender(
            dim=embed_dim,
            heads=num_heads,
            dim_head=64,
            pre_norm=True,
            depth=3,
            ff_no_bias=True,  # set this to True
            layer_dropout=0.1,
        )

        self.decoder = SEPTMultimodalDecoder(embed_dim,
                                             future_steps,
                                             hidden_dim=512)

        # self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def weight_init(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            fan_in = m.in_channels / m.groups
            fan_out = m.out_channels / m.groups
            bound = (6.0 / (fan_in + fan_out))**0.5
            nn.init.uniform_(m.weight, -bound, bound)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.MultiheadAttention):
            if m.in_proj_weight is not None:
                fan_in = m.embed_dim
                fan_out = m.embed_dim
                bound = (6.0 / (fan_in + fan_out))**0.5
                nn.init.uniform_(m.in_proj_weight, -bound, bound)
            else:
                nn.init.xavier_uniform_(m.q_proj_weight)
                nn.init.xavier_uniform_(m.k_proj_weight)
                nn.init.xavier_uniform_(m.v_proj_weight)
            if m.in_proj_bias is not None:
                nn.init.zeros_(m.in_proj_bias)
            nn.init.xavier_uniform_(m.out_proj.weight)
            if m.out_proj.bias is not None:
                nn.init.zeros_(m.out_proj.bias)
            if m.bias_k is not None:
                nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
            if m.bias_v is not None:
                nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
        elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for ih in param.chunk(4, 0):
                        nn.init.xavier_uniform_(ih)
                elif 'weight_hh' in name:
                    for hh in param.chunk(4, 0):
                        nn.init.orthogonal_(hh)
                elif 'weight_hr' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias_ih' in name:
                    nn.init.zeros_(param)
                elif 'bias_hh' in name:
                    nn.init.zeros_(param)
                    nn.init.ones_(param.chunk(4, 0)[1])
        elif isinstance(m, (nn.GRU, nn.GRUCell)):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for ih in param.chunk(3, 0):
                        nn.init.xavier_uniform_(ih)
                elif 'weight_hh' in name:
                    for hh in param.chunk(3, 0):
                        nn.init.orthogonal_(hh)
                elif 'bias_ih' in name:
                    nn.init.zeros_(param)
                elif 'bias_hh' in name:
                    nn.init.zeros_(param)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("net."):]: v
            for k, v in ckpt.items() if k.startswith("net.") and (
                set(['tempo_net', 'project_road', 'road_net', 'spa_net'])
                & set(k.split(".")))
        }
        print("[INFO]: Load pretrained weight: {}".format(ckpt_path))
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def forward(self, data):
        # 1. hist embedding
        hist_padding_mask = data["x_padding_mask"][:, :, :50]
        # time serires sequence mask; true means masked
        hist_key_padding_mask = data[
            "x_key_padding_mask"]  # agent padding mask
        hist_feat = torch.cat(
            [
                data["x_trans"][..., :50, :],  # [B, A, L, 2]
                data["x_velocity"][..., :50].unsqueeze(-1),
                data['x_angles'][..., :50].unsqueeze(-1),
                data['x_attr'][..., -1][..., None, None].repeat(1, 1, 50, 1)
            ],
            dim=-1,
        )  # [B, A, L, 5]
        B, A, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * A, L, D)
        hist_feat_key_padding = hist_key_padding_mask.view(B * A)
        hist_padding_mask = hist_padding_mask.view(-1,
                                                   hist_padding_mask.size(-1))

        # 1.1 projection layer
        actor_feat = self.project_agent_hist(
            hist_feat[~hist_feat_key_padding])  # [B*N, L, D_m]

        # 1.2 tempo net
        actor_feat = self.tempo_net(
            actor_feat, mask=~hist_padding_mask[~hist_feat_key_padding])

        # 1.3 max-pooling
        actor_feat_tmp = torch.zeros(B * A,
                                     actor_feat.shape[-1],
                                     device=actor_feat.device)
        actor_feat, _ = torch.max(actor_feat, dim=1)  # [B*N, D_m]
        actor_feat_tmp[~hist_feat_key_padding] = actor_feat
        actor_feat = actor_feat_tmp.reshape(B, A, -1)

        # 2. lane feat embedding
        lane_padding_mask = data["lane_padding_mask"]  # [B, R, 20]
        if self.road_embed_type != "full":
            lane_feat = torch.cat(
                [
                    data["lane_positions"][..., :19, :],  # [B, R, L, 2]
                    data["lane_positions"][..., 1:20, :],  # [B, R, L, 2]
                    data['lane_angles'][..., None, None].repeat(1, 1, 19, 1),
                    data['lane_attr'][..., None, :].repeat(1, 1, 19, 1),
                ],
                dim=-1,
            )  # [B, R, L, 4]
            lane_padding_mask = torch.cat([
                lane_padding_mask[..., :19][..., None],
                lane_padding_mask[..., 1:20][..., None],
            ],
                                          dim=-1).any(-1)
            B, M, L, D = lane_feat.shape
            lane_feat = lane_feat.view(-1, L, D)
            lane_padding_mask = lane_padding_mask.view(-1, L)
            lane_feat_key_padding = lane_padding_mask.all(-1)

            # 4. road projection layer
            lane_actor_feat = self.project_road(
                lane_feat[~lane_feat_key_padding])
            lane_actor_feat = self.road_net(
                lane_actor_feat,
                mask=~lane_padding_mask[~lane_feat_key_padding])

            lane_actor_feat_tmp = torch.zeros(B * M,
                                              lane_actor_feat.shape[-1],
                                              device=lane_actor_feat.device)
            lane_actor_feat = torch.mean(lane_actor_feat, dim=1)  # [B*N, D_m]
            lane_actor_feat_tmp[~lane_feat_key_padding] = lane_actor_feat
            lane_actor_feat = lane_actor_feat_tmp.reshape(B, M, -1)
        elif self.road_embed_type == "full":
            lane_padding_mask = data["lane_padding_mask_candidate"]  # [B, R]
            B, R = lane_padding_mask.shape

            lane_feat = data['lane_positions_candidate']

            lane_actor_feat = lane_feat.reshape(B*R, 8)
            lane_actor_feat = self.project_road(lane_actor_feat[~lane_padding_mask.reshape(B*R)])
            lane_actor_feat_tmp = torch.zeros(B * R,
                                              lane_actor_feat.shape[-1],
                                              device=lane_actor_feat.device)
            lane_actor_feat_tmp[~lane_padding_mask.reshape(B*R)] = lane_actor_feat
            lane_actor_feat = lane_actor_feat_tmp.reshape(B, R, -1)

        # 3. spa net
        x_encoder = torch.cat([actor_feat, lane_actor_feat], dim=1)

        if self.road_embed_type == "full":
            key_padding_mask = torch.cat(
                [data["x_key_padding_mask"],
                 lane_padding_mask.reshape(B, R)],
                dim=1)
        else:
            key_padding_mask = torch.cat([
                data["x_key_padding_mask"],
                lane_feat_key_padding.reshape(B, -1)
            ],
                                         dim=1)

        x_encoder = self.spa_net(x_encoder, mask=~key_padding_mask)

        # 4. cross-attender
        traj_query = self.agent_traj_query.weight.unsqueeze(0).repeat(B, 1, 1)
        # traj_query = self.agent_traj_query.repeat(x_encoder.size(0), 1, 1)

        traj_query = self.cross_attender(traj_query,
                                         x_encoder,
                                         context_mask=~key_padding_mask)

        # 5. MLP decoder
        y_hat, pi = self.decoder(traj_query)

        # only viz when batch_size = 1
        if self.viz and B == 1:
            save_dir = "./viz_0"
            import os
            import matplotlib.pyplot as plt
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            A, T, _ = hist_feat.shape
            hist_feat = hist_feat.detach().cpu()
            y = data['y'][0, 0].reshape(60, 2).cpu()
            fig, ax = plt.subplots(1, 2, figsize=(18,9))
            ax[0].plot(y[:, 0], y[:, 1], c='r', alpha=0.7)
            ax[1].plot(y[:, 0], y[:, 1], c='r', alpha=0.7)
            for j in range(A):
                if j == 0:
                    c = "k"
                else:
                    c = "g"
                ax[0].plot(hist_feat[j, :, 0],
                           hist_feat[j, :, 1],
                           c=c,
                           alpha=0.7)
                hist_feat_viz_j = hist_feat[j]
                seq_padding_mask = hist_padding_mask[j]
                valid_hist = hist_feat_viz_j[~seq_padding_mask]
                ax[1].plot(valid_hist[:, 0], valid_hist[:, 1], c=c, alpha=0.7)
            lane_feat = lane_feat.detach().cpu()
            if self.road_embed_type == "full":
                lane_feat = lane_feat.reshape(-1, 8)
                lane_all = data['lane_positions'].reshape(-1, 8).cpu()
                R, _ = lane_feat.shape
                for k in range(R):
                    ax[0].plot([lane_feat[k, 0], lane_feat[k, 2]],
                                [lane_feat[k, 1], lane_feat[k, 3]],
                                c='grey',)
                valid_lane = lane_feat[~lane_padding_mask.reshape(-1)]
                S, _ = valid_lane.shape
                for k in range(S):
                    ax[1].plot([valid_lane[k, 0], valid_lane[k, 2]],
                                [valid_lane[k, 1], valid_lane[k, 3]],
                                c='grey',)
                R, _ = lane_all.shape
                for k in range(R):
                    ax[0].plot([lane_all[k, 0], lane_all[k, 2]],
                                [lane_all[k, 1], lane_all[k, 3]],
                                c='grey',
                                alpha=0.4)
                S, _ = lane_all.shape
                for k in range(S):
                    ax[1].plot([lane_all[k, 0], lane_all[k, 2]],
                                [lane_all[k, 1], lane_all[k, 3]],
                                c='grey',
                                alpha=0.4)
            else:
              R, L, _ = lane_feat.shape
              for k in range(R):
                  ax[0].plot(lane_feat[k, :, 0],
                            lane_feat[k, :, 1],
                            c='grey',
                            alpha=0.5)
                  lane_feat_viz_k = lane_feat[k]
                  padding_mask = lane_padding_mask[k]
                  valid_lane = lane_feat_viz_k[~padding_mask]
                  # ax[1].plot(valid_lane[:, 0], valid_lane[:, 1], c='grey', alpha=0.4)
                  for u in range(valid_lane.shape[0]):
                      ax[1].plot([valid_lane[u, 0], valid_lane[u, 2]],
                                [valid_lane[u, 1], valid_lane[u, 3]],
                                c='grey',
                                alpha=0.4)
            plt.savefig(os.path.join(save_dir,
                                     f"{data['scenario_id'][0]}.jpg"))

        return {"y_hat": y_hat, "pi": pi}
