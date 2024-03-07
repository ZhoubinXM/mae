from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.agent_embedding import AgentEmbeddingLayer
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.multimodal_decoder import MultimodalDecoder, SEPTMultimodalDecoder, SEPTMAEDecoder
from .layers.transformer_blocks import Block, CrossAttenderBlock
from .layers.relative_position_bias import RelativePositionBias


class ModelSEPTMAE(nn.Module):

    def __init__(
        self,
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
    ) -> None:
        super().__init__()
        self.road_embed_type = road_embed_type
        self.tempo_embed_type = tempo_embed_type
        self.project_agent_hist = nn.Sequential(nn.Linear(5, embed_dim),
                                                nn.GELU())
        self.tempo_net = nn.ModuleList(
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  attn_drop=0.2,
                  post_norm=post_norm,
                  attn_type=tempo_embed_type) for i in range(tempo_depth))
        if self.tempo_embed_type == "norm":
            self.traj_pos_enc = nn.Embedding(50, embed_dim)
        if self.road_embed_type == "transform":
            self.project_road = nn.Sequential(nn.Linear(8, embed_dim),
                                              nn.ReLU(inplace=True))
            self.road_net = nn.ModuleList(
                Block(dim=embed_dim,
                      num_heads=num_heads,
                      attn_drop=0,
                      post_norm=post_norm) for i in range(roadnet_depth))
        elif self.road_embed_type == "pointnet":
            self.lane_embed = LaneEmbeddingLayer(7, embed_dim)
        elif self.road_embed_type == "lane":
            self.project_road = nn.Sequential(nn.Linear(43, embed_dim),
                                              nn.ReLU(inplace=True))
        elif self.road_embed_type == "full":
            self.project_road = nn.Sequential(nn.Linear(8, embed_dim),
                                              nn.GELU())
            self.road_norm = nn.LayerNorm(embed_dim)

        self.spa_net = nn.ModuleList(
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  attn_drop=0.2,
                  post_norm=post_norm) for i in range(3))
        self.norm = nn.LayerNorm(embed_dim)

        self.agent_traj_query = nn.Parameter(torch.randn(1, 6, embed_dim))
        self.cross_attender = nn.ModuleList([
            CrossAttenderBlock(embed_dim,
                               num_heads=8,
                               attn_drop=0.2,
                               kdim=embed_dim,
                               vdim=embed_dim,
                               post_norm=post_norm) for _ in range(3)
        ])

        self.decoder = SEPTMAEDecoder(embed_dim, 30, hidden_dim=512)

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("net."):]: v
            for k, v in ckpt.items() if k.startswith("net.")
        }
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def forward(self, data):
        Th = 30
        Rec_Th = 20
        hist_padding_mask = data["x_padding_mask"][:, :, :Rec_Th]
        # time serires sequence mask; true means masked
        hist_key_padding_mask = hist_padding_mask.all(-1)  # agent padding mask
        hist_feat = torch.cat(
            [
                data["x_trans"][..., :Rec_Th, :],  # [B, A, L, 2]
                data["x_velocity"][..., :Rec_Th].unsqueeze(-1),
                data['x_angles'][..., :Rec_Th].unsqueeze(-1),
                data['x_attr'][..., -1][..., None, None].repeat(
                    1, 1, Rec_Th, 1)
            ],
            dim=-1,
        )  # [B, A, L, 5]

        B, A, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * A, L, D)
        hist_feat_key_padding = hist_key_padding_mask.view(B * A)
        hist_padding_mask = hist_padding_mask.view(-1,
                                                   hist_padding_mask.size(-1))

        # 1. projection layer
        actor_feat = self.project_agent_hist(
            hist_feat[~hist_feat_key_padding])  # [B*N, L, D_m]

        # 2. tempo net
        if self.tempo_embed_type == "norm":
            positions = torch.arange(L).unsqueeze(0).repeat(
                actor_feat.size(0), 1).to(actor_feat.device)
            traj_pos_embed = self.traj_pos_enc(positions)
            actor_feat = actor_feat + traj_pos_embed
        position_bias = True
        for blk in self.tempo_net:
            if self.tempo_embed_type == "T5":
                actor_feat = blk(
                    actor_feat,
                    key_padding_mask=hist_padding_mask[~hist_feat_key_padding],
                    position_bias=position_bias)
            else:
                actor_feat = blk(
                    actor_feat,
                    key_padding_mask=hist_padding_mask[~hist_feat_key_padding],
                )
            position_bias = False

        # 3. max-pooling
        actor_feat_tmp = torch.zeros(B * A,
                                     actor_feat.shape[-1],
                                     device=actor_feat.device)
        actor_feat, _ = torch.max(actor_feat, dim=1)  # [B*N, D_m]
        actor_feat_tmp[~hist_feat_key_padding] = actor_feat
        actor_feat = actor_feat_tmp.reshape(B, A, -1)

        lane_padding_mask = data["lane_padding_mask"]  # [B, R, 20]
        lane_key_padding_mask = data['lane_key_padding_mask']  # [B, R]
        if self.road_embed_type == "transform":
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
                                          dim=-1).all(-1)
            B, M, L, D = lane_feat.shape
            lane_feat = lane_feat.view(-1, L, D)
            lane_padding_mask = lane_padding_mask.view(-1, L)
            lane_feat_key_padding = lane_key_padding_mask.view(-1)

            # 4. road projection layer
            lane_actor_feat = self.project_road(
                lane_feat[~lane_feat_key_padding])

            # 5. road feature extract
            for blk in self.road_net:
                lane_actor_feat = blk(
                    lane_actor_feat,
                    key_padding_mask=lane_padding_mask[~lane_feat_key_padding])

            lane_actor_feat_tmp = torch.zeros(B * M,
                                              lane_actor_feat.shape[-1],
                                              device=lane_actor_feat.device)
            lane_actor_feat = torch.mean(lane_actor_feat, dim=1)  # [B*N, D_m]
            lane_actor_feat_tmp[~lane_feat_key_padding] = lane_actor_feat
            lane_actor_feat = lane_actor_feat_tmp.reshape(B, M, -1)
        elif self.road_embed_type == "pointnet":
            lane_feat = torch.cat(
                [
                    data["lane_positions"],  # [B, R, L, 2]
                    data['lane_angles'][..., None, None].repeat(1, 1, 20, 1),
                    data['lane_attr'][..., None, :].repeat(1, 1, 20, 1),
                    ~lane_padding_mask[..., None]
                ],
                dim=-1,
            )  # [B, R, L, 5]
            B, R, L, D = lane_feat.shape
            lane_feat = self.lane_embed(lane_feat.view(-1, L, D).contiguous())
            lane_actor_feat = lane_feat.view(B, R, -1)
        elif self.road_embed_type == "lane":
            B, R, _ = lane_padding_mask.shape
            lane_feat = torch.cat(
                [
                    data["lane_positions"].view(-1, 20 * 2),  # [B*R, 40]
                    torch.cos(data['lane_angles'].view(-1, 1)),
                    torch.sin(data['lane_angles'].view(-1, 1)),
                    data['lane_attr'][..., -1].view(-1, 1),
                ],
                dim=-1,
            )  # [B*R, 43]
            lane_feat_key_padding = lane_key_padding_mask.view(-1)
            lane_actor_feat = self.project_road(lane_feat)
            lane_actor_feat = lane_actor_feat.view(B, R, -1)
        elif self.road_embed_type == "full":
            lane_padding_mask = data["lane_padding_mask_candidate"]  # [B, R]
            # lane_key_padding_mask = lane_padding_mask.all(-1)  # [B, R]
            B, R = lane_padding_mask.shape
            # distances = torch.sqrt(
            #     torch.sum((data["lane_positions_candidate"][..., :-1, :] -
            #                data["lane_positions_candidate"][..., 1:, :])**2,
            #               dim=-1))
            # lane_feat = torch.cat(
            #     [
            #         data["lane_positions_candidate"][
            #             ..., :-1, :],  # [B, R, L, 2]
            #         data["lane_positions_candidate"][...,
            #                                          1:, :],  # [B, R, L, 2]
            #         distances[..., None],
            #         data['lane_angles_candidate'][..., None, None].repeat(
            #             1, 1, L-1, 1),
            #         data['lane_attr_candidate'][..., None, :].repeat(
            #             1, 1, L-1, 1),
            #     ],
            #     dim=-1,
            # )  # [B, R, L, 9]
            lane_feat = data['lane_positions_candidate']
            # lane_padding_mask = torch.cat([
            #     lane_padding_mask[..., :-1][..., None],
            #     lane_padding_mask[..., 1:][..., None],
            # ],
            #   dim=-1).all(-1).reshape(B, -1)
            lane_actor_feat = lane_feat.reshape(B, R, 8)
            lane_actor_feat = self.project_road(lane_actor_feat)

        # 6. spa net
        x_encoder = torch.cat([actor_feat, lane_actor_feat], dim=1)
        if self.road_embed_type == "full":
            key_padding_mask = torch.cat(
                [data["x_key_padding_mask"],
                 lane_padding_mask.reshape(B, R)],
                dim=1)
        else:
            key_padding_mask = torch.cat(
                [data["x_key_padding_mask"], data["lane_key_padding_mask"]],
                dim=1)

        for blk in self.spa_net:
            x_encoder = blk(x_encoder, key_padding_mask=key_padding_mask)
        # pre-norm
        x_agent = x_encoder[:, :A, :]  # [B, A, D]

        y_hat = self.decoder(x_agent)

        return {
            "y_hat": y_hat,
        }
