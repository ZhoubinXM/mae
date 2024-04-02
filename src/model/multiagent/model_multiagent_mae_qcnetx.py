import torch
import torch.nn as nn

from ..layers.transformer_blocks import Block, CrossAttenderBlock
from ..layers.multimodal_decoder import MultiAgentDecoder, MultiAgentProposeDecoder, MultimodalPiDecoder
from ..layers.fourier_embedding import FourierEmbedding
from x_transformers import CrossAttender, Encoder
from typing import List, Union, Optional
from src.utils.weight_init import weight_init
import torch.nn.functional as Func
import math


class MLPDecoder(nn.Module):

    def __init__(self, embed_dim, out_channels) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, out_channels),
        )

    def forward(self, x):
        return self.mlp(x)


class MLPLayer(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int,
                 output_dim: int) -> None:
        super(MLPLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ModelMultiAgentMAE(nn.Module):

    def __init__(
        self,
        embed_dim=256,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.1,
        future_steps: int = 60,
        num_modes: int = 6,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.embed_dim = embed_dim
        self.num_future_steps = future_steps
        self.use_cls_token = use_cls_token

        a_input_dim = 4
        lane_input_dim = 3
        num_freq_bands = 64

        self.encoder_nums = 2
        self.num_recurrent_steps = 3

        # agent
        self.type_a_emb = nn.Embedding(5, embed_dim)
        self.traj_pos_enc = nn.Embedding(50, embed_dim)
        self.project_agent_hist = FourierEmbedding(
            input_dim=a_input_dim,
            hidden_dim=embed_dim,
            num_freq_bands=num_freq_bands)
        
        self.agent_tempo_net = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                attn_drop=drop_path,
                post_norm=False,
                drop=drop_path,
                # attn_type="Ro",
                # max_t_len=51,
            ) for _ in range(2))
        # self.agent_tempo_query = nn.Parameter(torch.randn(1, embed_dim))

        # lane
        self.project_lane = FourierEmbedding(input_dim=lane_input_dim,
                                             hidden_dim=embed_dim,
                                             num_freq_bands=num_freq_bands)
        self.project_lane_cate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim))
        self.lane_type_emb = nn.Embedding(3, embed_dim)
        self.lane_intersect_emb = nn.Embedding(2, embed_dim)
        self.lane_vector_net = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                attn_drop=drop_path,
                post_norm=False,
                drop=drop_path,
            ) for _ in range(2))
        self.lane_vector_query = nn.Parameter(torch.randn(1, embed_dim))

        # scene
        self.lane_pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # self.lane_pos_embed = FourierEmbedding(input_dim=3,
        #                                        hidden_dim=embed_dim,
        #                                        num_freq_bands=num_freq_bands)

        self.agent_pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # self.agent_pos_embed = FourierEmbedding(input_dim=3,
        #                                         hidden_dim=embed_dim,
        #                                         num_freq_bands=num_freq_bands)
        self.lane_spa_net = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_drop=drop_path,
                post_norm=False,
                drop=drop_path
            ) for _ in range(2))
        
        self.agent_encoder = nn.ModuleList()
        for _ in range(2):
            self.agent_encoder.append(Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_drop=drop_path,
                post_norm=False,
                drop=drop_path
            ))
            self.agent_encoder.append(Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_drop=drop_path,
                post_norm=False,
                drop=drop_path
            ))
            self.agent_encoder.append(Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_drop=drop_path,
                post_norm=False,
                drop=drop_path
            ))

        # modal_query
        self.agent_traj_query = nn.Parameter(torch.randn(6, embed_dim))
        self.cross_attender = nn.ModuleList()
        for _ in range(2):
            self.cross_attender.append(
                CrossAttenderBlock(embed_dim,
                                   num_heads=8,
                                   attn_drop=drop_path,
                                   kdim=embed_dim,
                                   vdim=embed_dim,
                                   post_norm=False,
                                   drop=drop_path,))
            self.cross_attender.append(
                CrossAttenderBlock(embed_dim,
                                   num_heads=8,
                                   attn_drop=drop_path,
                                   kdim=embed_dim,
                                   vdim=embed_dim,
                                   post_norm=False,
                                   drop=drop_path,))
            self.cross_attender.append(
                Block(dim=embed_dim,
                      num_heads=num_heads,
                      attn_drop=drop_path,
                      post_norm=False,
                      drop=drop_path))

        self.mode2mode_propose = Block(dim=embed_dim,
                                       num_heads=num_heads,
                                       attn_drop=drop_path,
                                       post_norm=False,
                                       drop=drop_path)

        self.scene_query = nn.Parameter(torch.randn(1, embed_dim))
        self.scene_2_mode = nn.ModuleList([
            CrossAttenderBlock(embed_dim,
                               num_heads=8,
                               attn_drop=drop_path,
                               kdim=embed_dim,
                               vdim=embed_dim,
                               post_norm=False,
                               drop=drop_path) for _ in range(2)
        ])

        self.traj_emb = nn.GRU(input_size=embed_dim,
                               hidden_size=embed_dim,
                               num_layers=1,
                               bias=True,
                               batch_first=False,
                               dropout=0.0,
                               bidirectional=False)
        self.traj_emb_h0 = nn.Parameter(torch.zeros(1, embed_dim))

        self.cross_attender_refine = nn.ModuleList()
        for _ in range(2):
            self.cross_attender_refine.append(
                CrossAttenderBlock(embed_dim,
                                   num_heads=8,
                                   attn_drop=drop_path,
                                   kdim=embed_dim,
                                   vdim=embed_dim,
                                   post_norm=False,
                                   drop=drop_path))
            self.cross_attender_refine.append(
                CrossAttenderBlock(embed_dim,
                                   num_heads=8,
                                   attn_drop=drop_path,
                                   kdim=embed_dim,
                                   vdim=embed_dim,
                                   post_norm=False,
                                   drop=drop_path))
            self.cross_attender_refine.append(
                Block(dim=embed_dim,
                      num_heads=num_heads,
                      attn_drop=drop_path,
                      post_norm=False,
                      drop=drop_path))

        self.mode2mode_refine = Block(dim=embed_dim,
                                      num_heads=num_heads,
                                      attn_drop=drop_path,
                                      post_norm=False,
                                      drop=drop_path)

        self.prob_decoder = MultimodalPiDecoder(
            embed_dim, future_steps=self.num_future_steps)

        self.to_loc_propose_pos = MLPLayer(input_dim=embed_dim,
                                           hidden_dim=embed_dim,
                                           output_dim=future_steps * 2 //
                                           self.num_recurrent_steps)
        self.to_loc_refine_pos = MLPLayer(input_dim=embed_dim,
                                          hidden_dim=embed_dim,
                                          output_dim=future_steps * 2)

        self.y_emb = FourierEmbedding(input_dim=2,
                                      hidden_dim=embed_dim,
                                      num_freq_bands=num_freq_bands)

        self.initialize_weights()

    def initialize_weights(self):
        if self.use_cls_token:
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.cls_pos, std=0.02)

        self.apply(weight_init)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, data):
        hist_padding_mask = data["x_padding_mask"][:, :, :50]
        hist_key_padding_mask = data["x_key_padding_mask"]
        head_a = data['x_angles'][:, :, :50].contiguous()
        # head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        motion_vector_a = data['x']
        vel_normal = data['x_velocity_diff']
        # vel = torch.stack([vel_normal.cos(), vel_normal.sin()], dim=-1)

        categorical_embeds = self.type_a_emb(
            data['x_attr'][..., -1].long()).unsqueeze(2).repeat(1, 1, 50, 1)

        x_a = torch.cat([
            motion_vector_a,
            head_a.unsqueeze(-1),
            vel_normal.unsqueeze(-1),
        ],
                          dim=-1)

        B, N, T, F = x_a.shape
        hist_feat = x_a.reshape(B * N, T, F)
        hist_feat_key_padding = hist_key_padding_mask.view(
            B * N)  # agent wheather masked
        hist_padding_mask = hist_padding_mask.reshape(B * N, T)

        # 1. Hist embdedding: projection
        actor_feat = self.project_agent_hist(
            continuous_inputs=hist_feat[~hist_feat_key_padding],
            categorical_embs=[
                categorical_embeds.reshape(B * N, T,
                                           -1)[~hist_feat_key_padding]
            ])
        actor_positions = torch.arange(T).unsqueeze(0).repeat(
            actor_feat.size(0), 1).to(actor_feat.device)
        traj_pos_embed = self.traj_pos_enc(actor_positions)
        actor_feat = actor_feat + traj_pos_embed

        # use self attention pooling
        # agent_tempo_query = self.agent_tempo_query[None, :, :].repeat(
        #     actor_feat.shape[0], 1, 1)
        # actor_feat = torch.cat([agent_tempo_query, actor_feat], dim=1)
        # hist_padding_mask = torch.cat([
        #     torch.zeros([B * N, 1]).to(hist_padding_mask.dtype).to(
        #         hist_padding_mask.device), hist_padding_mask
        # ],
        #                               dim=1)

        # 2. temo net for agent time self attention
        for agent_tempo_blk in self.agent_tempo_net:
            actor_feat = agent_tempo_blk(
                actor_feat,
                key_padding_mask=hist_padding_mask[~hist_feat_key_padding],
            )
        # 3. pooling
        actor_feat_tmp = torch.zeros(B * N, T,
                                     actor_feat.shape[-1],
                                     device=actor_feat.device)
        # agent_tempo_query = actor_feat[:, 0]
        actor_feat_tmp[~hist_feat_key_padding] = actor_feat
        actor_feat = actor_feat_tmp.view(B, N, T, actor_feat.shape[-1])

        # 3. lane embedding: projection
        lane_padding_mask = data["lane_padding_mask"]  # [B, M, L]
        B, M, L = lane_padding_mask.shape
        # lane_vector = data["lane_positions"] - data[
        #     "lane_positions"][:, :, 0].unsqueeze(2)
        lane_vector = data["lane_positions"][:,:,1:] - data["lane_positions"][:,:,:-1]
        lane_vector = torch.cat([torch.zeros(B,M,1,2).to(actor_feat.device), lane_vector], dim=2)
        lane_angles = torch.arctan2(lane_vector[..., 1], lane_vector[..., 0])

        # lane_feat = torch.stack(
        #     [
        #         # lane_vector,
        #         torch.norm(lane_vector[..., :2], p=2, dim=-1),
        #         # lane_angles.unsqueeze(-1),
        #         lane_angles
        #     ],
        #     dim=-1,
        # )  # [B, M, L, 2]
        lane_feat = torch.cat(
            [
                lane_vector,
                lane_angles.unsqueeze(-1),
            ],
            dim=-1,
        )  # [B, M, L, 2]

        B, M, L, LF = lane_feat.shape
        lane_feat = lane_feat.view(-1, L, LF)
        lane_padding_mask = lane_padding_mask.view(-1, L)
        lane_feat_key_padding = data['lane_key_padding_mask'].reshape(B * M)

        lane_categorical_embeds = torch.cat([
            self.lane_type_emb(data["lane_attr"][..., 0].long()),
            self.lane_intersect_emb(data["lane_attr"][..., -1].long())
        ],
                                            dim=-1).unsqueeze(2).repeat(
                                                1, 1, L,
                                                1).reshape(B * M, L, -1)
        lane_categorical_embeds = self.project_lane_cate(
            lane_categorical_embeds)
        # lane_actor_feat = lane_actor_feat + lane_categorical_embeds[
        #     ~lane_feat_key_padding]
        lane_actor_feat = self.project_lane(
            lane_feat[~lane_feat_key_padding],
            categorical_embs=[lane_categorical_embeds[~lane_feat_key_padding]])
        # lane_actor_feat = self.post_project_lane(lane_actor_feat)
        lane_query = self.lane_vector_query[None, :, :].repeat(
            lane_actor_feat.shape[0], 1, 1)
        lane_actor_feat = torch.cat([lane_query, lane_actor_feat], dim=1)
        lane_padding_mask = torch.cat([
            torch.zeros([B * M, 1]).to(lane_padding_mask.dtype).to(
                lane_padding_mask.device), lane_padding_mask
        ],
                                      dim=1)

        for lane_blk in self.lane_vector_net:
            lane_actor_feat = lane_blk(
                lane_actor_feat,
                key_padding_mask=lane_padding_mask[~lane_feat_key_padding])

        lane_actor_feat_tmp = torch.zeros(B * M,
                                          lane_actor_feat.shape[-1],
                                          device=lane_actor_feat.device)

        lane_actor_feat_tmp[~lane_feat_key_padding] = lane_actor_feat[:, 0]
        lane_actor_feat = lane_actor_feat_tmp.reshape(B, M, -1)

        # Now lane and actor feature is local
        # TODO: Trans lane and actor feature to each timestamp use av_index
        x_positions = data["x_positions"][:, :, :50]  # [B, N, 2] 
        x_angles = data["x_angles"][:, :, :50]  # [B, N]
        x_angles = torch.stack([torch.cos(x_angles), torch.sin(x_angles)], dim=-1)
        # x_angles = x_angles.unsqueeze(-1)
        x_pos_feat = torch.cat([x_positions, x_angles], dim=-1)  # [B, N, 4]
        x_pos_embed = self.agent_pos_embed(x_pos_feat)
        actor_feat = actor_feat + x_pos_embed
        actor_feat = actor_feat.reshape(B, N, -1)

        lane_centers = data["lane_positions"][:, :, 0].to(torch.float32)
        lane_angles = torch.atan2(
            data["lane_positions"][..., 1, 1] -
            data["lane_positions"][..., 0, 1],
            data["lane_positions"][..., 1, 0] -
            data["lane_positions"][..., 0, 0],
        )
        lane_angles = torch.stack(
            [torch.cos(lane_angles),
             torch.sin(lane_angles)], dim=-1)
        # lane_angles = lane_angles.unsqueeze(-1)
        lane_pos_feat = torch.cat([lane_centers, lane_angles], dim=-1)
        lane_pos_embed = self.lane_pos_embed(lane_pos_feat)
        lane_actor_feat = lane_actor_feat + lane_pos_embed

        # Now lane&agent feature is global feature.

        # Lane Self Attention
        for blk in self.lane_spa_net:
            lane_actor_feat = blk(lane_actor_feat, key_padding_mask=data["lane_key_padding_mask"])

        # 3. Scene
        # 3.1 Agent-Time attention QKV
        # 3.2 Agent-Map attention Q: Agent KV
        #     Agent dimension: [B,N,T,D] global in curr timestamp.
        #     Map dimension: [B,M,D] global.
        # 3.3 Social-Attention for each timestamp
        
        for i in range(0, len(self.agent_encoder), 3):
            actor_feat = actor_feat.reshape(B*N, T, -1)
            actor_feat = actor_feat[~hist_feat_key_padding]
            actor_feat = self.agent_encoder[i](actor_feat,
                key_padding_mask=hist_padding_mask[~hist_feat_key_padding],)
            actor_feat_tmp = torch.zeros(B * N, T,
                                     actor_feat.shape[-1],
                                     device=actor_feat.device)
            actor_feat_tmp[~hist_feat_key_padding] = actor_feat
            actor_feat = actor_feat_tmp.view(B, N, T, actor_feat.shape[-1])
            
            actor_time_feat = actor_feat.permute(0, 2, 1, 3).reshape(B*T, N, -1)
            lane_actor_time_feat = lane_actor_feat.unsqueeze(1).repeat(1,T,1,1).reshape(B*T, M, -1)
            feat = torch.cat([actor_time_feat, lane_actor_time_feat], dim=1)
            lane_time_mask = data["lane_key_padding_mask"].unsqueeze(1).repeat(1,T,1).reshape(B*T, M)
            time_mask = data["x_padding_mask"][:, :, :50].permute(0,2,1).reshape(B*T, -1)
            feat_mask = torch.cat([time_mask, lane_time_mask], dim=1)
            feat = self.agent_encoder[i+1](feat,
                                                 key_padding_mask=feat_mask)
            actor_feat = feat[:, :N]
            
            actor_feat = self.agent_encoder[i+2](actor_feat, key_padding_mask=time_mask)

            actor_feat = actor_feat.reshape(B,T,N,-1).permute(0,2,1,3).reshape(B*N, T, -1)

        
        # 4. Decoder
        # 4.1 Propose Module
        #     Mode2Time
        #     Mode2Map
        #     SelfModeKAttention
        #   Mode2Mode

        # 6. mode query
        traj_query = (self.agent_traj_query.unsqueeze(0).unsqueeze(0).repeat(
            B, N, 1, 1)).reshape(
                B*N, self.num_modes, self.embed_dim)

        locs_propose_pos: List[Optional[
            torch.Tensor]] = [None] * self.num_recurrent_steps
        # mode2scene
        for t in range(self.num_recurrent_steps):
            for i in range(0, len(self.cross_attender), 3):
                # Mode2Time
                traj_query = self.cross_attender[i](
                    traj_query[~hist_feat_key_padding],
                    actor_feat[~hist_feat_key_padding],
                    actor_feat[~hist_feat_key_padding],
                    key_padding_mask=hist_padding_mask[~hist_feat_key_padding])
                traj_query_tmp = torch.zeros(B * N, self.num_modes,
                                     actor_feat.shape[-1],
                                     device=actor_feat.device)
                traj_query_tmp[~hist_feat_key_padding] = traj_query
                traj_query = traj_query_tmp.reshape(B*N, self.num_modes, actor_feat.shape[-1])
                traj_query = traj_query.reshape(
                    B, N, self.num_modes,
                    self.embed_dim).reshape(B, N*self.num_modes,
                                                       self.embed_dim)
                traj_query = self.cross_attender[i+1](traj_query, lane_actor_feat, lane_actor_feat, key_padding_mask=data["lane_key_padding_mask"])
                traj_query = traj_query.reshape(B,N, self.num_modes, self.embed_dim).permute(0,2,1,3).reshape(B*self.num_modes, N, self.embed_dim)
                mask = data["x_key_padding_mask"].unsqueeze(1).repeat(
                    1, self.num_modes, 1, 1).reshape(B * self.num_modes, N)

                traj_query = self.cross_attender[i + 2](traj_query,
                                                        key_padding_mask=mask)
                traj_query = traj_query.reshape(
                    B, self.num_modes, N,
                    self.embed_dim).permute(0, 2, 1,
                                            3).reshape(B * N, self.num_modes,
                                                       self.embed_dim)
            # mode2mode
            traj_query = traj_query.reshape(B * N, self.num_modes,
                                            self.embed_dim)
            traj_query = traj_query[~hist_feat_key_padding]
            traj_query = self.mode2mode_propose(traj_query)
            traj_query_tmp = torch.zeros(B * N,
                                         self.num_modes,
                                         self.embed_dim,
                                         device=traj_query.device)
            traj_query_tmp[~hist_feat_key_padding] = traj_query
            traj_query = traj_query_tmp.reshape(B, N * self.num_modes,
                                                actor_feat.shape[-1])

            # propose and refine predict trajectory
            traj_query = traj_query.reshape(B, N, self.num_modes, -1)
            locs_propose_pos[t] = self.to_loc_propose_pos(traj_query)
            traj_query = traj_query.reshape(B*N, self.num_modes,
                                            actor_feat.shape[-1])
        loc_propose_pos = torch.cumsum(torch.cat(locs_propose_pos,
                                                 dim=-1).reshape(
                                                     -1, self.num_modes,
                                                     self.num_future_steps, 2),
                                       dim=-2)
        traj_query = self.y_emb(
            torch.cat([loc_propose_pos.detach()],
                      dim=-1).reshape(B * (N) * self.num_modes,
                                      self.num_future_steps,
                                      2))

        traj_query = traj_query.reshape(B, N, self.num_modes,
                                        self.num_future_steps, self.embed_dim)
        B, _, _, T, D = traj_query.shape
        traj_query = traj_query.reshape(B * (N) * self.num_modes, T,
                                        D).transpose(0, 1)
        traj_query = self.traj_emb(
            traj_query,
            self.traj_emb_h0.unsqueeze(1).repeat(1, traj_query.size(1),
                                                 1))[1].squeeze(0)
        traj_query = traj_query.reshape(B*(N),  self.num_modes,
                                        self.embed_dim)
        for i in range(0, len(self.cross_attender_refine), 3):
            traj_query = self.cross_attender[i](
                    traj_query[~hist_feat_key_padding],
                    actor_feat[~hist_feat_key_padding],
                    actor_feat[~hist_feat_key_padding],
                    key_padding_mask=hist_padding_mask[~hist_feat_key_padding])
            traj_query_tmp = torch.zeros(B * N, self.num_modes,
                                    actor_feat.shape[-1],
                                    device=actor_feat.device)
            traj_query_tmp[~hist_feat_key_padding] = traj_query
            traj_query = traj_query_tmp.reshape(B*N, self.num_modes, actor_feat.shape[-1])
            traj_query = traj_query.reshape(
                B, N, self.num_modes,
                self.embed_dim).reshape(B, N*self.num_modes,
                                                   self.embed_dim)
            traj_query = self.cross_attender[i+1](traj_query, lane_actor_feat, lane_actor_feat, key_padding_mask=data["lane_key_padding_mask"])
            traj_query=traj_query.reshape(B,N, self.num_modes, self.embed_dim).permute(0,2,1,3).reshape(B*self.num_modes, N, self.embed_dim)

            mask = data["x_key_padding_mask"].unsqueeze(1).repeat(
                1, self.num_modes, 1, 1).reshape(B * self.num_modes, N)

            traj_query = self.cross_attender_refine[i + 2](
                traj_query, key_padding_mask=mask)
            traj_query = traj_query.reshape(
                B, self.num_modes, N,
                self.embed_dim).permute(0, 2, 1,
                                        3).reshape(B * N, self.num_modes,
                                                   self.embed_dim)

        traj_query = traj_query.reshape(B * N, self.num_modes, self.embed_dim)
        traj_query = traj_query[~hist_feat_key_padding]
        traj_query = self.mode2mode_refine(traj_query)
        traj_query_tmp = torch.zeros(B * N,
                                     self.num_modes,
                                     self.embed_dim,
                                     device=traj_query.device)
        traj_query_tmp[~hist_feat_key_padding] = traj_query
        traj_query = traj_query_tmp.view(B, N * self.num_modes,
                                         actor_feat.shape[-1])

        # decoder trajectory
        traj_query = traj_query.reshape(B, N, self.num_modes, -1)
        loc_refine_pos = self.to_loc_refine_pos(traj_query[:, :(N)]).view(
            -1, self.num_modes, self.num_future_steps, 2)
        loc_refine_pos = loc_refine_pos + loc_propose_pos.detach()

        y_hat = loc_refine_pos.reshape(B, N, 6, 60, 2)
        traj_propose = loc_propose_pos.reshape(B, N, 6, 60, 2)

        # Scene scoring module using cross attention
        traj_query = traj_query.reshape(B, N, self.num_modes, -1).permute(
            0, 2, 1, 3).reshape(B * self.num_modes, N, -1)
        traj_query = traj_query.reshape(B, N * self.num_modes, -1)
        scene_query = self.scene_query.unsqueeze(0).repeat(B, 1, 1)
        for blk in self.scene_2_mode:
            scene_query = blk(scene_query, traj_query, traj_query)
        scene_query = scene_query.reshape(B, -1)

        pi = self.prob_decoder(scene_query).unsqueeze(-1)

        return {
            "y_hat": y_hat,
            "pi": pi,
            "y_propose": traj_propose,
        }
