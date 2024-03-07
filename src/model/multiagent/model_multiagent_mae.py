import torch
import torch.nn as nn

# from ..layers.agent_embedding import AgentEmbeddingLayer
from ..layers.lane_embedding import LaneEmbeddingLayer
from ..layers.transformer_blocks import Block, CrossAttenderBlock
from torch_scatter import scatter_mean
from ..layers.multimodal_decoder import MultiAgentDecoder, MultiAgentProposeDecoder


class MLPDecoder(nn.Module):

    def __init__(self, embed_dim, out_channels) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.ReLU(),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, out_channels),
        )

    def forward(self, x):
        return self.mlp(x)


class ModelMultiAgentMAE(nn.Module):

    def __init__(
        self,
        embed_dim=256,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        future_steps: int = 60,
        num_modes: int = 6,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.use_cls_token = use_cls_token

        # hist encoder
        self.project_agent_hist = nn.Sequential(nn.Linear(4, embed_dim),
                                                nn.ReLU(inplace=True))
        self.traj_pos_enc = nn.Embedding(51, embed_dim)
        self.tempo_net = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                attn_drop=0.2,
                post_norm=False,
            ) for _ in range(3))

        # lane encoder
        self.project_road = nn.Sequential(nn.Linear(5, embed_dim),
                                          nn.ReLU(inplace=True))
        self.road_net = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                attn_drop=0.2,
                post_norm=False,
            ) for _ in range(3))

        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.spa_net = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=0.2,
            ) for i in range(encoder_depth))
        self.norm = nn.LayerNorm(embed_dim)
        self.agent_traj_query = nn.Parameter(torch.randn(6, embed_dim))
        self.cross_attender = nn.ModuleList()
        for _ in range(3):
            self.cross_attender.append(
                CrossAttenderBlock(embed_dim,
                                   num_heads=8,
                                   attn_drop=0.2,
                                   kdim=embed_dim,
                                   vdim=embed_dim,
                                   post_norm=False))
            self.cross_attender.append(
                Block(dim=embed_dim,
                      num_heads=num_heads,
                      attn_drop=0.2,
                      post_norm=False))

        self.mode_2_mode_propose = nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  attn_drop=0.2,
                  post_norm=False) for _ in range(1)
        ])

        self.decoder_propose = MultiAgentProposeDecoder(embed_dim,
                                                        future_steps,
                                                        hidden_dim=512)
        self.traj_project = nn.Sequential(nn.Linear(2, embed_dim), nn.GELU())
        self.traj_emb = nn.GRU(input_size=embed_dim,
                               hidden_size=embed_dim,
                               num_layers=1,
                               bias=True,
                               batch_first=False,
                               dropout=0.0,
                               bidirectional=False)
        self.traj_emb_h0 = nn.Parameter(torch.zeros(1, embed_dim))

        self.cross_attender_refine = nn.ModuleList()
        for _ in range(3):
            self.cross_attender_refine.append(
                CrossAttenderBlock(embed_dim,
                                   num_heads=8,
                                   attn_drop=0.2,
                                   kdim=embed_dim,
                                   vdim=embed_dim,
                                   post_norm=False))
            self.cross_attender_refine.append(
                Block(dim=embed_dim,
                      num_heads=num_heads,
                      attn_drop=0.2,
                      post_norm=False))

        self.mode_2_mode_refine = nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  attn_drop=0.2,
                  post_norm=False) for _ in range(1)
        ])
        self.scene_query = nn.Parameter(torch.randn(6, embed_dim))
        self.scene_2_mode = nn.ModuleList([
            CrossAttenderBlock(embed_dim,
                               num_heads=8,
                               attn_drop=0.2,
                               kdim=embed_dim,
                               vdim=embed_dim,
                               post_norm=False) for _ in range(2)
        ])

        self.decoder = MultiAgentDecoder(embed_dim,
                                         future_steps,
                                         hidden_dim=256)

        # self.traj_decoder = MLPDecoder(embed_dim + self.num_modes,
        #                                self.future_steps * 2)
        self.prob_decoder = MLPDecoder(embed_dim, 1)

        self.register_buffer("one_hot_mask", torch.eye(self.num_modes))

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.tempo_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.initialize_weights()

    def initialize_weights(self):
        if self.use_cls_token:
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.cls_pos, std=0.02)

        self.apply(self._init_weights)

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
        radians = data['x_angles'][:, :, :50]
        cos = torch.cos(radians).unsqueeze(-1).unsqueeze(-1)
        sin = torch.sin(radians).unsqueeze(-1).unsqueeze(-1)
        rotation_matrix = torch.zeros(radians.shape[0], radians.shape[1], 50,
                                      2, 2).to(radians.device)
        rotation_matrix[..., 0, 0] = cos[..., 0].squeeze(-1)
        rotation_matrix[..., 0, 1] = sin[..., 0].squeeze(-1)
        rotation_matrix[..., 1, 0] = -sin[..., 0].squeeze(-1)
        rotation_matrix[..., 1, 1] = cos[..., 0].squeeze(-1)
        hist_feat = torch.cat(
            [
                torch.einsum('...ij,...j->...i', rotation_matrix, data['x']),
                # data["x"],
                data["x_velocity_diff"][..., None],
                data['x_attr'][..., -1][..., None, None].repeat(1, 1, 50, 1),
            ],
            dim=-1,
        )

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_padding = hist_key_padding_mask.view(
            B * N)  # agent wheather masked
        hist_padding_mask = hist_padding_mask.view(B * N, L)
        hist_padding_mask = torch.cat([
            torch.zeros(hist_padding_mask.shape[0], 1).to(
                hist_padding_mask.dtype).to(hist_padding_mask.device),
            hist_padding_mask
        ],
                                      dim=-1)

        # 1. Hist embdedding: projection
        actor_feat = self.project_agent_hist(hist_feat[~hist_feat_key_padding])
        tempo_token = self.tempo_token.repeat(actor_feat.shape[0], 1, 1)
        actor_feat = torch.cat([tempo_token, actor_feat], dim=1)

        # 2. temo net
        actor_positions = torch.arange(L + 1).unsqueeze(0).repeat(
            actor_feat.size(0), 1).to(actor_feat.device)
        traj_pos_embed = self.traj_pos_enc(actor_positions)
        actor_feat = actor_feat + traj_pos_embed
        for blk in self.tempo_net:
            actor_feat = blk(
                actor_feat,
                key_padding_mask=hist_padding_mask[~hist_feat_key_padding],
            )

        # max-pooling
        actor_feat_tmp = torch.zeros(B * N,
                                     actor_feat.shape[-1],
                                     device=actor_feat.device)
        # actor_feat, _ = torch.max(actor_feat, dim=1)
        actor_feat = actor_feat[:, 0]
        actor_feat_tmp[~hist_feat_key_padding] = actor_feat
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])

        # 3. lane embedding: projection
        lane_padding_mask = data["lane_padding_mask"]
        B, M, L = lane_padding_mask.shape
        lane_feat = torch.cat(
            [
                data["lane_positions"] - data["lane_centers"].unsqueeze(-2),
                # data['lane_angles'][..., None, None].repeat(1, 1, L, 1),
                data['lane_attr'][..., None, :].repeat(1, 1, L, 1),
            ],
            dim=-1,
        )  # [B, R, L, 4]

        B, M, L, D = lane_feat.shape
        lane_feat = lane_feat.view(-1, L, D)
        lane_padding_mask = lane_padding_mask.view(-1, L)
        lane_feat_key_padding = data['lane_key_padding_mask'].reshape(B * M)
        lane_actor_feat = self.project_road(lane_feat[~lane_feat_key_padding])

        # 4. road net
        for blk in self.road_net:
            lane_actor_feat = blk(
                lane_actor_feat,
                key_padding_mask=lane_padding_mask[~lane_feat_key_padding])
        lane_actor_feat_tmp = torch.zeros(B * M,
                                          lane_actor_feat.shape[-1],
                                          device=lane_actor_feat.device)
        # lane_actor_feat, _ = torch.max(lane_actor_feat, dim=1)  # [B*N, D_m]
        # lane_actor_feat = torch.mean(lane_actor_feat, dim=1)  # [B*N, D_m]
        lane_actor_feat = scatter_mean(
            lane_actor_feat,
            index=lane_padding_mask[~lane_feat_key_padding].long(),
            dim=1)[:, 0]
        lane_actor_feat_tmp[~lane_feat_key_padding] = lane_actor_feat
        lane_actor_feat = lane_actor_feat_tmp.reshape(B, M, -1)

        # lane_feat = data["lane_trans"].to(torch.float32)
        # B, M, D = lane_feat.shape
        # lane_feat_mask = data["lane_trans_padding_mask"]
        # lane_feat = lane_feat.reshape(-1, lane_feat.shape[-1])
        # lane_key_padding_mask = lane_feat_mask.all(-1)
        # data["lane_key_padding_mask"] = lane_key_padding_mask
        # lane_feat_key_padding = lane_key_padding_mask.view(-1)
        # lane_actor_feat = self.project_road(lane_feat[~lane_feat_key_padding])
        # lane_actor_feat_tmp = torch.zeros(B * M,
        #                                   lane_actor_feat.shape[-1],
        #                                   device=lane_actor_feat.device)
        # lane_actor_feat_tmp[~lane_feat_key_padding] = lane_actor_feat
        # lane_actor_feat = lane_actor_feat_tmp.reshape(B, M, -1)

        # pos embed MLP, used to get global information
        x_centers = torch.cat([data["x_centers"], data["lane_centers"].to(torch.float32)],
                              dim=1)  # [B, A+M, 2]
        # x_centers = data["x_centers"]
        angles = torch.cat([data["x_angles"][:, :, 49], data["lane_angles"]],
                           dim=1)  # [B, A+M]
        # angles = data["x_angles"][:, :, 49]
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)
        # lane_pos_embed = torch.zeros_like(lane_actor_feat)
        # pos_embed = torch.cat([pos_embed, lane_pos_embed], dim=1)

        x_encoder = torch.cat([actor_feat, lane_actor_feat], dim=1)
        key_padding_mask = torch.cat(
            [data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1)

        if self.use_cls_token:
            x_encoder = torch.cat(
                [x_encoder, self.cls_token.repeat(B, 1, 1)], dim=1)
            key_padding_mask = torch.cat(
                [
                    key_padding_mask,
                    torch.zeros(B, 1, device=key_padding_mask.device)
                ],
                dim=1,
            )
            pos_embed = torch.cat(
                [pos_embed, self.cls_pos.repeat(B, 1, 1)], dim=1)
        # else:
        #     cls_token = scatter_mean(x_encoder, key_padding_mask.long(),
        #                              dim=1)[:, 0]

        # feature and position embeding. inject pos in global
        x_encoder = x_encoder + pos_embed

        # 5. spa net
        for _, blk in enumerate(self.spa_net):
            x_encoder = blk(x_encoder, key_padding_mask=key_padding_mask)
        x_encoder = self.norm(x_encoder)

        if self.use_cls_token:
            cls_token = x_encoder[:, -1]
            x_encoder = x_encoder[:, :-1]
            key_padding_mask = key_padding_mask[:, :-1]

        # 6. mode query
        traj_query = (self.agent_traj_query.unsqueeze(0).unsqueeze(0).repeat(
            B, N, 1, 1) + pos_embed[:, :N].unsqueeze(2) +
                      actor_feat.unsqueeze(2)).reshape(B, N * self.num_modes,
                                                       pos_embed.shape[-1])

        # mode2scene
        # for blk in self.cross_attender:
        for i in range(0, len(self.cross_attender), 2):
            traj_query = self.cross_attender[i](
                traj_query,
                x_encoder,
                x_encoder,
                key_padding_mask=key_padding_mask)
            traj_query = traj_query.reshape(B, N, self.num_modes, -1).permute(
                0, 2, 1, 3).reshape(B * self.num_modes, N, -1)
            traj_query = self.cross_attender[i + 1](traj_query)
            traj_query = traj_query.reshape(B, self.num_modes, N, -1).permute(
                0, 2, 1, 3).reshape(B, N, self.num_modes,
                                    -1).reshape(B, N * self.num_modes, -1)
        # mode2mode
        traj_query = traj_query.reshape(B, N, self.num_modes,
                                        -1).reshape(B * N, self.num_modes, -1)
        for blk in self.mode_2_mode_propose:
            traj_query = blk(traj_query)

        traj_query = traj_query.reshape(B, N, self.num_modes, -1)

        # propose trajectory
        traj_propose = self.decoder_propose(
            traj_query)  # [B, N, num_mode, T, 2]
        traj_query = self.traj_project(traj_propose.detach())
        B, N, _, T, D = traj_query.shape
        traj_query = traj_query.reshape(B * N * self.num_modes, T,
                                        D).transpose(0, 1)
        traj_query = self.traj_emb(
            traj_query,
            self.traj_emb_h0.unsqueeze(1).repeat(1, traj_query.size(1),
                                                 1))[1].squeeze(0)
        traj_query = traj_query.reshape(B, N * self.num_modes, D)
        # mode2scene
        # for blk in self.cross_attender_refine:
        for i in range(0, len(self.cross_attender_refine), 2):
            traj_query = self.cross_attender_refine[i](
                traj_query,
                x_encoder,
                x_encoder,
                key_padding_mask=key_padding_mask)
            traj_query = traj_query.reshape(B, N, self.num_modes, -1).permute(
                0, 2, 1, 3).reshape(B * self.num_modes, N, -1)
            traj_query = self.cross_attender_refine[i + 1](traj_query)
            traj_query = traj_query.reshape(B, self.num_modes, N, -1).permute(
                0, 2, 1, 3).reshape(B, N, self.num_modes,
                                    -1).reshape(B, N * self.num_modes, -1)
        # mode2mode
        traj_query = traj_query.reshape(B, N, self.num_modes,
                                        -1).reshape(B * N, self.num_modes, -1)
        for blk in self.mode_2_mode_refine:
            traj_query = blk(traj_query)

        traj_query = traj_query.reshape(B, N, self.num_modes, -1)
        y_hat, _ = self.decoder(traj_query)
        y_hat += traj_propose.detach()

        # cls_token = cls_token.unsqueeze(1).repeat(1, self.num_modes, 1)
        # one_hot_mask = self.one_hot_mask
        # cls_token = torch.cat([
        #     cls_token,
        #     one_hot_mask.view(1, self.num_modes, self.num_modes).repeat(
        #         B, 1, 1)
        # ],
        #                       dim=-1)

        # Scene scoring module using cross attention
        traj_query = traj_query.reshape(B, N * self.num_modes, -1)
        scene_query = self.scene_query.unsqueeze(0).repeat(B, 1, 1)
        for blk in self.scene_2_mode:
            scene_query = blk(scene_query, traj_query, traj_query)

        pi = self.prob_decoder(scene_query)
        # if self.use_cls_token:
        #     cls_token = x_encoder[:, -1]
        # else:
        #     # global avg pooling
        #     cls_token = scatter_mean(x_encoder, key_padding_mask.long(),
        #                              dim=1)[:, 0]

        # K = self.num_modes
        # x_actors = x_encoder[:, :N].unsqueeze(1).repeat(1, K, 1, 1) # [B, m, A, D]
        # cls_token = cls_token.unsqueeze(1).repeat(1, K, 1) # [B, m, D]
        # one_hot_mask = self.one_hot_mask # identity [m, m]

        # x_actors = torch.cat(
        #     [x_actors,
        #      one_hot_mask.view(1, K, 1, K).repeat(B, 1, N, 1)],
        #     dim=-1) # 添加mode的区分，避免module collaspe
        # cls_token = torch.cat(
        #     [cls_token, one_hot_mask.view(1, K, K).repeat(B, 1, 1)], dim=-1)

        # y_hat = self.traj_decoder(x_actors).view(B, K, N, self.future_steps, 2)
        # pi = self.prob_decoder(cls_token).view(B, K)

        return {"y_hat": y_hat, "pi": pi, "y_propose": traj_propose}
