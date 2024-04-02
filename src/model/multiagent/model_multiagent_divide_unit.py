import torch
import torch.nn as nn

from ..layers.transformer_blocks import Block, CrossAttenderBlock
from ..layers.multimodal_decoder import MultiAgentDecoder, MultiAgentProposeDecoder


def angle_between_2d_vectors(ctr_vector: torch.Tensor,
                             nbr_vector: torch.Tensor) -> torch.Tensor:
    return torch.atan2(
        ctr_vector[..., 0] * nbr_vector[..., 1] -
        ctr_vector[..., 1] * nbr_vector[..., 0],
        (ctr_vector[..., :2] * nbr_vector[..., :2]).sum(dim=-1))


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


class ModelMultiAgentDU(nn.Module):

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
        self.future_steps = future_steps
        self.use_cls_token = use_cls_token

        a_input_dim = 4
        lane_input_dim = 2
        self.encoder_nums = 2

        # agent
        self.type_a_emb = nn.Embedding(5, embed_dim)
        self.traj_pos_enc = nn.Embedding(50, embed_dim)
        self.project_agent_hist = nn.Sequential(
            nn.Linear(a_input_dim, embed_dim), nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True), nn.Linear(embed_dim, embed_dim))
        self.post_project_agent = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim))

        self.agent_tempo_net = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                attn_drop=drop_path,
                post_norm=False,
            ) for _ in range(3))
        self.agent_tempo_query = nn.Parameter(torch.randn(1, embed_dim))
        self.agent_tempo_attend_pooling = nn.ModuleList(
            CrossAttenderBlock(
                dim=embed_dim,
                num_heads=8,
                attn_drop=drop_path,
                post_norm=False,
            ) for _ in range(1))

        # lane
        self.project_lane = nn.Sequential(nn.Linear(lane_input_dim, embed_dim),
                                          nn.LayerNorm(embed_dim),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(embed_dim, embed_dim))
        self.project_lane_cate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim))
        self.post_project_lane = nn.Sequential(nn.LayerNorm(embed_dim),
                                               nn.ReLU(inplace=True),
                                               nn.Linear(embed_dim, embed_dim))
        self.lane_type_emb = nn.Embedding(3, embed_dim)
        self.lane_intersect_emb = nn.Embedding(2, embed_dim)
        self.lane_vector_net = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                attn_drop=drop_path,
                post_norm=False,
            ) for _ in range(3))
        self.lane_vector_query = nn.Parameter(torch.randn(1, embed_dim))
        self.lane_vector_attend_pooling = nn.ModuleList(
            CrossAttenderBlock(
                dim=embed_dim,
                num_heads=8,
                attn_drop=drop_path,
                post_norm=False,
            ) for _ in range(1))

        # scene
        self.lane_pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.agent_pos_embed = nn.Sequential(
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
                attn_drop=drop_path,
            ) for _ in range(4))

        self.scene_norm = nn.LayerNorm(embed_dim)

        # modal_query
        self.agent_traj_query = nn.Parameter(torch.randn(6, embed_dim))
        self.cross_attender = nn.ModuleList()
        for _ in range(3):
            self.cross_attender.append(
                CrossAttenderBlock(embed_dim,
                                   num_heads=8,
                                   attn_drop=drop_path,
                                   kdim=embed_dim,
                                   vdim=embed_dim,
                                   post_norm=False))
            self.cross_attender.append(
                Block(dim=embed_dim,
                      num_heads=num_heads,
                      attn_drop=drop_path,
                      post_norm=False))
            self.cross_attender.append(
                Block(dim=embed_dim,
                      num_heads=num_heads,
                      attn_drop=drop_path,
                      post_norm=False))

        self.scene_query = nn.Parameter(torch.randn(6, embed_dim))
        self.scene_2_mode = nn.ModuleList([
            CrossAttenderBlock(embed_dim,
                               num_heads=8,
                               attn_drop=drop_path,
                               kdim=embed_dim,
                               vdim=embed_dim,
                               post_norm=False) for _ in range(2)
        ])

        # propose and refine predict result.
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
                                   attn_drop=drop_path,
                                   kdim=embed_dim,
                                   vdim=embed_dim,
                                   post_norm=False))
            self.cross_attender_refine.append(
                Block(dim=embed_dim,
                      num_heads=num_heads,
                      attn_drop=drop_path,
                      post_norm=False))
            self.cross_attender_refine.append(
                Block(dim=embed_dim,
                      num_heads=num_heads,
                      attn_drop=drop_path,
                      post_norm=False))

        self.decoder = MultiAgentDecoder(embed_dim,
                                         future_steps,
                                         hidden_dim=embed_dim * 2)
        self.prob_decoder = MLPDecoder(embed_dim, 1)

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
        head_a = data['x_angles'][:, :, :50].contiguous()
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        motion_vector_a = data['x']
        vel_normal = data['x_velocity_diff']
        vel = torch.stack([vel_normal.cos(), vel_normal.sin()], dim=-1)
        pos_a = data['x_positions']

        categorical_embeds = self.type_a_emb(
            data['x_attr'][..., -1].long()).unsqueeze(2).repeat(1, 1, 50, 1)

        x_a = torch.stack([
            torch.norm(motion_vector_a[..., :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=head_vector_a,
                                     nbr_vector=motion_vector_a[..., :2]),
            torch.norm(vel[..., :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=head_vector_a,
                                     nbr_vector=vel[..., :2])
        ],
                          dim=-1)

        B, N, T, F = x_a.shape
        hist_feat = x_a.reshape(B * N, T, F)
        hist_feat_key_padding = hist_key_padding_mask.view(
            B * N)  # agent wheather masked
        hist_padding_mask = hist_padding_mask.reshape(B * N, T)

        # 1. Hist embdedding: projection
        actor_feat = self.project_agent_hist(hist_feat[~hist_feat_key_padding])
        actor_positions = torch.arange(T).unsqueeze(0).repeat(
            actor_feat.size(0), 1).to(actor_feat.device)
        traj_pos_embed = self.traj_pos_enc(actor_positions)
        actor_feat = actor_feat + categorical_embeds.reshape(
            B * N, T, -1)[~hist_feat_key_padding] + traj_pos_embed
        actor_feat = self.post_project_agent(actor_feat)

        # 2. temo net
        for agent_tempo_blk in self.agent_tempo_net:
            actor_feat = agent_tempo_blk(
                actor_feat,
                key_padding_mask=hist_padding_mask[~hist_feat_key_padding],
            )
        # 3. pooling
        actor_feat_tmp = torch.zeros(B * N,
                                     actor_feat.shape[-1],
                                     device=actor_feat.device)
        agent_tempo_query = self.agent_tempo_query[None, :, :].repeat(
            actor_feat.shape[0], 1, 1)
        for pool_blk in self.agent_tempo_attend_pooling:
            agent_tempo_query = pool_blk(
                agent_tempo_query,
                actor_feat,
                actor_feat,
                key_padding_mask=hist_padding_mask[~hist_feat_key_padding],
            )
        actor_feat_tmp[~hist_feat_key_padding] = agent_tempo_query.squeeze(1)
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])

        # 3. lane embedding: projection
        lane_padding_mask = data["lane_padding_mask"]  # [B, M, L]
        B, M, L = lane_padding_mask.shape
        lane_vector = data["lane_positions"] - data[
            "lane_positions"][:, :, 0].unsqueeze(2)
        lane_feat = torch.cat(
            [
                lane_vector,
            ],
            dim=-1,
        )  # [B, M, L, 2]

        B, M, L, LF = lane_feat.shape
        lane_feat = lane_feat.view(-1, L, LF)
        lane_padding_mask = lane_padding_mask.view(-1, L)
        lane_feat_key_padding = data['lane_key_padding_mask'].reshape(B * M)
        lane_actor_feat = self.project_lane(lane_feat[~lane_feat_key_padding])
        lane_categorical_embeds = torch.cat([
            self.lane_type_emb(data["lane_attr"][..., 0].long()),
            self.lane_intersect_emb(data["lane_attr"][..., -1].long())
        ],
                                            dim=-1).unsqueeze(2).repeat(
                                                1, 1, L,
                                                1).reshape(B * M, L, -1)
        lane_categorical_embeds = self.project_lane_cate(
            lane_categorical_embeds)
        lane_actor_feat = lane_actor_feat + lane_categorical_embeds[
            ~lane_feat_key_padding]
        lane_actor_feat = self.post_project_lane(lane_actor_feat)

        for lane_blk in self.lane_vector_net:
            lane_actor_feat = lane_blk(
                lane_actor_feat,
                key_padding_mask=lane_padding_mask[~lane_feat_key_padding])

        lane_query = self.lane_vector_query[None, :, :].repeat(
            lane_actor_feat.shape[0], 1, 1)
        for lane_pool in self.lane_vector_attend_pooling:
            lane_query = lane_pool(
                lane_query,
                lane_actor_feat,
                lane_actor_feat,
                key_padding_mask=lane_padding_mask[~lane_feat_key_padding])

        lane_actor_feat_tmp = torch.zeros(B * M,
                                          lane_actor_feat.shape[-1],
                                          device=lane_actor_feat.device)

        lane_actor_feat_tmp[~lane_feat_key_padding] = lane_query.squeeze(1)
        lane_actor_feat = lane_actor_feat_tmp.reshape(B, M, -1)

        # 3. Scene
        # 3.1 Divide
        x_positions = data["x_positions"][:, :, 49]  # [B, N, 2] relative to ego
        angles = data["x_angles"][:, :, 49]  # [B, N]
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        x_pos_feat = torch.cat([x_positions, x_angles], dim=-1)  # [B, N, 4]
        x_pos_embed = self.agent_pos_embed(x_pos_feat)
        actor_feat = actor_feat + x_pos_embed
        actor_feat = actor_feat.reshape(B, N, -1)

        lane_centers = data["lane_positions"][:, :, 0].to(torch.float32)
        lanes_angle = torch.atan2(
            data["lane_positions"][..., 1, 1] -
            data["lane_positions"][..., 0, 1],
            data["lane_positions"][..., 1, 0] -
            data["lane_positions"][..., 0, 0],
        )
        lane_angles = torch.stack(
            [torch.cos(lanes_angle),
             torch.sin(lanes_angle)], dim=-1)
        lane_pos_feat = torch.cat([lane_centers, lane_angles], dim=-1)
        lane_pos_embed = self.lane_pos_embed(lane_pos_feat)
        lane_actor_feat = lane_actor_feat + lane_pos_embed

        x_encoder = torch.cat([actor_feat, lane_actor_feat], dim=1)
        key_padding_mask = torch.cat(
            [data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1)
        for blk in self.spa_net:
            x_encoder = blk(x_encoder, key_padding_mask=key_padding_mask)
        x_encoder = self.scene_norm(x_encoder)  # [B, M, D]

        # 6. mode query
        traj_query = (self.agent_traj_query.unsqueeze(0).unsqueeze(0).repeat(
            B, N, 1, 1) + x_pos_embed.unsqueeze(2)).reshape(
                B, N * self.num_modes, self.embed_dim)
        # mode2scene
        for i in range(0, len(self.cross_attender), 3):
            traj_query = self.cross_attender[i](
                traj_query,
                x_encoder,
                x_encoder,
                key_padding_mask=key_padding_mask)
            traj_query = traj_query.reshape(
                B, N, self.num_modes,
                self.embed_dim).permute(0, 2, 1,
                                        3).reshape(B * self.num_modes, N,
                                                   self.embed_dim)
            traj_query = traj_query + x_pos_embed.unsqueeze(1).repeat(
                1, self.num_modes, 1, 1).reshape(B * self.num_modes, N,
                                                 self.embed_dim)
            mask = data["x_key_padding_mask"].unsqueeze(1).repeat(
                1, self.num_modes, 1, 1).reshape(B * self.num_modes, N)

            traj_query = self.cross_attender[i + 1](traj_query,
                                                    key_padding_mask=mask)
            traj_query = traj_query.reshape(
                B, self.num_modes, N,
                self.embed_dim).permute(0, 2, 1,
                                        3).reshape(B * N, self.num_modes,
                                                   self.embed_dim)
            traj_query = traj_query[~hist_feat_key_padding]
            traj_query = self.cross_attender[i + 2](traj_query)
            traj_query_tmp = torch.zeros(B * N,
                                         self.num_modes,
                                         self.embed_dim,
                                         device=traj_query.device)
            traj_query_tmp[~hist_feat_key_padding] = traj_query
            traj_query = traj_query_tmp.view(B, N * self.num_modes,
                                             actor_feat.shape[-1])

        # propose and refine predict trajectory
        traj_query = traj_query.reshape(B, N, self.num_modes, -1)
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
        traj_query = traj_query.reshape(B, N * self.num_modes, self.embed_dim)
        for i in range(0, len(self.cross_attender_refine), 3):
            traj_query = self.cross_attender_refine[i](
                traj_query,
                x_encoder,
                x_encoder,
                key_padding_mask=key_padding_mask)
            traj_query = traj_query.reshape(
                B, N, self.num_modes,
                self.embed_dim).permute(0, 2, 1,
                                        3).reshape(B * self.num_modes, N,
                                                   self.embed_dim)
            traj_query = traj_query + x_pos_embed.unsqueeze(1).repeat(
                1, self.num_modes, 1, 1).reshape(B * self.num_modes, N,
                                                 self.embed_dim)
            mask = data["x_key_padding_mask"].unsqueeze(1).repeat(
                1, self.num_modes, 1, 1).reshape(B * self.num_modes, N)

            traj_query = self.cross_attender_refine[i + 1](
                traj_query, key_padding_mask=mask)
            traj_query = traj_query.reshape(
                B, self.num_modes, N,
                self.embed_dim).permute(0, 2, 1,
                                        3).reshape(B * N, self.num_modes,
                                                   self.embed_dim)
            traj_query = traj_query[~hist_feat_key_padding]
            traj_query = self.cross_attender_refine[i + 2](traj_query)
            traj_query_tmp = torch.zeros(B * N,
                                         self.num_modes,
                                         self.embed_dim,
                                         device=traj_query.device)
            traj_query_tmp[~hist_feat_key_padding] = traj_query
            traj_query = traj_query_tmp.view(B, N * self.num_modes,
                                             actor_feat.shape[-1])

        # decoder trajectory
        traj_query = traj_query.reshape(B, N, self.num_modes, -1)
        y_hat = self.decoder(traj_query)
        y_hat = y_hat + traj_propose.detach()

        # Scene scoring module using cross attention
        traj_query = traj_query.reshape(B, N * self.num_modes, -1)
        scene_query = self.scene_query.unsqueeze(0).repeat(B, 1, 1)
        for blk in self.scene_2_mode:
            scene_query = blk(scene_query, traj_query, traj_query)

        pi = self.prob_decoder(scene_query)

        return {"y_hat": y_hat, "pi": pi, "y_propose": traj_propose}
