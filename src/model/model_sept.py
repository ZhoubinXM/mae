from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.agent_embedding import AgentEmbeddingLayer
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.multimodal_decoder import MultimodalDecoder, SEPTMultimodalDecoder, SEPTProposeDecoder
from .layers.transformer_blocks import Block, CrossAttenderBlock
from .layers.relative_position_bias import RelativePositionBias


class ModelSEPT(nn.Module):

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
        self.viz = True
        self.embed_dim = embed_dim
        self.road_embed_type = road_embed_type
        self.tempo_embed_type = tempo_embed_type
        self.project_agent_hist = nn.Sequential(nn.Linear(5, embed_dim),
                                                nn.GELU())
        # self.agent_norm = nn.LayerNorm(embed_dim)
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
            # self.road_pos_enc = nn.Embedding(19, embed_dim)
            self.road_net = nn.ModuleList(
                Block(dim=embed_dim,
                      num_heads=num_heads,
                      attn_drop=0.2,
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

        # self.pos_embed = nn.Sequential(
        #     nn.Linear(4, embed_dim),
        #     nn.GELU(),
        #     nn.Linear(embed_dim, embed_dim),
        # )
        # self.actor_type_embed = nn.Parameter(torch.Tensor(4, embed_dim))
        # self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        # self.spa_net = nn.ModuleList(
        #     Block(dim=embed_dim,
        #           num_heads=num_heads,
        #           attn_drop=0.2,
        #           post_norm=post_norm) for i in range(2))
        # self.norm = nn.LayerNorm(embed_dim)
        self.norm_lane = nn.LayerNorm(embed_dim)
        self.norm_agent = nn.LayerNorm(embed_dim)

        self.agent_traj_query = nn.Parameter(torch.randn(1, 6, embed_dim))

        self.mode2agent = nn.ModuleList([
            CrossAttenderBlock(embed_dim,
                               num_heads=8,
                               attn_drop=0.2,
                               kdim=embed_dim,
                               vdim=embed_dim,
                               post_norm=post_norm) for _ in range(2)])
        
        self.mode2road = nn.ModuleList([
            CrossAttenderBlock(embed_dim,
                               num_heads=8,
                               attn_drop=0.2,
                               kdim=embed_dim,
                               vdim=embed_dim,
                               post_norm=post_norm) for _ in range(2)])


        # self.cross_attender = nn.ModuleList([
        #     CrossAttenderBlock(embed_dim,
        #                        num_heads=8,
        #                        attn_drop=0.2,
        #                        kdim=embed_dim,
        #                        vdim=embed_dim,
        #                        post_norm=post_norm) for _ in range(3)
        # ])
        self.mode_2_mode_propose = nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  attn_drop=0.2,
                  post_norm=post_norm) for _ in range(1)
        ])
        self.decoder_propose = SEPTProposeDecoder(embed_dim,
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

        # self.cross_attender_refine = nn.ModuleList([
        #     CrossAttenderBlock(embed_dim,
        #                        num_heads=8,
        #                        attn_drop=0.2,
        #                        kdim=embed_dim,
        #                        vdim=embed_dim,
        #                        post_norm=post_norm) for _ in range(3)
        # ])

        self.mode2agent_refine = nn.ModuleList([
            CrossAttenderBlock(embed_dim,
                               num_heads=8,
                               attn_drop=0.2,
                               kdim=embed_dim,
                               vdim=embed_dim,
                               post_norm=post_norm) for _ in range(2)])
        
        self.mode2road_refine = nn.ModuleList([
            CrossAttenderBlock(embed_dim,
                               num_heads=8,
                               attn_drop=0.2,
                               kdim=embed_dim,
                               vdim=embed_dim,
                               post_norm=post_norm) for _ in range(2)])
        self.mode_2_mode_refine = nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  attn_drop=0.2,
                  post_norm=post_norm) for _ in range(1)
        ])
        self.decoder = SEPTMultimodalDecoder(embed_dim,
                                             future_steps,
                                             hidden_dim=512)

        # self.decoder = MultimodalDecoder(embed_dim, future_steps)

        self.initialize_weights()

    def initialize_weights(self):
        # nn.init.normal_(self.actor_type_embed, std=0.02)
        # nn.init.normal_(self.lane_type_embed, std=0.02)
        # # nn.init.normal_(self.agent_traj_query, std=0.02)

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
            # and (
            #     set(['tempo_net', 'project_road', 'road_net', 'spa_net'])
            #     & set(k.split(".")))
        }
        print("[INFO]: Load pretrained weight: {}".format(ckpt_path))
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def forward(self, data):
        hist_padding_mask = data["x_padding_mask"][:, :, :50]
        # time serires sequence mask; true means masked
        hist_key_padding_mask = data[
            "x_key_padding_mask"]  # agent padding mask
        hist_feat = torch.cat(
            [
                data["x_trans"][..., :50, :],  # [B, A, L, 2]
                # data['x'],
                # data["x_velocity_diff"][..., None],
                data["x_velocity"][..., :50].unsqueeze(-1),
                # torch.cos(data['x_angles'][..., :50].unsqueeze(-1)),
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
        if self.road_embed_type != "full":
            lane_key_padding_mask = data['lane_key_padding_mask']  # [B, R]
        if self.road_embed_type == "transform":
            lane_feat = torch.cat(
                [
                    # data["lane_positions"] - data["lane_centers"].unsqueeze(-2),
                    data["lane_positions"][..., :-1, :],  # [B, R, L, 2]
                    data["lane_positions"][..., 1:, :],  # [B, R, L, 2]
                    data['lane_angles'][..., None, None].repeat(1, 1, 19, 1),
                    # torch.sin(data['lane_angles'][..., None, None].repeat(
                    #     1, 1, 19, 1)),
                    data['lane_attr'][..., None, :].repeat(1, 1, 19, 1),
                ],
                dim=-1,
            )  # [B, R, L, 4]
            # lane_padding_mask = lane_padding_mask[:, :, :-1]
            lane_padding_mask = torch.cat([
                lane_padding_mask[..., :-1][..., None],
                lane_padding_mask[..., 1:][..., None],
            ],
                                          dim=-1).any(-1)
            B, M, L, D = lane_feat.shape
            lane_feat = lane_feat.view(-1, L, D)
            lane_padding_mask = lane_padding_mask.view(-1, L)
            lane_feat_key_padding = lane_padding_mask.all(-1)

            # 4. road projection layer
            lane_actor_feat = self.project_road(
                lane_feat[~lane_feat_key_padding])

            # 5. road feature extract
            # positions = torch.arange(L).unsqueeze(0).repeat(
            #     lane_actor_feat.size(0), 1).to(lane_actor_feat.device)
            # pos_embed = self.road_pos_enc(positions)
            # lane_actor_feat = lane_actor_feat + pos_embed
            for blk in self.road_net:
                lane_actor_feat = blk(
                    lane_actor_feat,
                    key_padding_mask=lane_padding_mask[~lane_feat_key_padding])

            lane_actor_feat_tmp = torch.zeros(B * M,
                                              lane_actor_feat.shape[-1],
                                              device=lane_actor_feat.device)
            # lane_actor_feat, _ = torch.max(lane_actor_feat, dim=1)  # [B*N, D_m]
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

        # x_centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
        # angles = torch.cat([data["x_angles"][:, :, 49], data["lane_angles"]], dim=1)
        # x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        # pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        # pos_embed = self.pos_embed(pos_feat)

        # actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()]
        # # lane_type_embed = self.lane_type_embed.repeat(B, M, 1)
        # actor_feat += actor_type_embed
        # lane_actor_feat += lane_type_embed

        # 6. spa net
        # x_encoder = torch.cat([actor_feat, lane_actor_feat], dim=1)
        # # x_encoder = x_encoder + pos_embed
        # if self.road_embed_type == "full":
        #     key_padding_mask = torch.cat(
        #         [data["x_key_padding_mask"],
        #          lane_padding_mask.reshape(B, R)],
        #         dim=1)
        # else:
        #     key_padding_mask = torch.cat([
        #         data["x_key_padding_mask"],
        #         lane_feat_key_padding.reshape(B, -1)
        #     ],
        #                                  dim=1)

        # for blk in self.spa_net:
        #     x_encoder = blk(x_encoder, key_padding_mask=key_padding_mask)
        # # pre-norm
        # x_encoder = self.norm(x_encoder)  # [B, A+R, D]

        # 7. cross-attender
        traj_query = self.agent_traj_query.repeat(B, 1,
                                                  1)  # [B,6,D]
        actor_feat = self.norm_agent(actor_feat)
        lane_actor_feat = self.norm_lane(lane_actor_feat)
        for blk in self.mode2agent:
            traj_query = blk(traj_query,
                             actor_feat,
                             actor_feat,
                             key_padding_mask=data["x_key_padding_mask"])
        for blk in self.mode2road:
            traj_query = blk(traj_query,
                             lane_actor_feat,
                             lane_actor_feat,
                             key_padding_mask=lane_feat_key_padding.reshape(B, -1))
        # mode2scene
        # for blk in self.cross_attender:
        #     traj_query = blk(traj_query,
        #                      x_encoder,
        #                      x_encoder,
        #                      key_padding_mask=key_padding_mask)

        # mode2mode
        for blk in self.mode_2_mode_propose:
            traj_query = blk(traj_query)

        traj_propose = self.decoder_propose(traj_query)  # [B, 6, 60, 2]
        traj_query = traj_propose.reshape(-1, 2)
        traj_query = self.traj_project(traj_propose.detach())  # [B*6*60, 256]
        traj_query = traj_query.reshape(-1, 60, self.embed_dim).transpose(0, 1)
        traj_query = self.traj_emb(
            traj_query,
            self.traj_emb_h0.unsqueeze(1).repeat(1, traj_query.size(1),
                                                 1))[1].squeeze(0)
        traj_query = traj_query.reshape(B, 6, self.embed_dim)
        # mode2scene
        # for blk in self.cross_attender_refine:
        #     traj_query = blk(traj_query,
        #                      x_encoder,
        #                      x_encoder,
        #                      key_padding_mask=key_padding_mask)
        for blk in self.mode2agent_refine:
            traj_query = blk(traj_query,
                             actor_feat,
                             actor_feat,
                             key_padding_mask=data["x_key_padding_mask"])
        for blk in self.mode2road_refine:
            traj_query = blk(traj_query,
                             lane_actor_feat,
                             lane_actor_feat,
                             key_padding_mask=lane_feat_key_padding.reshape(B, -1))

        # mode2mode
        for blk in self.mode_2_mode_refine:
            traj_query = blk(traj_query)

        # x_agent = x_encoder[:, 0]
        # x_agent = self.norm(traj_query)
        y_hat, pi = self.decoder(traj_query)
        y_hat += traj_propose.detach()

        # only viz when batch_size = 1
        if self.viz and B == 1:
            save_dir = "./viz_4"
            import os
            import matplotlib.pyplot as plt
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            A, T, _ = hist_feat.shape
            hist_feat = hist_feat.detach().cpu()
            y = data['y'][0, 0].reshape(60, 2).cpu()
            y_predict = traj_propose[0].reshape(6, 60, 2).cpu()
            fig, ax = plt.subplots(1, 2, figsize=(25, 15))
            ax[0].plot(y[:, 0], y[:, 1], c='g',  linewidth=2)
            ax[1].plot(y[:, 0], y[:, 1], c='g',  linewidth=2)
            for i in range(6):
                ax[0].plot(y_predict[i, :, 0],
                        y_predict[i, :, 1],
                        c='r',alpha=0.7,
                        linewidth=1)
                ax[1].plot(y_predict[i, :, 0],
                        y_predict[i, :, 1],
                        c='r',alpha=0.7,
                        linewidth=1)
            for j in range(A):
                if j == 0:
                    c = "k"
                else:
                    c = "grey"
                ax[0].plot(hist_feat[j, :, 0],
                           hist_feat[j, :, 1],
                           c=c,
                           alpha=0.7, linewidth=1)
                hist_feat_viz_j = hist_feat[j]
                seq_padding_mask = hist_padding_mask[j]
                valid_hist = hist_feat_viz_j[~seq_padding_mask]
                ax[1].plot(valid_hist[:, 0], valid_hist[:, 1], c=c, alpha=0.7, linewidth=1)
            lane_feat = lane_feat.detach().cpu()
            R, L, _ = lane_feat.shape
            for k in range(R):
                ax[0].plot(lane_feat[k, :, 0],
                           lane_feat[k, :, 1],
                           c='grey',
                           alpha=0.5, linewidth=1)
                lane_feat_viz_k = lane_feat[k]
                padding_mask = lane_padding_mask[k]
                valid_lane = lane_feat_viz_k[~padding_mask]
                # ax[1].plot(valid_lane[:, 0], valid_lane[:, 1], c='grey', alpha=0.4)
                for u in range(valid_lane.shape[0]):
                    ax[1].plot([valid_lane[u, 0], valid_lane[u, 2]],
                               [valid_lane[u, 1], valid_lane[u, 3]],
                               c='grey',
                               alpha=0.3, linewidth=1)
            plt.savefig(os.path.join(save_dir,
                                     f"{data['scenario_id'][0]}.jpg"))

        return {
            "y_hat": y_hat,
            "pi": pi,
            "y_hat_propose": traj_propose,
        }
