import torch
from torch import nn
from src.model.layers.fourier_embedding import FourierEmbedding
from src.model.layers.transformer_blocks import Block, MLPLayer

from src.utils.weight_init import weight_init
from src.model.layers.common_layers import build_mlps


class LaneEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        embedding_type: str,
        num_head: int,
        dropout: float,
        act_layer: nn.Module,
        norm_layer: nn.Module,
        post_norm: bool,
        attn_bias: bool,
        ffn_bias: bool,
        tempo_depth: int,
    ) -> None:
        super().__init__()

        category_input_dim = 5
        intersection_dim = 2
        lane_history_steps = 20
        input_dim = 2
        lane_pos_input_dim = 4

        self.lane_category_emb = nn.Embedding(category_input_dim, hidden_dim)
        self.lane_intersect_emb = nn.Embedding(intersection_dim, hidden_dim)
        # self.lane_position_emb = nn.Embedding(lane_history_steps, hidden_dim)

        if embedding_type == "fourier":
            num_freq_bands: int = 64
            self.lane_projection = FourierEmbedding(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_freq_bands=num_freq_bands,
                norm_layer=norm_layer)
        else:
            raise NotImplementedError(f"{embedding_type} is not implement!")

        self.lane_vector_query = nn.Parameter(torch.randn(1, hidden_dim))
        self.lane_tempo_net = nn.ModuleList(
            Block(
                dim=hidden_dim,
                num_heads=num_head,
                drop=dropout,
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                attn_bias=attn_bias,
                ffn_bias=ffn_bias,
            ) for _ in range(tempo_depth))

        # self.lane_pos_embed = MLPLayer(input_dim=lane_pos_input_dim,
        #                                 hidden_dim=hidden_dim,
        #                                 output_dim=hidden_dim,
        #                                 norm_layer=None)

        self.apply(weight_init)

    def forward(self, data: dict):
        lane_pt_padding_mask: torch.Tensor = data[
            "lane_padding_mask"]  # [B, M, L]
        B, M, L = lane_pt_padding_mask.shape
        lane_vector = data["lane_positions"][:, :, 1:] - data[
            "lane_positions"][:, :, :-1]
        lane_vector = torch.cat(
            [torch.zeros(B, M, 1, 2).to(lane_vector.device), lane_vector],
            dim=2)
        # lane_angles = torch.arctan2(lane_vector[..., 1], lane_vector[..., 0])
        # lane_angles_vector = torch.stack([lane_angles.cos(), lane_angles.sin()], dim=-1)

        lane_feat = torch.cat(
            [
                lane_vector,
                # lane_angles.unsqueeze(-1),
            ],
            dim=-1,
        )  # [B, M, L, 2]

        B, M, L, LF = lane_feat.shape
        lane_feat = lane_feat.view(-1, L, LF)
        lane_pt_padding_mask = lane_pt_padding_mask.view(-1, L)
        lane_padding_mask = data['lane_key_padding_mask'].reshape(B * M)

        lane_categorical_embeds = torch.stack(
            [
                self.lane_category_emb(data["lane_attr"][..., 0].long()),
                self.lane_intersect_emb(data["lane_attr"][..., -1].long())
            ],
            dim=0).sum(dim=0).unsqueeze(2).repeat(1, 1, L,
                                                  1).reshape(B * M, L, -1)

        lane_actor_feat: torch.Tensor = self.lane_projection(
            lane_feat[~lane_padding_mask],
            categorical_embs=[lane_categorical_embeds[~lane_padding_mask]])

        lane_query = self.lane_vector_query[None, :, :].repeat(
            lane_actor_feat.shape[0], 1, 1)
        lane_actor_feat = torch.cat([lane_actor_feat, lane_query], dim=1)
        lane_pt_padding_mask = torch.cat([
            lane_pt_padding_mask,
            torch.zeros([B * M, 1]).to(lane_padding_mask.dtype).to(
                lane_padding_mask.device)
        ],
                                         dim=1)

        # lane_pt_positions = torch.arange(L+1).unsqueeze(0).repeat(
        #     lane_actor_feat.shape[0], 1).to(lane_actor_feat.device)
        # lane_pos_embed = self.lane_position_emb(lane_pt_positions)
        # lane_actor_feat = lane_actor_feat + lane_pos_embed

        for lane_blk in self.lane_tempo_net:
            lane_actor_feat = lane_blk(
                lane_actor_feat,
                key_padding_mask=lane_pt_padding_mask[~lane_padding_mask])

        lane_actor_feat_tmp = torch.zeros(B * M,
                                          lane_actor_feat.shape[-1],
                                          device=lane_actor_feat.device)

        lane_actor_feat_tmp[~lane_padding_mask] = lane_actor_feat[:,
                                                                  -1].clone()
        lane_actor_feat = lane_actor_feat_tmp.reshape(B, M, -1)

        # lane_centers = data["lane_positions"][:, :, 0].to(torch.float32)
        # lane_angles = torch.atan2(
        #     data["lane_positions"][..., 1, 1] -
        #     data["lane_positions"][..., 0, 1],
        #     data["lane_positions"][..., 1, 0] -
        #     data["lane_positions"][..., 0, 0],
        # )
        # lane_angles = torch.stack(
        #     [torch.cos(lane_angles),
        #      torch.sin(lane_angles)], dim=-1)
        # lane_pos_feat = torch.cat([lane_centers, lane_angles], dim=-1)
        # lane_pos_embed = self.lane_pos_embed(lane_pos_feat)
        # lane_pos_embed_tmp = torch.zeros(B * M,
        #                               lane_actor_feat.shape[-1],
        #                               device=lane_actor_feat.device)
        # lane_pos_embed_tmp[~lane_padding_mask] = lane_pos_embed
        # lane_pos_embed = lane_pos_embed_tmp.reshape(B, M, -1)
        # lane_actor_feat = lane_actor_feat + lane_pos_embed

        return lane_actor_feat


class MLP(nn.Module):

    def __init__(self, in_channels, hidden_unit, verbose=False):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(in_channels, hidden_unit),
                                 nn.LayerNorm(hidden_unit), nn.ReLU())

    def forward(self, x):
        x = self.mlp(x)
        return x


class LaneNet(nn.Module):

    def __init__(self, in_channels, hidden_unit, num_subgraph_layers):
        super(LaneNet, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(f'lmlp_{i}', MLP(in_channels,
                                                       hidden_unit))
            in_channels = hidden_unit * 2

    def forward(self, pts_lane_feats):
        '''
            Extract lane_feature from vectorized lane representation

        Args:
            pts_lane_feats: [batch size, max_pnum, pts, D]

        Returns:
            inst_lane_feats: [batch size, max_pnum, D]
        '''
        x = pts_lane_feats
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                # x [bs,max_lane_num,9,dim]
                x = layer(x)
                x_max = torch.max(x, -2)[0]
                x_max = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
                x = torch.cat([x, x_max], dim=-1)
        x_max = torch.max(x, -2)[0]
        return x_max


class LanePointNetEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        embedding_type: str,
        num_head: int,
        dropout: float,
        act_layer: nn.Module,
        norm_layer: nn.Module,
        post_norm: bool,
        attn_bias: bool,
        ffn_bias: bool,
        tempo_depth: int,
    ) -> None:
        super().__init__()
        # TODO: hidden_dim = 64
        input_dim = 5

        self.pre_mlps = build_mlps(c_in=input_dim,
                                   mlp_channels=[hidden_dim] * 3,
                                   ret_before_act=False)
        self.mlps = build_mlps(c_in=hidden_dim * 2,
                               mlp_channels=[hidden_dim] * 2,
                               ret_before_act=False)

        self.out_mlps = build_mlps(c_in=hidden_dim,
                                   mlp_channels=[hidden_dim, hidden_dim],
                                   ret_before_act=True,
                                   without_norm=True)

        self.apply(weight_init)

    def forward(self, data: dict):
        # TODO: feat encoder use vectornet encoding method
        lane_pt_padding_mask: torch.Tensor = data[
            "lane_padding_mask"]  # [B, M, L]
        lane_padding_mask = data["lane_key_padding_mask"]
        B, M, L = lane_pt_padding_mask.shape
        lane_vector = data["lane_positions"][:, :, 1:] - data[
            "lane_positions"][:, :, :-1]
        lane_vector = torch.cat(
            [torch.zeros(B, M, 1, 2).to(lane_vector.device), lane_vector],
            dim=2)
        lane_angles = torch.arctan2(lane_vector[..., 1], lane_vector[..., 0])
        lane_type = data["lane_attr"][..., 0][:, :, None,
                                              None].repeat(1, 1, L, 1)
        lane_width = data["lane_attr"][..., -1][:, :, None,
                                                None].repeat(1, 1, L, 1)

        lane = torch.cat(
            [
                lane_vector,
                lane_angles.unsqueeze(-1),
                lane_type,
                lane_width,
            ],
            dim=-1,
        )  # [B, M, L, 2]

        B, M, L, C = lane.shape
        # pre-mlp
        lane_feat_valid = self.pre_mlps(lane[~lane_pt_padding_mask])
        lane_feature = lane.new_zeros(B, M, L, lane_feat_valid.shape[-1])
        lane_feature[~lane_pt_padding_mask] = lane_feat_valid

        # max_pooling 1: after pre mlp, get global feature
        pooled_feature = lane_feature.max(dim=2)[0]  # [B, N, D]
        lane_feature = torch.cat(
            [lane_feature, pooled_feature[:, :, None, :].repeat(1, 1, L, 1)],
            dim=-1)

        # mlp
        lane_feat_valid = self.mlps(lane_feature[~lane_pt_padding_mask])
        feature_buffers = lane_feature.new_zeros(B, M, L,
                                                 lane_feat_valid.shape[-1])
        feature_buffers[~lane_pt_padding_mask] = lane_feat_valid

        # max-pooling 2: after mlp
        feature_buffers = feature_buffers.max(
            dim=2)[0]  # (batch_size, num_polylines, C)

        # out-mlp
        feature_buffers_valid = self.out_mlps(
            feature_buffers[~lane_padding_mask])  # (N, C)
        feature_buffers = feature_buffers.new_zeros(
            B, M, feature_buffers_valid.shape[-1])
        feature_buffers[~lane_padding_mask] = feature_buffers_valid

        return feature_buffers  # [B, N, D]
