import torch
from torch import nn
from src.model.layers.fourier_embedding import FourierEmbedding
from src.model.layers.transformer_blocks import Block, MLPLayer

from src.utils.weight_init import weight_init


class SceneEncoder(nn.Module):

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
        spa_depth: int,
    ) -> None:
        super().__init__()

        self.spa_net = nn.ModuleList(
            Block(dim=hidden_dim,
                  num_heads=num_head,
                  attn_drop=dropout,
                  post_norm=post_norm,
                  drop=dropout,
                  act_layer=act_layer,
                  norm_layer=norm_layer,
                  attn_bias=attn_bias,
                  ffn_bias=ffn_bias) for _ in range(spa_depth))

        self.pos_embed = MLPLayer(input_dim=4,
                                  hidden_dim=hidden_dim,
                                  output_dim=hidden_dim,
                                  norm_layer=None)
        self.pos_neg_embed = MLPLayer(input_dim=4,
                                  hidden_dim=hidden_dim,
                                  output_dim=hidden_dim,
                                  norm_layer=None)

        # self.scene_norm = norm_layer(hidden_dim)

        self.apply(weight_init)

    def forward(self, data: dict, agent_feat: torch.Tensor,
                lane_feat: torch.Tensor):
        scene_feat = torch.cat([agent_feat, lane_feat], dim=1)
        scene_padding_mask = torch.cat(
            [data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1)

        x_positions = data["x_positions"][:, :, 49]  # [B, N, 2]
        x_angles = data["x_angles"][:, :, 49]  # [B, N]
        # x_angles = torch.stack(
        #     [torch.cos(x_angles), torch.sin(x_angles)], dim=-1)
        x_angles = x_angles.unsqueeze(-1)
        x_pos_feat = torch.cat([x_positions, x_angles], dim=-1)  # [B, N, 4]
        lane_centers = data["lane_positions"][:, :, 0].to(torch.float32)
        lane_angles = torch.atan2(
            data["lane_positions"][..., 1, 1] -
            data["lane_positions"][..., 0, 1],
            data["lane_positions"][..., 1, 0] -
            data["lane_positions"][..., 0, 0],
        ).unsqueeze(-1)
        # lane_angles = torch.stack(
        #     [torch.cos(lane_angles),
        #      torch.sin(lane_angles)], dim=-1)
        lane_pos_feat = torch.cat([lane_centers, lane_angles], dim=-1)
        pos_feat = torch.cat([x_pos_feat, lane_pos_feat], dim=1)
        x1 = pos_feat[..., :2].unsqueeze(2).repeat(1, 1, pos_feat.size(1), 1)
        x2 = pos_feat[..., :2].unsqueeze(1)
        x = x1 - x2
        dist = torch.sqrt(((x1 - x2)**2).sum(-1))  # [B, N, N)]
        angle1 = pos_feat[..., 2].unsqueeze(2).repeat(1, 1, pos_feat.size(1))
        angle2 = pos_feat[..., 2].unsqueeze(1)
        angle_diff = angle1 - angle2
        rel_pos = torch.stack([dist, angle_diff], dim=-1)
        rel_pos = torch.cat([x, rel_pos], dim=-1)
        rel_pos = self.pos_embed(rel_pos) # [B, N, N]
        rel_pos_neg = torch.stack([dist, -angle_diff], dim=-1)
        rel_pos_neg = torch.cat([-x, rel_pos], dim=-1)
        rel_pos_neg = self.pos_neg_embed(rel_pos_neg)  # [B, N, N]

        for blk in self.spa_net:
            scene_feat = blk(scene_feat,
                             key_padding_mask=scene_padding_mask,
                             position_bias=[rel_pos, rel_pos_neg])

        # scene_feat = self.scene_norm(scene_feat)

        return scene_feat
