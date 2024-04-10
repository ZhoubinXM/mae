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

        # self.scene_norm = norm_layer(hidden_dim)

        self.apply(weight_init)

    def forward(self, data: dict, agent_feat: torch.Tensor,
                lane_feat: torch.Tensor):
        scene_feat = torch.cat([agent_feat, lane_feat], dim=1)
        scene_padding_mask = torch.cat([
            data["x_key_padding_mask"], data["lane_padding_mask"].reshape(
                scene_feat.shape[0], -1)
        ],
                                       dim=1)

        for blk in self.spa_net:
            scene_feat = blk(scene_feat, key_padding_mask=scene_padding_mask)

        # scene_feat = self.scene_norm(scene_feat)

        return scene_feat
