import torch
from torch import nn
from src.model.layers.fourier_embedding import FourierEmbedding
from src.model.layers.transformer_blocks import Block, MLPLayer, CrossAttenderBlock

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

        self.agent2map = nn.ModuleList(
            CrossAttenderBlock(hidden_dim,
                               num_heads=8,
                               attn_drop=dropout,
                               kdim=hidden_dim,
                               vdim=hidden_dim,
                               post_norm=post_norm,
                               drop=dropout,
                               act_layer=act_layer,
                               norm_layer=norm_layer,
                               attn_bias=attn_bias,
                               ffn_bias=ffn_bias) for _ in range(spa_depth))

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
        # scene_feat = torch.cat([agent_feat, lane_feat], dim=1)
        # scene_padding_mask = torch.cat(
        #     [data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1)
        B, N, T, D = agent_feat.shape
        agent_feat = agent_feat.reshape(B, N * T, D)
        for blk in self.agent2map:
            agent_feat = blk(agent_feat,
                             lane_feat,
                             lane_feat,
                             key_padding_mask=data["lane_key_padding_mask"])
        agent_feat = agent_feat.reshape(B, N, T,
                                        D).permute(0, 2, 1,
                                                   3).reshape(B * T, N, D)
        for blk in self.spa_net:
            agent_feat = blk(
                agent_feat,
                key_padding_mask=data["x_padding_mask"][:, :, :50].permute(
                    0, 2, 1).reshape(B * T, N))
        agent_feat = agent_feat.reshape(B, T, N,
                                        D).permute(0, 2, 1,
                                                   3).reshape(B, N, T, D)

        # scene_feat = self.scene_norm(scene_feat)
        scene_feat = [agent_feat, lane_feat]
        return scene_feat
