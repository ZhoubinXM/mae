import torch
from torch import nn
from src.model.layers.fourier_embedding import FourierEmbedding
from src.model.layers.transformer_blocks import Block, MLPLayer
from src.model.layers.utils.transformer.transformer_encoder_layer import TransformerEncoderLayer
from src.model.layers.utils.transformer.position_encoding_utils import gen_sineembed_for_position, gen_relative_input

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
            Block(
                dim=hidden_dim,
                num_heads=num_head,
                attn_drop=dropout,
                post_norm=post_norm,
                drop=dropout,
                act_layer=act_layer,
                norm_layer=norm_layer,
                attn_bias=attn_bias,
                ffn_bias=ffn_bias,
                use_simpl=True,
                update_rpe=spa_depth - 1 > i,
            ) for i in range(spa_depth))

        # self.scene_norm = norm_layer(hidden_dim)
        self.pos_embed = MLPLayer(input_dim=5,
                                  hidden_dim=hidden_dim * 4,
                                  output_dim=hidden_dim)

        self.apply(weight_init)

    def forward(self, data: dict, agent_feat: torch.Tensor,
                lane_feat: torch.Tensor):
        scene_feat = torch.cat([agent_feat, lane_feat], dim=1)
        scene_padding_mask = torch.cat(
            [data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1)

        rel_pos = data["rpe"]
        rel_pos = self.pos_embed(rel_pos)  # [B, N, N]

        for blk in self.spa_net:
            scene_feat, rel_pos = blk(scene_feat,
                                      key_padding_mask=scene_padding_mask,
                                      position_bias=rel_pos)

        return scene_feat, rel_pos


class SceneMTREncoder(nn.Module):

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
            TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_head,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                normalize_before=False,
            ) for _ in range(spa_depth))

        self.pos_embed = nn.Linear(2 * hidden_dim, hidden_dim)

        self.apply(weight_init)

    def forward(self, data: dict, agent_feat: torch.Tensor,
                lane_feat: torch.Tensor):
        B, N, D = agent_feat.shape
        _, M, _ = lane_feat.shape

        scene_feat = torch.cat([agent_feat, lane_feat], dim=1)
        scene_padding_mask = torch.cat(
            [data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1)

        _, A = scene_padding_mask.shape
        assert A == N + M

        scene_relative_pose = data["scene_relative_pose"].reshape(
            B * A, A, 3)  # [B*A, A, 3]

        scene_feat = scene_feat.reshape(B, A, D)

        scene_rel_padding_mask, scene_relative_pose_embed, self_pos_embed = gen_relative_input(
            scene_relative_pose=scene_relative_pose,
            scene_padding_mask=scene_padding_mask,
            pos_embed=self.pos_embed,
            hidden_dim=D)

        for blk in self.spa_net:
            scene_feat = blk(src=scene_feat,
                             src_key_padding_mask=scene_rel_padding_mask,
                             pos=[self_pos_embed, scene_relative_pose_embed])

        return [scene_feat[:, :N], scene_feat[:, N:]], _
