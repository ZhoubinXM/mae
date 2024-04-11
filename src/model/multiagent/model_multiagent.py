import torch
import torch.nn as nn

# from ..layers.agent_embedding import AgentEmbeddingLayer
from ..layers.transformer_blocks import Block, CrossAttenderBlock
from torch_scatter import scatter_mean
from ..layers.multimodal_decoder import MultiAgentDecoder, MultiAgentProposeDecoder
from src.model.layers.agent_encoder import AgentEncoder
from src.model.layers.lane_encoder import LaneEncoder
from src.model.layers.scene_encoder import SceneEncoder
from src.model.layers.scene_decoder import SceneDecoder


class ModelMultiAgent(nn.Module):

    def __init__(
        self,
        embed_dim=128,
        embedding_type="fourier",
        encoder_depth=3,
        spa_depth=3,
        decoder_depth=3,
        scene_score_depth=2,
        num_heads=8,
        attn_bias=True,
        ffn_bias=True,
        dropout=0.1,
        future_steps: int = 60,
        num_modes: int = 6,
        use_cls_token: bool = True,
        act_layer=nn.ReLU,
        norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.embed_dim = embed_dim
        self.future_steps = future_steps
        self.use_cls_token = use_cls_token

        # Agent Encoder
        self.agent_encoder = AgentEncoder(hidden_dim=embed_dim,
                                          embedding_type=embedding_type,
                                          num_head=num_heads,
                                          dropout=dropout,
                                          act_layer=act_layer,
                                          norm_layer=norm_layer,
                                          post_norm=False,
                                          attn_bias=attn_bias,
                                          ffn_bias=ffn_bias,
                                          tempo_depth=encoder_depth)

        # Lane Encoder
        self.lane_encoder = LaneEncoder(hidden_dim=embed_dim,
                                        embedding_type=embedding_type,
                                        num_head=num_heads,
                                        dropout=dropout,
                                        act_layer=act_layer,
                                        norm_layer=norm_layer,
                                        post_norm=False,
                                        attn_bias=attn_bias,
                                        ffn_bias=ffn_bias,
                                        tempo_depth=encoder_depth)

        # Scene Encoder
        self.scene_encoder = SceneEncoder(hidden_dim=embed_dim,
                                          embedding_type=embedding_type,
                                          num_head=num_heads,
                                          dropout=dropout,
                                          act_layer=act_layer,
                                          norm_layer=norm_layer,
                                          post_norm=False,
                                          attn_bias=attn_bias,
                                          ffn_bias=ffn_bias,
                                          spa_depth=spa_depth)

        # Scene Decoder
        self.scene_decoder = SceneDecoder(hidden_dim=embed_dim,
                                          embedding_type=embedding_type,
                                          num_head=num_heads,
                                          dropout=dropout,
                                          act_layer=act_layer,
                                          norm_layer=norm_layer,
                                          post_norm=False,
                                          attn_bias=attn_bias,
                                          ffn_bias=ffn_bias,
                                          num_modes=num_modes,
                                          future_steps=future_steps,
                                          num_recurrent_steps=3,
                                          depth=decoder_depth,
                                          scene_score_depth=scene_score_depth)

    def forward(self, data):
        agent_feat, x_pos_embed = self.agent_encoder(data)
        lane_actor_feat, lane_pos_embed = self.lane_encoder(data)
        scene_feat, scene_pos_embed = self.scene_encoder(
            data, agent_feat, lane_actor_feat, x_pos_embed, lane_pos_embed)
        return self.scene_decoder(data, scene_feat, x_pos_embed,
                                  scene_pos_embed)
