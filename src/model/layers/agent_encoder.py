import torch
from torch import nn
from src.model.layers.fourier_embedding import FourierEmbedding
from src.model.layers.transformer_blocks import Block, MLPLayer

from src.utils.weight_init import weight_init


class AgentEncoder(nn.Module):

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
        agent_history_steps = 50
        input_dim = 4
        agent_pos_input_dim = 4

        self.agent_category_emb = nn.Embedding(category_input_dim, hidden_dim)
        self.traj_hist_position_emb = nn.Embedding(agent_history_steps,
                                                   hidden_dim)

        if embedding_type == "fourier":
            num_freq_bands: int = 64
            self.agent_projection = FourierEmbedding(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_freq_bands=num_freq_bands,
            )
        else:
            raise NotImplementedError(f"{embedding_type} is not implement!")

        self.agent_tempo_query = nn.Parameter(torch.randn(1, hidden_dim))
        self.agent_tempo_net = nn.ModuleList(
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

        # self.agent_pos_embed = MLPLayer(
        #     input_dim=agent_pos_input_dim,
        #     hidden_dim=hidden_dim,
        #     output_dim=hidden_dim,
        #     norm_layer=None,
        # )

        self.apply(weight_init)

    def forward(self, data: dict):
        agent_hist_padding_mask = data["x_padding_mask"][:, :, :50]
        agent_padding_mask = data["x_key_padding_mask"]
        agent_hist_angles = data['x_angles'][:, :, :50].contiguous() # instance frame
        # agent_hist_angles_vec = torch.stack(
        #     [agent_hist_angles.cos(),
        #      agent_hist_angles.sin()], dim=-1)
        agent_hist_diff_vec = data['x']
        agent_hist_vel_diff = data['x_velocity_diff']
        # agent_hist_vel_diff_vec = torch.stack(
        #     [agent_hist_vel_diff.cos(),
        #      agent_hist_vel_diff.sin()], dim=-1)

        agent_categorical_embeds: torch.Tensor = self.agent_category_emb(
            data['x_attr'][..., -1].long()).unsqueeze(2).repeat(1, 1, 50, 1)

        agent_feat: torch.Tensor = torch.cat([
            agent_hist_diff_vec,
            agent_hist_angles.unsqueeze(-1),
            agent_hist_vel_diff.unsqueeze(-1),
        ],
                                             dim=-1)

        B, N, T, F = agent_feat.shape
        agent_feat = agent_feat.reshape(B * N, T, F)
        agent_padding_mask = agent_padding_mask.view(
            B * N)  # agent wheather masked
        agent_hist_padding_mask = agent_hist_padding_mask.reshape(B * N, T)

        # 1. Hist embdedding: projection
        agent_feat = self.agent_projection(
            continuous_inputs=agent_feat[~agent_padding_mask],
            categorical_embs=[
                agent_categorical_embeds.reshape(B * N, T,
                                                 -1)[~agent_padding_mask]
            ])

        agent_hist_positions = torch.arange(T).unsqueeze(0).repeat(
            agent_feat.shape[0], 1).to(agent_feat.device)
        traj_pos_embed = self.traj_hist_position_emb(agent_hist_positions)
        agent_feat = agent_feat + traj_pos_embed

        agent_tempo_query = self.agent_tempo_query[None, :, :].repeat(
            agent_feat.shape[0], 1, 1)
        agent_feat = torch.cat([agent_tempo_query, agent_feat], dim=1)
        agent_hist_padding_mask = torch.cat([
            torch.zeros([B * N, 1]).to(agent_hist_padding_mask.dtype).to(
                agent_hist_padding_mask.device), agent_hist_padding_mask
        ],
                                            dim=1)

        # 2. temo net for agent time self attention
        for agent_tempo_blk in self.agent_tempo_net:
            agent_feat = agent_tempo_blk(
                agent_feat,
                key_padding_mask=agent_hist_padding_mask[~agent_padding_mask],
            )

        # 3. pooling
        agent_feat_tmp = torch.zeros(B * N,
                                     agent_feat.shape[-1],
                                     device=agent_feat.device)

        agent_feat_tmp[~agent_padding_mask] = agent_feat[:, 0]
        agent_feat = agent_feat_tmp.reshape(B, N, agent_feat.shape[-1])

        # x_positions = data["x_positions"][:, :, 49]  # [B, N, 2]
        # x_angles = data["x_angles"][:, :, 49]  # [B, N]
        # x_angles = torch.stack(
        #     [torch.cos(x_angles), torch.sin(x_angles)], dim=-1)
        # # x_angles = x_angles.unsqueeze(-1)
        # x_pos_feat = torch.cat([x_positions, x_angles], dim=-1)  # [B, N, 4]
        # x_pos_embed = self.agent_pos_embed(x_pos_feat)
        # # x_pos_embed_tmp = torch.zeros(B * N,
        # #                               agent_feat.shape[-1],
        # #                               device=agent_feat.device)
        # # x_pos_embed_tmp[~agent_padding_mask] = x_pos_embed
        # # x_pos_embed = x_pos_embed_tmp.reshape(B, N, -1)
        # agent_feat = agent_feat + x_pos_embed
        # agent_feat = agent_feat.reshape(B, N, -1)

        return agent_feat
