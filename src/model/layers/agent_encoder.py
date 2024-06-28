import torch
from torch import nn
from src.model.layers.fourier_embedding import FourierEmbedding
from src.model.layers.transformer_blocks import Block, MLPLayer

from src.utils.weight_init import weight_init
from src.utils.utils import angle_between_2d_vectors


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
        # self.traj_hist_position_emb = nn.Embedding(agent_history_steps,
        #                                            hidden_dim)

        if embedding_type == "fourier":
            num_freq_bands: int = 64
            self.agent_projection = FourierEmbedding(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_freq_bands=num_freq_bands,
            )
            self.agent_rpe_projection = FourierEmbedding(
                input_dim=5,
                hidden_dim=hidden_dim,
                num_freq_bands=num_freq_bands,
                norm_layer=norm_layer)
        else:
            raise NotImplementedError(f"{embedding_type} is not implement!")

        self.agent_tempo_query = nn.Parameter(torch.randn(1, hidden_dim))
        self.agent_tempo_net = nn.ModuleList(
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
                update_rpe=tempo_depth - 1 > i,
            ) for i in range(tempo_depth))

        self.agent_pool_net = nn.ModuleList(
            Block(
                dim=hidden_dim,
                num_heads=num_head,
                drop=dropout,
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                attn_bias=attn_bias,
                ffn_bias=ffn_bias,
            ) for _ in range(1))

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
        agent_hist_angles = data['x_angles'][:, :, :50].contiguous(
        )  # instance frame
        agent_hist_angles_vec = torch.stack(
            [agent_hist_angles.cos(),
             agent_hist_angles.sin()], dim=-1)
        agent_hist_diff_vec = data['x']
        agent_hist_vel_diff = data['x_velocity_diff']
        agent_hist_vel_diff = torch.stack(
            [agent_hist_vel_diff.cos(),
             agent_hist_vel_diff.sin()], dim=-1)

        agent_categorical_embeds: torch.Tensor = self.agent_category_emb(
            data['x_attr'][..., -1].long()).unsqueeze(2).repeat(1, 1, 50, 1)

        agent_feat = torch.stack([
            torch.norm(agent_hist_diff_vec, p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=agent_hist_angles_vec,
                                     nbr_vector=agent_hist_diff_vec),
            torch.norm(agent_hist_vel_diff, p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=agent_hist_angles_vec,
                                     nbr_vector=agent_hist_vel_diff)
        ],
                                 dim=-1)

        # agent_feat: torch.Tensor = torch.cat([
        #     agent_hist_diff_vec,
        #     agent_hist_angles.unsqueeze(-1),
        #     agent_hist_vel_diff.unsqueeze(-1),
        # ],
        #                                      dim=-1)

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

        # TODO: check delete time pose embedding
        # agent_hist_positions = torch.arange(T).unsqueeze(0).repeat(
        #     agent_feat.shape[0], 1).to(agent_feat.device)
        # traj_pos_embed = self.traj_hist_position_emb(agent_hist_positions)
        # agent_feat = agent_feat + traj_pos_embed

        # 2. temo net for agent time self attention
        rel_pos = data["x_t_rpe"].reshape(B * N, T, T, 5)[~agent_padding_mask]
        rel_pos = self.agent_rpe_projection(
            rel_pos.reshape(rel_pos.shape[0] * rel_pos.shape[1], T,
                            5)).reshape((~agent_padding_mask).sum(), T, T,
                                        -1)  # [B*N, T, T, 128]
        for agent_tempo_blk in self.agent_tempo_net:
            agent_feat, rel_pos = agent_tempo_blk(
                agent_feat,
                key_padding_mask=agent_hist_padding_mask[~agent_padding_mask],
                position_bias=rel_pos)

        # 3. pooling
        agent_tempo_query = self.agent_tempo_query[None, :, :].repeat(
            agent_feat.shape[0], 1, 1)
        agent_feat = torch.cat([agent_feat, agent_tempo_query], dim=1)
        agent_hist_padding_mask = torch.cat([
            agent_hist_padding_mask,
            torch.zeros([B * N, 1]).to(agent_hist_padding_mask.dtype).to(
                agent_hist_padding_mask.device)
        ],
                                            dim=1)
        for agent_pool_blk in self.agent_pool_net:
            agent_feat = agent_pool_blk(
                agent_feat,
                key_padding_mask=agent_hist_padding_mask[~agent_padding_mask],
            )
        agent_feat_tmp = torch.zeros(B * N,
                                     agent_feat.shape[-1],
                                     device=agent_feat.device)

        agent_feat_tmp[~agent_padding_mask] = agent_feat[:, -1].clone()
        agent_feat = agent_feat_tmp.reshape(B, N, agent_feat.shape[-1])

        return agent_feat
