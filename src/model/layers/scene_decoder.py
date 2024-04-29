import torch
from torch import nn
from src.model.layers.fourier_embedding import FourierEmbedding
from src.model.layers.transformer_blocks import Block, CrossAttenderBlock, MLPLayer

from src.utils.weight_init import weight_init
from typing import List, Optional
from torch_scatter import scatter_mean


class SceneDecoder(nn.Module):

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
        depth: int,
        scene_score_depth: int,
        num_modes: int,
        future_steps: int,
        num_recurrent_steps: int,
    ) -> None:
        super().__init__()

        future_steps = future_steps
        num_recurrent_steps = num_recurrent_steps
        self.future_steps = future_steps
        self.num_recurrent_steps = num_recurrent_steps
        self.num_modes = num_modes
        self.use_refine = False

        self.x_pos_embed = MLPLayer(input_dim=5,
                                    hidden_dim=hidden_dim * 4,
                                    output_dim=hidden_dim)
        self.x_scene_pos_embed = MLPLayer(input_dim=5,
                                          hidden_dim=hidden_dim * 4,
                                          output_dim=hidden_dim)

        self.agent_traj_query = nn.Parameter(
            torch.randn(self.num_modes, hidden_dim))
        self.cross_attender_propose = nn.ModuleList()
        for i in range(depth):
            if self.use_refine:
                update_rpe=True
            else:
                update_rpe=depth - 1 > i
            self.cross_attender_propose.append(
                CrossAttenderBlock(
                    hidden_dim,
                    num_heads=8,
                    attn_drop=dropout,
                    kdim=hidden_dim,
                    vdim=hidden_dim,
                    post_norm=post_norm,
                    drop=dropout,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    attn_bias=attn_bias,
                    ffn_bias=ffn_bias,
                    use_simpl=True,
                    update_rpe=update_rpe
                ))
            self.cross_attender_propose.append(
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
                    update_rpe=update_rpe
                ))

        self.mode2mode_propose = Block(
            dim=hidden_dim,
            num_heads=num_head,
            attn_drop=dropout,
            post_norm=post_norm,
            drop=dropout,
            act_layer=act_layer,
            norm_layer=norm_layer,
            attn_bias=attn_bias,
            ffn_bias=ffn_bias,
        )
        if self.use_refine:
            self.traj_emb = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                bias=True,
                batch_first=False,
                dropout=0.0,
                bidirectional=False,
            )
            self.traj_emb_h0 = nn.Parameter(torch.zeros(1, hidden_dim))

            if embedding_type == "fourier":
                num_freq_bands = 64
                self.y_emb = FourierEmbedding(
                    input_dim=2,
                    hidden_dim=hidden_dim,
                    num_freq_bands=num_freq_bands,
                )
            else:
                raise NotImplementedError(f"{embedding_type} is not implement!")

            self.cross_attender_refine = nn.ModuleList()
            for i in range(depth):
                self.cross_attender_refine.append(
                    CrossAttenderBlock(
                        hidden_dim,
                        num_heads=8,
                        attn_drop=dropout,
                        kdim=hidden_dim,
                        vdim=hidden_dim,
                        post_norm=post_norm,
                        drop=dropout,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        attn_bias=attn_bias,
                        ffn_bias=ffn_bias,
                        use_simpl=True,
                        update_rpe=depth - 1 > i,
                    ))
                self.cross_attender_refine.append(
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
                        update_rpe=depth - 1 > i,
                    ))

            self.mode2mode_refine = Block(
                dim=hidden_dim,
                num_heads=num_head,
                attn_drop=dropout,
                post_norm=post_norm,
                drop=dropout,
                act_layer=act_layer,
                norm_layer=norm_layer,
                attn_bias=attn_bias,
                ffn_bias=ffn_bias,
            )

            self.to_loc_refine_pos = MLPLayer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim * 4,
            output_dim=future_steps * 2,
            )

        self.to_loc_propose_pos = MLPLayer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim * 4,
            output_dim=future_steps * 2 // self.num_recurrent_steps,
        )

        self.scene_query = nn.Parameter(torch.randn(self.num_modes,
                                                    hidden_dim))
        self.scene_2_mode = nn.ModuleList([
            CrossAttenderBlock(
                hidden_dim,
                num_heads=num_head,
                attn_drop=dropout,
                kdim=hidden_dim,
                vdim=hidden_dim,
                post_norm=post_norm,
                drop=dropout,
                act_layer=act_layer,
                norm_layer=norm_layer,
                attn_bias=attn_bias,
                ffn_bias=ffn_bias,
            ) for _ in range(scene_score_depth)
        ])

        self.prob_decoder = MLPLayer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim * 4,
            output_dim=1,
        )

        self.apply(weight_init)

    def forward(self, data: dict, scene_feat: torch.Tensor,
                agent_pos_emb: torch.Tensor):
        B, N, D = agent_pos_emb.shape
        scene_padding_mask = torch.cat(
            [data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1)
        agent_padding_mask = data["x_key_padding_mask"].reshape(B * N)
        traj_query: torch.Tensor = (
            self.agent_traj_query.unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)
            + scene_feat[:, :N].unsqueeze(2)).reshape(B, N * self.num_modes, D)

        locs_propose_pos: List[Optional[
            torch.Tensor]] = [None] * self.num_recurrent_steps
        # mode2scene
        x_scene_rel_pos = self.x_scene_pos_embed(data["x_scene_rpe"])
        x_rel_pos = self.x_pos_embed(data["x_rpe"])  # [B,N,N,D]
        x_rel_pos = (x_rel_pos[:, None, :, None, :, :].repeat(
            1, self.num_modes, 1, self.num_modes, 1,
            1).reshape(B, self.num_modes * N, self.num_modes * N, D))
        for t in range(self.num_recurrent_steps):
            for i in range(0, len(self.cross_attender_propose), 2):
                # Mode&Agent2Scene
                traj_query, x_scene_rel_pos = self.cross_attender_propose[i](
                    traj_query,
                    scene_feat,
                    scene_feat,
                    key_padding_mask=scene_padding_mask,
                    position_bias=x_scene_rel_pos,
                )
                traj_query = (traj_query.reshape(
                    B, N, self.num_modes,
                    D).permute(0, 2, 1, 3).reshape(B, self.num_modes * N, D))

                mask = (data["x_key_padding_mask"].unsqueeze(1).repeat(
                    1, self.num_modes, 1, 1).reshape(B, self.num_modes * N))

                traj_query, x_rel_pos = self.cross_attender_propose[i + 1](
                    traj_query, key_padding_mask=mask, position_bias=x_rel_pos)
                traj_query = (traj_query.reshape(
                    B, self.num_modes, N,
                    D).permute(0, 2, 1, 3).reshape(B, N * self.num_modes, D))
            # mode2mode
            traj_query = traj_query.reshape(B * N, self.num_modes, D)
            traj_query = traj_query[~agent_padding_mask]
            traj_query = self.mode2mode_propose(traj_query)
            traj_query_tmp = torch.zeros(B * N,
                                         self.num_modes,
                                         D,
                                         device=traj_query.device)
            traj_query_tmp[~agent_padding_mask] = traj_query.clone()
            traj_query = traj_query_tmp.reshape(B, N, self.num_modes, -1)

            # propose and refine predict trajectory
            locs_propose_pos[t] = self.to_loc_propose_pos(traj_query)
            traj_query = traj_query.reshape(B, N * self.num_modes, D)
        loc_propose_pos = torch.cumsum(
            torch.cat(locs_propose_pos,
                      dim=-1).reshape(-1, self.num_modes, self.future_steps,
                                      2),
            dim=-2,
        )
        if self.use_refine:
            traj_query = self.y_emb(
                torch.cat([loc_propose_pos.detach()],
                        dim=-1).reshape(B * N * self.num_modes,
                                        self.future_steps, 2))

            traj_query = traj_query.reshape(B, N, self.num_modes,
                                            self.future_steps, D)
            B, _, _, T, D = traj_query.shape
            traj_query = traj_query.reshape(B * N * self.num_modes, T,
                                            D).transpose(0, 1)
            traj_query = self.traj_emb(
                traj_query,
                self.traj_emb_h0.unsqueeze(1).repeat(1, traj_query.size(1),
                                                    1))[1].squeeze(0)
            traj_query = traj_query.reshape(B, N * self.num_modes, D)
            for i in range(0, len(self.cross_attender_refine), 2):
                traj_query, x_scene_rel_pos = self.cross_attender_refine[i](
                    traj_query,
                    scene_feat,
                    scene_feat,
                    key_padding_mask=scene_padding_mask,
                    position_bias=x_scene_rel_pos,
                )

                traj_query = (traj_query.reshape(B, N, self.num_modes, D).permute(
                    0, 2, 1, 3).reshape(B, self.num_modes * N, D))

                mask = (data["x_key_padding_mask"].unsqueeze(1).repeat(
                    1, self.num_modes, 1, 1).reshape(B, self.num_modes * N))

                traj_query, x_rel_pos = self.cross_attender_refine[i + 1](
                    traj_query, key_padding_mask=mask, position_bias=x_rel_pos)
                traj_query = (traj_query.reshape(B, self.num_modes, N, D).permute(
                    0, 2, 1, 3).reshape(B, N * self.num_modes, D))

            traj_query = traj_query.reshape(B * N, self.num_modes, D)
            traj_query = traj_query[~agent_padding_mask]
            traj_query = self.mode2mode_refine(traj_query)
            traj_query_tmp = torch.zeros(B * N,
                                        self.num_modes,
                                        D,
                                        device=traj_query.device)
            traj_query_tmp[~agent_padding_mask] = traj_query
            traj_query = traj_query_tmp.view(B, N * self.num_modes, D)

            # decoder trajectory
            traj_query = traj_query.reshape(B, N, self.num_modes, -1)
            loc_refine_pos = self.to_loc_refine_pos(traj_query[:, :N]).reshape(
                -1, self.num_modes, self.future_steps, 2)
            loc_refine_pos = loc_refine_pos + loc_propose_pos.detach()

            y_hat = loc_refine_pos.reshape(B, N, self.num_modes, self.future_steps,
                                        2)
            traj_propose = loc_propose_pos.reshape(B, N, self.num_modes,
                                                self.future_steps, 2)
        if not self.use_refine:                
            y_hat = loc_propose_pos.reshape(B, N, self.num_modes, self.future_steps,
                                        2)
        # Scene scoring module using cross attention
        traj_query = (traj_query.reshape(B, N, self.num_modes, -1).permute(
            0, 2, 1, 3).reshape(B * self.num_modes, N, -1))

        scene_query = (self.scene_query.unsqueeze(0).unsqueeze(2).repeat(
            B, 1, 1, 1).reshape(B * self.num_modes, 1, D))
        for blk in self.scene_2_mode:
            scene_query = blk(scene_query, traj_query, traj_query)
        scene_query = scene_query.reshape(B, self.num_modes, D)

        pi = self.prob_decoder(scene_query)

        if self.use_refine:
            return {
                "y_hat": y_hat,
                "pi": pi,
                "y_propose": traj_propose,
            }
        else:
            return {
                "y_hat": y_hat,
                "pi": pi,
            }


class SceneSimplDecoder(nn.Module):

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
        depth: int,
        scene_score_depth: int,
        num_modes: int,
        future_steps: int,
        num_recurrent_steps: int,
    ) -> None:
        super().__init__()

        future_steps = future_steps
        num_recurrent_steps = num_recurrent_steps
        self.future_steps = future_steps
        self.num_recurrent_steps = num_recurrent_steps
        self.num_modes = num_modes

        self.register_buffer("one_hot_mask", torch.eye(self.num_modes))

        self.traj_decoder = MLPLayer(
            input_dim=hidden_dim + self.num_modes,
            hidden_dim=hidden_dim * 4,
            output_dim=future_steps * 2,
        )

        self.prob_decoder = MLPLayer(
            input_dim=hidden_dim + self.num_modes,
            hidden_dim=hidden_dim * 4,
            output_dim=1,
        )

        self.apply(weight_init)

    def forward(self, data: dict, scene_feat: torch.Tensor,
                agent_pos_emb: torch.Tensor):
        B, N, D = agent_pos_emb.shape
        K = self.num_modes
        key_padding_mask = torch.cat(
            [data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1)
        x_actors = scene_feat[:, :N].unsqueeze(1).repeat(1, K, 1, 1)
        cls_token = scatter_mean(scene_feat, key_padding_mask.long(), dim=1)[:,
                                                                             0]
        cls_token = cls_token.unsqueeze(1).repeat(1, K, 1)
        one_hot_mask = self.one_hot_mask

        x_actors = torch.cat(
            [x_actors,
             one_hot_mask.view(1, K, 1, K).repeat(B, 1, N, 1)],
            dim=-1)
        cls_token = torch.cat(
            [cls_token, one_hot_mask.view(1, K, K).repeat(B, 1, 1)], dim=-1)

        y_hat = self.traj_decoder(x_actors).view(B, K, N, self.future_steps, 2)
        pi = self.prob_decoder(cls_token).view(B, K)

        return {"y_hat": y_hat, "pi": pi}
