import torch
from torch import nn
from src.model.layers.fourier_embedding import FourierEmbedding
from src.model.layers.transformer_blocks import Block, CrossAttenderBlock, MLPLayer
from src.model.layers.common_layers import build_mlps
from src.model.layers.utils.transformer.position_encoding_utils import gen_sineembed_for_position, batch_nms, gen_relative_input
from src.model.layers.utils.transformer.transformer_decoder_layer import TransformerDecoderLayer
from src.model.layers.utils.transformer.transformer_encoder_layer import TransformerEncoderLayer
from src.datamodule.av2_dataset import calculate_relative_positions_angles

from src.utils.weight_init import weight_init
from typing import List, Optional
from torch_scatter import scatter_mean
import numpy as np
import os
import copy

TYPE_MAP = {0: "Vehicle", 1: "Pedestrain", 2: "Cyclist"}


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
        self.use_refine = True

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
                update_rpe = True
            else:
                update_rpe = depth - 1 > i
            self.cross_attender_propose.append(
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
                                   ffn_bias=ffn_bias,
                                   use_simpl=True,
                                   update_rpe=update_rpe))
            self.cross_attender_propose.append(
                Block(dim=hidden_dim,
                      num_heads=num_head,
                      attn_drop=dropout,
                      post_norm=post_norm,
                      drop=dropout,
                      act_layer=act_layer,
                      norm_layer=norm_layer,
                      attn_bias=attn_bias,
                      ffn_bias=ffn_bias,
                      use_simpl=True,
                      update_rpe=update_rpe))

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
                raise NotImplementedError(
                    f"{embedding_type} is not implement!")

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
            output_dim=future_steps * 2,
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

        # locs_propose_pos: List[Optional[
        #     torch.Tensor]] = [None] * self.num_recurrent_steps
        # mode2scene
        x_scene_rel_pos = self.x_scene_pos_embed(
            data["x_scene_rpe"]).unsqueeze(2).repeat(
                1, 1, self.num_modes, 1, 1).reshape(B, N * self.num_modes, -1,
                                                    D)
        x_rel_pos = self.x_pos_embed(data["x_rpe"])  # [B,N,N,D]
        x_rel_pos = (x_rel_pos[:, None, :, :, :].repeat(
            1, self.num_modes, 1, 1, 1).reshape(B * self.num_modes, N, N, D))

        x_scene_padding_mask = (
            data["x_key_padding_mask"].unsqueeze(2)
            & scene_padding_mask.unsqueeze(1)).unsqueeze(2).repeat(
                1, 1, self.num_modes, 1, 1).reshape(B, N * self.num_modes, -1)
        # for t in range(self.num_recurrent_steps):
        for i in range(0, len(self.cross_attender_propose), 2):
            # Mode&Agent2Scene
            traj_query, x_scene_rel_pos = self.cross_attender_propose[i](
                traj_query,
                scene_feat,
                scene_feat,
                key_padding_mask=x_scene_padding_mask,
                position_bias=x_scene_rel_pos,
            )
            traj_query = (traj_query.reshape(B, N, self.num_modes, D).permute(
                0, 2, 1, 3).reshape(B * self.num_modes, N, D))

            mask = (data["x_key_padding_mask"].unsqueeze(1).repeat(
                1, self.num_modes, 1, 1).reshape(B * self.num_modes, N))

            traj_query, x_rel_pos = self.cross_attender_propose[i + 1](
                traj_query, key_padding_mask=mask, position_bias=x_rel_pos)
            traj_query = (traj_query.reshape(B, self.num_modes, N, D).permute(
                0, 2, 1, 3).reshape(B, N * self.num_modes, D))
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
        loc_propose_pos = self.to_loc_propose_pos(traj_query)
        # traj_query = traj_query.reshape(B, N * self.num_modes, D)
        loc_propose_pos = torch.cumsum(
            loc_propose_pos.reshape(-1, self.num_modes, self.future_steps, 2),
            dim=-2,
        )
        if self.use_refine:
            traj_query = self.y_emb(loc_propose_pos.detach().reshape(
                B * N * self.num_modes, self.future_steps,
                2)).reshape(B, N, self.num_modes, self.future_steps, D)
            B, _, _, T, D = traj_query.shape
            traj_query = traj_query.reshape(B * N * self.num_modes, T,
                                            D).transpose(0, 1)
            traj_query = self.traj_emb(
                traj_query,
                self.traj_emb_h0.unsqueeze(1).repeat(1, traj_query.size(1),
                                                     1))[1].squeeze(0).reshape(
                                                         B, N * self.num_modes,
                                                         D)
            for i in range(0, len(self.cross_attender_refine), 2):
                traj_query, x_scene_rel_pos = self.cross_attender_refine[i](
                    traj_query,
                    scene_feat,
                    scene_feat,
                    key_padding_mask=x_scene_padding_mask,
                    position_bias=x_scene_rel_pos,
                )

                traj_query = (traj_query.reshape(
                    B, N, self.num_modes,
                    D).permute(0, 2, 1, 3).reshape(B * self.num_modes, N, D))

                mask = (data["x_key_padding_mask"].unsqueeze(1).repeat(
                    1, self.num_modes, 1, 1).reshape(B * self.num_modes, N))

                traj_query, x_rel_pos = self.cross_attender_refine[i + 1](
                    traj_query, key_padding_mask=mask, position_bias=x_rel_pos)
                traj_query = (traj_query.reshape(
                    B, self.num_modes, N,
                    D).permute(0, 2, 1, 3).reshape(B, N * self.num_modes, D))

            traj_query = traj_query.reshape(B * N, self.num_modes, D)
            traj_query = traj_query[~agent_padding_mask]
            traj_query = self.mode2mode_refine(traj_query)
            traj_query_tmp = torch.zeros(B * N,
                                         self.num_modes,
                                         D,
                                         device=traj_query.device)
            traj_query_tmp[~agent_padding_mask] = traj_query.clone()
            traj_query = traj_query_tmp.view(B, N, self.num_modes, D)

            # decoder trajectory
            # traj_query = traj_query.reshape(B, N, self.num_modes, -1)
            loc_refine_pos = self.to_loc_refine_pos(traj_query[:, :N]).reshape(
                -1, self.num_modes, self.future_steps, 2)
            loc_refine_pos = loc_refine_pos + loc_propose_pos.detach().reshape(
                -1, self.num_modes, self.future_steps, 2)

            y_hat = loc_refine_pos.reshape(B, N, self.num_modes,
                                           self.future_steps, 2)
            traj_propose = loc_propose_pos.reshape(B, N, self.num_modes,
                                                   self.future_steps, 2)
        if not self.use_refine:
            y_hat = loc_propose_pos.reshape(B, N, self.num_modes,
                                            self.future_steps, 2)
        # Scene scoring module using cross attention
        traj_query = (traj_query.reshape(B, N, self.num_modes, -1).permute(
            0, 2, 1, 3).reshape(B * self.num_modes, N, -1))
        traj_mask = ~data["x_scored"].unsqueeze(1).repeat(
            1, self.num_modes, 1, 1).reshape(B * self.num_modes, N)

        scene_query = (self.scene_query.unsqueeze(0).unsqueeze(2).repeat(
            B, 1, 1, 1).reshape(B * self.num_modes, 1, D))
        for blk in self.scene_2_mode:
            scene_query = blk(scene_query,
                              traj_query,
                              traj_query,
                              key_padding_mask=traj_mask)
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


class SceneMTRDecoder(nn.Module):
    """MTR++ Decoder"""

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
        self.future_steps = future_steps
        self.K = 32
        self.num_motion_modes = 6
        self.hidden_dim = hidden_dim
        self.depth = depth

        # obj feature projection
        self.obj_in_proj_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Dense future prediction,
        # obj pos projection
        self.obj_pos_encoder_layer = build_mlps(
            c_in=2,
            mlp_channels=[hidden_dim, hidden_dim, hidden_dim],
            ret_before_act=True,
            without_norm=True)
        self.dense_future_head = build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim, hidden_dim, future_steps * 2],
            ret_before_act=True)
        self.future_trajs_encoder = build_mlps(
            c_in=future_steps * 2,
            mlp_channels=[hidden_dim, hidden_dim, hidden_dim],
            ret_before_act=True,
            without_norm=True)
        self.fuse_future_past = build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim, hidden_dim, hidden_dim],
            ret_before_act=True,
            without_norm=True)

        # intention
        self.intention_points = []
        intention_points_file = "/data/jerome.zhou/prediction_dataset/av2/anchor/"
        for i in range(3):
            with open(
                    os.path.join(
                        intention_points_file,
                        f"train_{TYPE_MAP[i]}_anchor_end_pts_{self.K}.npy"),
                    'rb') as f:
                self.intention_points.append(
                    torch.from_numpy(np.load(f)).float())
        self.intention_query_mlps = build_mlps(
            c_in=hidden_dim,
            mlp_channels=[hidden_dim, hidden_dim],
            ret_before_act=True)

        # Decoder Layer
        self.pos_embed = nn.Linear(2 * hidden_dim, hidden_dim)
        self.obj_mutually_guide_decoder_layers = nn.ModuleList(
            TransformerEncoderLayer(d_model=hidden_dim,
                                    nhead=num_head,
                                    dim_feedforward=hidden_dim * 4,
                                    dropout=dropout,
                                    activation='relu',
                                    normalize_before=False,
                                    use_local_attn=False)
            for _ in range(depth))
        self.obj_transformer_self_attn_decoder_layers = nn.ModuleList(
            TransformerDecoderLayer(d_model=hidden_dim,
                                    nhead=num_head,
                                    dim_feedforward=hidden_dim * 4,
                                    dropout=dropout,
                                    activation='relu',
                                    normalize_before=False,
                                    keep_query_pos=False,
                                    rm_self_attn_decoder=False,
                                    use_local_attn=False)
            for _ in range(depth))
        self.obj_transformer_cross_attn_decoder_layers = nn.ModuleList(
            TransformerDecoderLayer(d_model=hidden_dim,
                                    nhead=num_head,
                                    dim_feedforward=hidden_dim * 4,
                                    dropout=dropout,
                                    activation='relu',
                                    normalize_before=False,
                                    keep_query_pos=False,
                                    rm_self_attn_decoder=True,
                                    use_local_attn=False)
            for _ in range(depth))

        # self.map_transformer_self_attn_decoder_layers = nn.ModuleList(
        #     TransformerDecoderLayer(d_model=hidden_dim,
        #                             nhead=num_head,
        #                             dim_feedforward=hidden_dim * 4,
        #                             dropout=dropout,
        #                             activation='relu',
        #                             normalize_before=False,
        #                             keep_query_pos=False,
        #                             rm_self_attn_decoder=False,
        #                             use_local_attn=False)
        #     for _ in range(depth))
        self.map_transformer_cross_attn_decoder_layers = nn.ModuleList(
            TransformerDecoderLayer(d_model=hidden_dim,
                                    nhead=num_head,
                                    dim_feedforward=hidden_dim * 4,
                                    dropout=dropout,
                                    activation='relu',
                                    normalize_before=False,
                                    keep_query_pos=False,
                                    rm_self_attn_decoder=True,
                                    use_local_attn=False)
            for _ in range(depth))

        # motion head
        temp_layer = build_mlps(
            c_in=self.hidden_dim * 2,
            mlp_channels=[self.hidden_dim, self.hidden_dim],
            ret_before_act=True)
        self.query_feature_fusion_layers = nn.ModuleList(
            [copy.deepcopy(temp_layer) for _ in range(self.depth)])
        motion_reg_head = build_mlps(
            c_in=hidden_dim,
            mlp_channels=[hidden_dim, hidden_dim, self.future_steps * 2],
            ret_before_act=True)
        motion_cls_head = build_mlps(c_in=hidden_dim,
                                     mlp_channels=[hidden_dim, hidden_dim, 1],
                                     ret_before_act=True)
        self.motion_reg_heads = nn.ModuleList(
            [copy.deepcopy(motion_reg_head) for _ in range(depth)])
        self.motion_cls_heads = nn.ModuleList(
            [copy.deepcopy(motion_cls_head) for _ in range(depth)])

    def forward(self, data: dict, scene_feat: List[torch.Tensor],
                agent_pos_emb: torch.Tensor):
        # set of input.
        obj_feature, map_feature = scene_feat[0], scene_feat[1]
        B, N, D = obj_feature.shape
        _, M, _ = map_feature.shape

        obj_padding_mask = data["x_key_padding_mask"]
        map_padding_mask = data["lane_key_padding_mask"]

        obj_pos = data["x_centers"]  # [B, N, 2]
        object_types = data['x_attr'][..., -1]  # [B, N]
        objects_type_mask = object_types > 2
        object_types_valid = object_types[
            ~objects_type_mask]  # [B*N_type_valid]

        # input projection
        obj_feature_valid = self.obj_in_proj_layer(
            obj_feature[~obj_padding_mask])
        obj_feature = obj_feature.new_zeros(B, N, obj_feature_valid.shape[-1])
        obj_feature[~obj_padding_mask] = obj_feature_valid

        # dense future prediction
        obj_feature, pred_dense_future_trajs = self.dense_future_pred_module(
            obj_feature, obj_padding_mask, obj_pos)

        # transformer decoder
        pred_list, intention_points = self.transformer_decoder(
            data=data,
            object_types_valid=object_types_valid,
            objects_type_mask=objects_type_mask,
            obj_feature=obj_feature,
            obj_mask=obj_padding_mask,
            map_feature=map_feature,
            map_mask=map_padding_mask,
            map_pos=data["lane_centers"])

        # if not self.training:
        #     pred_scores, pred_trajs = self.generate_final_prediction(
        #         pred_list=pred_list)
        #     return {"y_hat": pred_trajs, "pi": pred_scores.unsqueeze(-1)}
        # else:
        return pred_list, intention_points, pred_dense_future_trajs

    def generate_final_prediction(self, pred_list: List[torch.Tensor]):
        pred_scores, pred_trajs = pred_list[-1]
        pred_scores = torch.softmax(pred_scores,
                                    dim=-1)  # (num_center_objects, num_query)

        B, N, num_query, num_future_timestamps, num_feat = pred_trajs.shape
        pred_trajs = pred_trajs.flatten(start_dim=0, end_dim=1)
        pred_scores = pred_scores.reshape(B * N, num_query)

        if self.num_motion_modes != num_query:
            assert num_query > self.num_motion_modes
            pred_trajs_final, pred_scores_final, selected_idxs = batch_nms(
                pred_trajs=pred_trajs,
                pred_scores=pred_scores,
                dist_thresh=2.5,
                num_ret_modes=self.num_motion_modes)
        else:
            pred_trajs_final = pred_trajs
            pred_scores_final = pred_scores

        pred_trajs_final = pred_trajs_final.reshape(B, N,
                                                    pred_trajs_final.shape[1],
                                                    num_future_timestamps,
                                                    num_feat)
        pred_scores_final = pred_scores_final.reshape(
            B, N, pred_scores_final.shape[1])
        return pred_scores_final, pred_trajs_final

    def dense_future_pred_module(self, obj_feature, obj_padding_mask, obj_pos):
        """This module used to dense future prediction, like forecaset-mae"""
        B, N, D = obj_feature.shape
        obj_pos_valid = obj_pos[~obj_padding_mask][..., :2]
        obj_feature_valid = obj_feature[~obj_padding_mask]
        obj_pos_feature_valid = self.obj_pos_encoder_layer(obj_pos_valid)

        obj_future_fuse_pos = torch.cat(
            [obj_feature_valid, obj_pos_feature_valid], dim=-1)
        pred_dense_future_valid = self.dense_future_head(obj_future_fuse_pos)
        pred_dense_future_valid = pred_dense_future_valid.view(
            pred_dense_future_valid.shape[0], self.future_steps, 2)

        temp_center = pred_dense_future_valid[:, :,
                                              0:2] + obj_pos_valid[:, None,
                                                                   0:2]
        pred_dense_future_valid = temp_center

        # future feature encoding and fuse to past obj_feature
        # encoding timestamp
        obj_future_input_valid = pred_dense_future_valid[:, :, [0, 1]].flatten(
            start_dim=1, end_dim=2)  # (num_valid_objects, C)
        obj_future_feature_valid = self.future_trajs_encoder(
            obj_future_input_valid)

        obj_full_trajs_feature = torch.cat(
            (obj_feature_valid, obj_future_feature_valid), dim=-1)
        obj_feature_valid = self.fuse_future_past(obj_full_trajs_feature)

        ret_obj_feature = torch.zeros_like(obj_feature)
        ret_obj_feature[~obj_padding_mask] = obj_feature_valid

        ret_pred_dense_future_trajs = obj_feature.new_zeros(
            B, N, self.future_steps, 2)
        ret_pred_dense_future_trajs[
            ~obj_padding_mask] = pred_dense_future_valid

        return ret_obj_feature, ret_pred_dense_future_trajs

    def transformer_decoder(self, data, object_types_valid, objects_type_mask,
                            obj_feature: torch.Tensor, obj_mask, map_feature,
                            map_mask, map_pos):
        B, N, D = obj_feature.shape
        _, M = map_mask.shape

        intention_query_valid, intention_points_valid = self.get_motion_query(
            object_types_valid)
        intention_query = obj_feature.new_zeros(
            self.K, B, N, intention_query_valid.shape[-1])
        intention_query[:,
                        ~objects_type_mask] = intention_query_valid  # [64, B, N, 128]
        intention_points = obj_feature.new_zeros(
            self.K, B, N, intention_points_valid.shape[-1])
        intention_points[:,
                         ~objects_type_mask] = intention_points_valid  # [64, B, N, 2]

        intention_query = intention_query.permute(1, 2, 0,
                                                  3).reshape(B, N * self.K, D)

        # trans intention point to global
        intention_points = intention_points.permute(1, 2, 0, 3).reshape(
            B, N * self.K, 2)
        ctr_pt_angle = data['x_theta']  # [B, N]
        cos_matrix1 = ctr_pt_angle.cos()
        sin_matrix1 = ctr_pt_angle.sin()
        rotation_matrix1 = torch.stack(
            [cos_matrix1, sin_matrix1, -sin_matrix1, cos_matrix1], dim=-1)
        rotation_matrix1 = rotation_matrix1.view(
            B, N, 2, 2)[:, :,
                        None, :, :].repeat(1, 1, self.K, 1,
                                           1).reshape(B, N * self.K, 2, 2)
        intention_points_glb = torch.matmul(
            intention_points.unsqueeze(-2), rotation_matrix1).squeeze(
            ) + data['x_centers'][:, :, None, :].repeat(
                1, 1, self.K, 1).reshape(B, N * self.K, 2)
        # calculate intention_points relative pos and angle [B, N*64, N*64, 3]
        intention_rel = calculate_relative_positions_angles(
            intention_points_glb,
            ctr_pt_angle[..., None].repeat(1, 1,
                                           self.K).reshape(B, N * self.K),
            intention_points_glb,
            ctr_pt_angle[..., None].repeat(1, 1, self.K).reshape(
                B, N * self.K)).reshape(B * N * self.K, N * self.K, 3)

        rel_padding_mask, relative_pose_embed, self_pos_embed = gen_relative_input(
            scene_relative_pose=intention_rel,
            scene_padding_mask=obj_mask[:, :, None].repeat(1, 1,
                                                           self.K).reshape(
                                                               B, N * self.K),
            pos_embed=self.pos_embed,
            hidden_dim=D)

        query_content = torch.zeros_like(intention_query)  # [B, N*K, D]

        dynamic_query_center = intention_points  # [B, N*K, 2]
        obj_invalid_mask = obj_mask.reshape(B * N)

        obj_relative_pose = data["obj_relative_pose"].reshape(
            B * N, N, -1)  # [B, N, N, 2]
        obj_rel_padding_mask, obj_relative_pose_embed, obj_self_pos_embed = gen_relative_input(
            scene_relative_pose=obj_relative_pose,
            scene_padding_mask=obj_mask,
            pos_embed=self.pos_embed,
            hidden_dim=D)
        obj_rel_padding_mask = obj_rel_padding_mask.reshape(B, N, N)

        obj_map_relative_pose = data["obj_map_relative_pose"].reshape(
            B * N, M, -1)  # [B, N, M, 2]
        # relative pos mask
        padding_mask_1 = map_mask[:, None, :]
        padding_mask_2 = obj_mask[:, :, None]
        padding_mask = padding_mask_1 & padding_mask_2
        obj_map_rel_padding_mask = padding_mask.reshape(B, N, M)

        obj_map_relative_angle = torch.stack([
            obj_map_relative_pose[..., -1].sin(),
            obj_map_relative_pose[..., -1].cos()
        ],
                                             dim=-1)
        obj_map_relative_pose = torch.cat(
            [obj_map_relative_pose[..., :2], obj_map_relative_angle],
            dim=-1)  # [B, N, M, 4]
        obj_map_relative_pose_embed = gen_sineembed_for_position(
            obj_map_relative_pose, hidden_dim=D)  # [B * N, M, 2 * D]

        obj_map_relative_pose_embed = self.pos_embed(
            obj_map_relative_pose_embed).reshape(B, N, M, D)

        pred_list = []
        for layer_idx in range(self.depth):
            # mutually-guided module
            src = query_content + intention_query  # [B, N*K, D]
            query_content: torch.Tensor = self.obj_mutually_guide_decoder_layers[
                layer_idx](src=src,
                           src_key_padding_mask=rel_padding_mask,
                           pos=[self_pos_embed, relative_pose_embed])
            query_content = query_content.reshape(B * N, self.K, D)
            query_content_valid = query_content[~obj_invalid_mask]
            intention_query = intention_query.reshape(B * N, self.K, D)
            intention_query_valid = intention_query[~obj_invalid_mask]

            # intention query self attn
            query_content_valid = self.obj_transformer_self_attn_decoder_layers[
                layer_idx](tgt=query_content_valid,
                           query_pos=intention_query_valid,
                           memory=None)

            query_content = query_content.new_zeros(B * N, self.K, D)
            query_content[~obj_invalid_mask] = query_content_valid
            query_content = query_content.reshape(B, N * self.K, D)
            intention_query = intention_query.reshape(B, N * self.K, D)

            # query object feature [B, NK, D]
            obj_query_feature = self.apply_cross_attention(
                kv_feature=obj_feature[:,
                                       None, :, :].repeat(1, N * self.K, 1, 1),
                kv_mask=obj_rel_padding_mask[:, :, None, :].repeat(
                    1, 1, self.K, 1).reshape(B, N * self.K, N),
                kv_pos=obj_relative_pose_embed[:, :, None, :, :].repeat(
                    1, 1, self.K, 1, 1).reshape(B, N * self.K, N, D),
                query_content=query_content,
                query_embed=intention_query,
                attention_layer=self.
                obj_transformer_cross_attn_decoder_layers[layer_idx],
                dynamic_query_center=dynamic_query_center,
                layer_idx=layer_idx,
                num_k=self.K)  # [B, NK, D]

            # query map feature [B, NK, D]
            map_query_feature = self.apply_cross_attention(
                kv_feature=map_feature[:,
                                       None, :, :].repeat(1, N * self.K, 1, 1),
                kv_mask=obj_map_rel_padding_mask[:, :, None, :].repeat(
                    1, 1, self.K, 1).reshape(B, N * self.K, M),
                kv_pos=obj_map_relative_pose_embed[:, :, None, :, :].repeat(
                    1, 1, self.K, 1, 1).reshape(B, N * self.K, M, D),
                query_content=query_content,
                query_embed=intention_query,
                attention_layer=self.
                map_transformer_cross_attn_decoder_layers[layer_idx],
                layer_idx=layer_idx,
                dynamic_query_center=dynamic_query_center,
                use_local_attn=False,
                num_k=self.K)

            query_feature = torch.cat([obj_query_feature, map_query_feature],
                                      dim=-1)  # B, NK, 2D
            query_content = self.query_feature_fusion_layers[layer_idx](
                query_feature.flatten(start_dim=0,
                                      end_dim=1)).view(B, N * self.K, -1)

            # motion prediction
            query_content_t = query_content.contiguous().view(
                B * N * self.K, -1)
            pred_scores = self.motion_cls_heads[layer_idx](
                query_content_t).view(B, N, self.K)

            pred_trajs = self.motion_reg_heads[layer_idx](
                query_content_t).view(B, N, self.K, self.future_steps, 2)

            pred_list.append([pred_scores, pred_trajs])

            # update
            dynamic_query_center = pred_trajs[:, :, :, -1,
                                              0:2].contiguous().reshape(
                                                  B, N * self.K,
                                                  2)  # B,N,self.K,2

        assert len(pred_list) == self.depth
        return pred_list, intention_points.reshape(B, N, self.K, 2)

    def get_motion_query(self, center_objects_type):
        num_center_objects = len(center_objects_type)
        intention_points = torch.stack([
            self.intention_points[center_objects_type[obj_idx]].to(center_objects_type.device)
            for obj_idx in range(num_center_objects)
        ],
                                       dim=0)
        intention_points = intention_points.permute(
            1, 0, 2)  # (num_query, num_center_objects, 2)

        intention_query = gen_sineembed_for_position(
            intention_points, hidden_dim=self.hidden_dim)
        intention_query = self.intention_query_mlps(
            intention_query.view(-1, self.hidden_dim)).view(
                -1, num_center_objects,
                self.hidden_dim)  # (num_query, num_center_objects, C)

        return intention_query, intention_points

    def apply_cross_attention(self,
                              kv_feature,
                              kv_mask,
                              kv_pos,
                              query_content,
                              query_embed,
                              attention_layer,
                              dynamic_query_center=None,
                              layer_idx=0,
                              use_local_attn=False,
                              query_index_pair=None,
                              query_content_pre_mlp=None,
                              query_embed_pre_mlp=None,
                              num_k=32):
        """
        Args:
            kv_feature (B, N, C):
            kv_mask (B, N):
            kv_pos (B, N, 3):
            query_tgt (M, B, C):
            query_embed (M, B, C):
            dynamic_query_center (M, B, 2): . Defaults to None.
            attention_layer (layer):

            query_index_pair (B, M, K)

        Returns:
            attended_features: (B, M, C)
            attn_weights:
        """
        if query_content_pre_mlp is not None:
            query_content = query_content_pre_mlp(query_content)
        if query_embed_pre_mlp is not None:
            query_embed = query_embed_pre_mlp(query_embed)

        batch_size, num_q, d_model = query_content.shape
        # [B, N*A, D]
        searching_query = gen_sineembed_for_position(dynamic_query_center,
                                                     hidden_dim=d_model)

        query_feature = attention_layer(tgt=query_content,
                                        query_pos=query_embed,
                                        query_sine_embed=searching_query,
                                        memory=kv_feature,
                                        memory_key_padding_mask=kv_mask,
                                        pos=kv_pos,
                                        is_first=(layer_idx == 0),
                                        num_k=num_k)  # (M, B, C)

        return query_feature
