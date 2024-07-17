# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Modified by Shaoshuai Shi
# All Rights Reserved

import torch
import math


def gen_relative_input(scene_relative_pose, scene_padding_mask, pos_embed,
                       hidden_dim):
    B, A = scene_padding_mask.shape
    # relative pos mask
    scene_padding_mask_1 = scene_padding_mask[:, None, :]
    scene_padding_mask_2 = scene_padding_mask[:, :, None]
    scene_padding_mask = scene_padding_mask_1 & scene_padding_mask_2
    scene_padding_mask = scene_padding_mask.reshape(B * A, A)

    scene_relative_angle = torch.stack([
        scene_relative_pose[..., -1].sin(), scene_relative_pose[..., -1].cos()
    ],
                                       dim=-1)
    scene_relative_pose = torch.cat(
        [scene_relative_pose[..., :2], scene_relative_angle],
        dim=-1)  # [B, A, A, 4]
    scene_relative_pose_embed = gen_sineembed_for_position(
        scene_relative_pose, hidden_dim=hidden_dim)  # [B * A, A, 2 * D]

    scene_relative_pose_embed = pos_embed(scene_relative_pose_embed).reshape(
        B, A, A, hidden_dim)

    self_pos = scene_relative_pose.new_zeros(B, A, 4)
    self_pos[..., -1] = 1
    self_pos_embed = gen_sineembed_for_position(self_pos,
                                                hidden_dim=hidden_dim)
    self_pos_embed = pos_embed(self_pos_embed)

    return scene_padding_mask, scene_relative_pose_embed, self_pos_embed


def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    """Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR/
    """
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim,
                         dtype=torch.float32,
                         device=pos_tensor.device)
    # dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    dim_t = 10000**(2 * (torch.div(dim_t, 2, rounding_mode='trunc')) /
                    half_hidden_dim)

    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()),
                        dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()),
                        dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)

    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()),
                            dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()),
                            dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(
            pos_tensor.size(-1)))
    return pos


def batch_nms(pred_trajs, pred_scores, dist_thresh, num_ret_modes=6):
    """

    Args:
        pred_trajs (batch_size, num_modes, num_timestamps, 7)
        pred_scores (batch_size, num_modes):
        dist_thresh (float):
        num_ret_modes (int, optional): Defaults to 6.

    Returns:
        ret_trajs (batch_size, num_ret_modes, num_timestamps, 5)
        ret_scores (batch_size, num_ret_modes)
        ret_idxs (batch_size, num_ret_modes)
    """
    batch_size, num_modes, num_timestamps, num_feat_dim = pred_trajs.shape

    sorted_idxs = pred_scores.argsort(dim=-1, descending=True)
    bs_idxs_full = torch.arange(batch_size).type_as(sorted_idxs)[:,
                                                                 None].repeat(
                                                                     1,
                                                                     num_modes)
    sorted_pred_scores = pred_scores[bs_idxs_full, sorted_idxs]
    sorted_pred_trajs = pred_trajs[
        bs_idxs_full,
        sorted_idxs]  # (batch_size, num_modes, num_timestamps, 7)
    sorted_pred_goals = sorted_pred_trajs[:, :,
                                          -1, :]  # (batch_size, num_modes, 7)

    dist = (sorted_pred_goals[:, :, None, 0:2] -
            sorted_pred_goals[:, None, :, 0:2]).norm(dim=-1)
    point_cover_mask = (dist < dist_thresh)

    point_val = sorted_pred_scores.clone()  # (batch_size, N)
    point_val_selected = torch.zeros_like(point_val)  # (batch_size, N)

    ret_idxs = sorted_idxs.new_zeros(batch_size, num_ret_modes).long()
    ret_trajs = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes,
                                            num_timestamps, num_feat_dim)
    ret_scores = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes)
    bs_idxs = torch.arange(batch_size).type_as(ret_idxs)

    for k in range(num_ret_modes):
        cur_idx = point_val.argmax(dim=-1)  # (batch_size)
        ret_idxs[:, k] = cur_idx

        new_cover_mask = point_cover_mask[bs_idxs, cur_idx]  # (batch_size, N)
        point_val = point_val * (~new_cover_mask).float()  # (batch_size, N)
        point_val_selected[bs_idxs, cur_idx] = -1
        point_val += point_val_selected

        ret_trajs[:, k] = sorted_pred_trajs[bs_idxs, cur_idx]
        ret_scores[:, k] = sorted_pred_scores[bs_idxs, cur_idx]

    bs_idxs = torch.arange(batch_size).type_as(sorted_idxs)[:, None].repeat(
        1, num_ret_modes)

    ret_idxs = sorted_idxs[bs_idxs, ret_idxs]
    return ret_trajs, ret_scores, ret_idxs
