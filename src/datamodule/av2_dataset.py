from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch.nn.functional as F
import tqdm
import av2.geometry.interpolate as interp_utils

from .av2_extractor import Av2Extractor


class Av2Dataset(Dataset):

    def __init__(
        self,
        data_root: Path,
        cached_split: str = None,
        extractor: Av2Extractor = None,
    ):
        super(Av2Dataset, self).__init__()

        if cached_split is not None:
            self.data_folder = Path(data_root) / cached_split
            self.file_list = sorted(list(self.data_folder.glob("*.pt")))
            self.load = True
        elif extractor is not None:
            self.extractor = extractor
            self.data_folder = Path(data_root)
            print(f"Extracting data from {self.data_folder}")
            self.file_list = list(self.data_folder.rglob("*.parquet"))
            self.load = False
        else:
            raise ValueError(
                "Either cached_split or extractor must be specified")
        # if cached_split == "train":
        #     import pandas as pd
        #     self.file_list = pd.read_csv("./notebook/res.csv").file_path.values.tolist()
        print(
            f"data root: {data_root}/{cached_split}, total number of files: {len(self.file_list)}"
        )

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        if self.load:
            data = torch.load(self.file_list[index])
        else:
            data = self.extractor.get_data(self.file_list[index])

        return data


def collate_fn(batch):
    data = {}

    for key in [
            "x",
            "x_attr",
            "x_positions",
            "x_centers",
            "x_angles",
            "x_velocity",
            "x_velocity_diff",
            "lane_positions",
            "lane_centers",
            "lane_angles",
            "lane_attr",
            "is_intersections",
    ]:
        data[key] = pad_sequence([b[key] for b in batch], batch_first=True)

    for key in [
            "x_theta",
            "lane_positions_orignal",
    ]:
        data[key] = pad_sequence([b[key] for b in batch], batch_first=True)

    # used in multi-agent prediction
    if "x_scored" in batch[0]:
        data["x_scored"] = pad_sequence([b["x_scored"] for b in batch],
                                        batch_first=True)

    if batch[0]["y"] is not None:
        data["y"] = pad_sequence([b["y"] for b in batch], batch_first=True)

    for key in ["x_padding_mask", "lane_padding_mask"]:
        data[key] = pad_sequence([b[key] for b in batch],
                                 batch_first=True,
                                 padding_value=True)

    data["x_key_padding_mask"] = data["x_padding_mask"].all(-1)
    data["lane_key_padding_mask"] = data["lane_padding_mask"].all(-1)
    data["num_actors"] = (~data["x_key_padding_mask"]).sum(-1)
    data["num_lanes"] = (~data["lane_key_padding_mask"]).sum(-1)

    data["scenario_id"] = [b["scenario_id"] for b in batch]
    data["track_id"] = [b["track_id"] for b in batch]

    data["origin"] = torch.cat([b["origin"] for b in batch], dim=0)
    data["theta"] = torch.cat([b["theta"] for b in batch])

    scene_ctrs = torch.cat([data['x_centers'], data["lane_centers"]], dim=1)
    x_vets = torch.stack([data['x_theta'].cos(),
                              data['x_theta'].sin()], dim=-1)
    lane_vets = torch.stack([data['lane_angles'].cos(),
                             data["lane_angles"].sin()],
                             dim=-1)
    scene_vets = torch.cat([x_vets, lane_vets], dim=1)

    d_pos = (scene_ctrs.unsqueeze(1) - scene_ctrs.unsqueeze(2)).norm(dim=-1)
    d_pos = d_pos * 2 / 150  # scale [0, radius] to [0, 2]
    pos_rpe = d_pos.unsqueeze(-1)

    ang2ang = _get_rel_pe(scene_vets.unsqueeze(1), scene_vets.unsqueeze(2))
    v_pos = scene_ctrs.unsqueeze(1) - scene_ctrs.unsqueeze(2)
    ang2vec = _get_rel_pe(scene_vets.unsqueeze(1), v_pos)

    data['rpe'] = torch.cat([ang2ang, ang2vec, pos_rpe], dim=-1)
    # data["lane_trans"], data["lane_trans_padding_mask"], data[
    #     "lane_centers"], data["lane_centers_padding_mask"], data[
    #         "lane_angles"] = resample(data)

    return data

def _get_rel_pe(v1: torch.Tensor, v2: torch.Tensor):
    # 扩展 heading 的维度以便进行广播
    # v1 = heading.unsqueeze(1)  # shape: [B, 1, M, 2]
    # v2 = heading.unsqueeze(2)  # shape: [B, M, 1, 2]

    # 计算 v1 和 v2 的范数
    v1_norm = v1.norm(dim=-1)  # shape: [B, 1, M, 1]
    v2_norm = v2.norm(dim=-1)  # shape: [B, M, 1, 1]

    # 计算 v1 和 v2 的 cos 和 sin 值
    cos_val = (v1 * v2).sum(dim=-1) / (v1_norm * v2_norm + 1e-10)  # shape: [B, M, M]
    sin_val = (v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]) / (v1_norm * v2_norm + 1e-10)  # shape: [B, M, M]

    # 将 cos_val 和 sin_val 堆叠起来得到最终的结果
    return torch.stack([cos_val, sin_val], dim=-1)  # shape: [B, M, M, 2]


def sept_collate_fn(batch):
    data = {}

    for key in [
            "x",
            "x_trans",
            "x_attr",
            "x_positions",
            "x_centers",
            "x_angles",
            "x_velocity",
            "x_velocity_diff",
            "lane_positions",
            "lane_centers",
            "lane_angles",
            "lane_attr",
            "is_intersections",
    ]:
        data[key] = pad_sequence([b[key] for b in batch], batch_first=True)

    # used in multi-agent prediction
    if "x_scored" in batch[0]:
        data["x_scored"] = pad_sequence([b["x_scored"] for b in batch],
                                        batch_first=True)

    if batch[0]["y"] is not None:
        data["y"] = pad_sequence([b["y"] for b in batch], batch_first=True)

    for key in ["x_padding_mask", "lane_padding_mask"]:
        data[key] = pad_sequence([b[key] for b in batch],
                                 batch_first=True,
                                 padding_value=True)

    data["x_key_padding_mask"] = data["x_padding_mask"].all(-1)
    data["lane_key_padding_mask"] = data["lane_padding_mask"].all(-1)
    data["num_actors"] = (~data["x_key_padding_mask"]).sum(-1)
    data["num_lanes"] = (~data["lane_key_padding_mask"]).sum(-1)

    data["scenario_id"] = [b["scenario_id"] for b in batch]
    data["track_id"] = [b["track_id"] for b in batch]

    data["origin"] = torch.cat([b["origin"] for b in batch], dim=0)
    data["theta"] = torch.cat([b["theta"] for b in batch])

    return data


def sept_collate_fn_lane_candidate(batch):
    """Purning lane.

    Args:
        batch (_type_): _description_

    Returns:
        _type_: _description_
    """
    data = {}

    for key in [
            "x",
            "x_trans",
            "x_attr",
            "x_positions",
            "x_centers",
            "x_angles",
            "x_velocity",
            "x_velocity_diff",
            # "lane_positions",
            "lane_centers",
            "lane_angles",
            "lane_attr",
            "is_intersections",
            "lane_candidate"
    ]:
        data[key] = pad_sequence([b[key] for b in batch], batch_first=True)

    for key in [
            # "lane_positions",
            "lane_centers",
            "lane_angles",
            "lane_attr",
            "is_intersections",
    ]:
        data[key + '_candidate'] = pad_sequence(
            [b[key][b['lane_candidate'].bool()] for b in batch],
            batch_first=True)
    # all trajectories
    batch_lanes, batch_lanes_candidate = [], []
    for b in batch:
        lane_segment = b['lane_positions']
        lane_candidate = b['lane_candidate']
        lane_angles = b['lane_angles']
        lane_attr = b['lane_attr']
        lanes, lanes_candidate = [], []
        for i in range(len(lane_segment)):
            lane_pts = lane_segment[i][lane_segment[i].all(-1)]
            if len(lane_pts) >= 1:
                if len(lane_pts) == 1:
                    vec_pt = torch.cat([
                        lane_pts,
                        lane_pts,
                        lane_angles[i].unsqueeze(-1)[None, :].repeat(1, 1),
                        lane_attr[i][None, :].repeat(1, 1),
                    ],
                                       dim=-1)
                else:
                    vec_pt = torch.cat([
                        lane_pts[:-1],
                        lane_pts[1:],
                        lane_angles[i].unsqueeze(-1)[None, :].repeat(
                            len(lane_pts) - 1, 1),
                        lane_attr[i][None, :].repeat(len(lane_pts) - 1, 1),
                    ],
                                       dim=-1)
                lanes.append(vec_pt)
                if lane_candidate[i]:
                    lanes_candidate.append(vec_pt)
            # else:
            #     print(i,lane_segment[i])
        lanes = torch.cat(lanes, dim=0)
        lanes_candidate = torch.cat(lanes_candidate, dim=0)
        batch_lanes.append(lanes)
        batch_lanes_candidate.append(lanes_candidate)

    data['lane_positions'] = pad_sequence([b for b in batch_lanes],
                                          batch_first=True)
    data["lane_padding_mask"] = (data['lane_positions'] == 0).all(
        dim=-1)  # pad is True
    data['lane_positions_candidate'] = pad_sequence(
        [b for b in batch_lanes_candidate], batch_first=True)
    data["lane_padding_mask_candidate"] = (
        data['lane_positions_candidate'] == 0).all(dim=-1)  # pad is True
    # lane_positions = [b['lane_positions'] for b in batch]
    # # 找到n和m的最大值
    # max_n = max(tensor.size(0) for tensor in lane_positions)
    # max_m = max(tensor.size(1) for tensor in lane_positions)
    # # 使用F.pad函数将每个tensor填充到相同的维度
    # for i in range(len(lane_positions)):
    #     lane_positions[i] = F.pad(lane_positions[i],
    #                               (0, 0, 0, max_m - lane_positions[i].size(1),
    #                                0, max_n - lane_positions[i].size(0)))
    # lane_positions = torch.stack(lane_positions)
    # lane_positions_candidate = [b['lane_positions'][b['lane_candidate'].bool()] for b in batch]
    # # 找到n和m的最大值
    # max_n = max(tensor.size(0) for tensor in lane_positions_candidate)
    # max_m = max(tensor.size(1) for tensor in lane_positions_candidate)
    # # 使用F.pad函数将每个tensor填充到相同的维度
    # for i in range(len(lane_positions_candidate)):
    #     lane_positions_candidate[i] = F.pad(lane_positions_candidate[i],
    #                               (0, 0, 0, max_m - lane_positions_candidate[i].size(1),
    #                                0, max_n - lane_positions_candidate[i].size(0)))
    # lane_positions_candidate = torch.stack(lane_positions_candidate)
    # data['lane_positions'] = lane_positions
    # data['lane_positions_candidate'] = lane_positions_candidate
    # data['lane_padding_mask'] = lane_positions.eq(0)
    # data['lane_padding_mask_candidate'] = lane_positions_candidate.eq(0)

    # used in multi-agent prediction
    if "x_scored" in batch[0]:
        data["x_scored"] = pad_sequence([b["x_scored"] for b in batch],
                                        batch_first=True)

    if batch[0]["y"] is not None:
        data["y"] = pad_sequence([b["y"] for b in batch], batch_first=True)

    for key in ["x_padding_mask"]:
        data[key] = pad_sequence([b[key] for b in batch],
                                 batch_first=True,
                                 padding_value=True)
    # for key in ["lane_padding_mask"]:
    #     data[key + '_candidate'] = pad_sequence(
    #         [b[key][b['lane_candidate'].bool()] for b in batch],
    #         batch_first=True,
    #         padding_value=True)

    data["x_key_padding_mask"] = data["x_padding_mask"].all(-1)
    # data["lane_key_padding_mask"] = data["lane_padding_mask"].all(-1)
    # data["lane_key_padding_mask_candidate"] = data[
    #     "lane_padding_mask_candidate"].all(-1)
    data["num_actors"] = (~data["x_key_padding_mask"]).sum(-1)
    # data["num_lanes"] = (~data["lane_key_padding_mask"]).sum(-1)
    # data["num_lanes_candidate"] = (
    #     ~data["lane_key_padding_mask_candidate"]).sum(-1)

    data["scenario_id"] = [b["scenario_id"] for b in batch]
    data["track_id"] = [b["track_id"] for b in batch]

    data["origin"] = torch.cat([b["origin"] for b in batch], dim=0)
    data["theta"] = torch.cat([b["theta"] for b in batch])

    return data


def resample(data: dict, distance=5.0):
    # get lane related feature
    tensor = data['lane_positions']
    lane_mask = data["lane_padding_mask"]
    attr = data['lane_attr']
    angle = data['lane_angles']
    B, M, L, _ = tensor.shape
    output, lane_center, lane_angle, outputs, lane_centers, lane_angles = [], [], [], [], [], []
    for b in range(B):
        for m in range(M):
            lane = tensor[b, m][~lane_mask[b, m]]  # shape: [L, 2]
            diff = lane[1:] - lane[:-1]  # shape: [L-1, 2]
            norm = torch.norm(diff, dim=-1)  # shape: [L-1]
            norm = torch.cat([torch.zeros(1), norm])  # shape: [L]
            cum_norm = torch.cumsum(norm, dim=0)  # shape: [L]
            max_cum_norm = cum_norm[-1]
            N = int(torch.ceil(max_cum_norm / distance))
            if N == 1:
                N = 2
            if N == 0:
                continue
            lane_resampled = interp_utils.interp_arc(N, points=lane)
            lane_vec = lane_resampled[:-1] - lane_resampled[1:]
            output.append(
                torch.cat([
                    lane_vec, attr[b, m].unsqueeze(0).repeat(
                        lane_vec.shape[0], 1), angle[b, m].unsqueeze(0).repeat(
                            lane_vec.shape[0], 1)
                ],
                          dim=-1))
            lane_center.append(lane_resampled[:-1])
            lane_angle.append(angle[b, m].unsqueeze(0).repeat(
                lane_vec.shape[0], 1))
        outputs.append(torch.cat(output, dim=0))  ## [M*l, 6]
        lane_centers.append(torch.cat(lane_center, dim=0))
        lane_angles.append(torch.cat(lane_angle, dim=0))
    lane_positions = pad_sequence(outputs,
                                  batch_first=True,
                                  padding_value=torch.nan)
    lane_centers = pad_sequence(lane_centers, batch_first=True)
    lane_positions_padding_mask = lane_positions.isnan()
    lane_centers_padding_mask = lane_centers.ne(0).bool()
    lane_angles = pad_sequence(lane_angles, batch_first=True)
    return lane_positions, lane_positions_padding_mask, lane_centers, lane_centers_padding_mask, lane_angles.squeeze(
        -1)
