import traceback
from pathlib import Path
from typing import List

import av2.geometry.interpolate as interp_utils
import numpy as np
import torch
from av2.map.map_api import ArgoverseStaticMap

from .av2_data_utils import (
    OBJECT_TYPE_MAP,
    OBJECT_TYPE_MAP_COMBINED,
    load_av2_df,
    LaneTypeMap,
)


class Av2ExtractorMultiAgentNorm:

    def __init__(
        self,
        radius: float = 150,
        save_path: Path = None,
        mode: str = "train",
        ignore_type: List[int] = [5, 6, 7, 8, 9],
        remove_outlier_actors: bool = True,
    ) -> None:
        self.save_path = save_path
        self.mode = mode
        self.radius = radius
        self.remove_outlier_actors = remove_outlier_actors
        self.ignore_type = ignore_type

    def save(self, file: Path):
        assert self.save_path is not None

        try:
            data = self.get_data(file)
        except Exception:
            print(traceback.format_exc())
            print("found error while extracting data from {}".format(file))
        save_file = self.save_path / (file.stem + ".pt")
        torch.save(data, save_file)

    def get_data(self, file: Path):
        return self.process(file)

    def process(self, raw_path: str, agent_id: str = None):
        df, am, scenario_id = load_av2_df(raw_path)
        city = df.city.values[0]
        agent_id = "AV"
        # get AV pose
        local_df = df[df["track_id"] == agent_id].iloc
        origin = torch.tensor(
            [local_df[49]["position_x"], local_df[49]["position_y"]],
            dtype=torch.float)
        # heading is local coord relative to global coord
        theta = torch.tensor([local_df[49]["heading"]], dtype=torch.float)
        # rot matrix is local to global
        rotate_mat = torch.tensor([
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)],
        ], )

        timestamps = list(np.sort(df["timestep"].unique()))
        cur_df = df[df["timestep"] == timestamps[49]]
        actor_ids = list(cur_df["track_id"].unique())
        object_category = torch.from_numpy(
            cur_df["object_category"].values).float()
        cur_pos = torch.from_numpy(cur_df[["position_x", "position_y"
                                           ]].values).float()  # 预测帧各个障碍物的位置

        scored_agents_mask = object_category > 1.5  # 可以被预测的障碍物 mask
        out_of_range = np.linalg.norm(
            cur_pos - origin,
            axis=1) > self.radius  # check agent is out of 150m relative AV
        out_of_range[scored_agents_mask] = False  # keep all scored agents

        actor_ids = [
            aid for i, aid in enumerate(actor_ids) if not out_of_range[i]
        ]  # get in range agent id list
        av_idx = actor_ids.index(agent_id)
        scored_agents_mask = scored_agents_mask[
            ~out_of_range]  # can be predict mask
        num_nodes = len(actor_ids)  # get in range agents num

        df = df[df["track_id"].isin(actor_ids)]  # get all agent frames

        # initialization, 填充每个场景内每个障碍物的属性，trans to AV local coornidate
        x = torch.zeros(num_nodes, 110, 2, dtype=torch.float)
        x_attr = torch.zeros(num_nodes, 3, dtype=torch.uint8)
        x_heading = torch.zeros(num_nodes, 110, dtype=torch.float)
        x_velocity = torch.zeros(num_nodes, 110, dtype=torch.float)
        x_track_horizon = torch.zeros(num_nodes, dtype=torch.int)
        padding_mask = torch.ones(num_nodes, 110, dtype=torch.bool)

        for actor_id, actor_df in df.groupby("track_id"):
            node_idx = actor_ids.index(actor_id)  # 第几个agent
            node_steps = [timestamps.index(ts)
                          for ts in actor_df["timestep"]]  # 该agent的时间戳
            object_type = OBJECT_TYPE_MAP[actor_df["object_type"].values[0]]
            x_attr[node_idx, 0] = object_type
            x_attr[node_idx,
                   1] = actor_df["object_category"].values[0]  # scored
            x_attr[node_idx, 2] = OBJECT_TYPE_MAP_COMBINED[
                actor_df["object_type"].values[0]]
            x_track_horizon[node_idx] = node_steps[-1] - node_steps[0]  # 存在时长

            padding_mask[node_idx, node_steps] = False  # 存在值， mask给False
            if padding_mask[node_idx, 49] or object_type in self.ignore_type:
                padding_mask[node_idx, 50:] = True
            # 如果预测帧没有数据且agent要被忽略，未来帧直接mask
            # global coordinate
            pos_xy = torch.from_numpy(
                np.stack(
                    [
                        actor_df["position_x"].values,
                        actor_df["position_y"].values
                    ],
                    axis=-1,
                )).float()
            heading = torch.from_numpy(actor_df["heading"].values).float()
            velocity = torch.from_numpy(actor_df[["velocity_x", "velocity_y"
                                                  ]].values).float()
            velocity_norm = torch.norm(velocity, dim=1)
            # transform to AV coordinate， global->local
            x[node_idx, node_steps, :2] = torch.matmul(pos_xy - origin,
                                                       rotate_mat)
            x_heading[node_idx, node_steps] = (heading - theta + np.pi) % (
                2 * np.pi) - np.pi  # bounding to
            x_velocity[node_idx, node_steps] = velocity_norm
        # 获得AV周围的车道特征
        (
            lane_positions,
            is_intersections,
            lane_ctrs,
            lane_angles,
            lane_attr,
            lane_padding_mask,
        ) = self.get_lane_features(am, origin, origin, rotate_mat, self.radius)
        # 剔除掉距离车道较远的障碍物
        if self.remove_outlier_actors:
            lane_samples = lane_positions[:, ::1, :2].view(-1, 2)
            nearest_dist = torch.cdist(
                x[:, 49, :2],
                lane_samples).min(dim=1).values  # 获取每个agent与最近车道线段的距离
            valid_actor_mask = nearest_dist < 5  # 距离太远的剔除
            valid_actor_mask[0] = True  # always keep av and scored agents
            valid_actor_mask[scored_agents_mask] = True

            x = x[valid_actor_mask]
            x_heading = x_heading[valid_actor_mask]
            x_velocity = x_velocity[valid_actor_mask]
            x_attr = x_attr[valid_actor_mask]
            actor_ids = [
                aid for i, aid in enumerate(actor_ids) if valid_actor_mask[i]
            ]
            scored_agents_mask = scored_agents_mask[valid_actor_mask]
            padding_mask = padding_mask[valid_actor_mask]
            num_nodes = x.shape[0]

        # norm each agent
        x_ctrs = x[:, 49, :2].clone()
        x_theta = x[:49].clone()
        x_positions = x[:, :, :2].clone()
        for agent_i in range(x.shape[0]):
            orig = x[agent_i, 49, :2]
            theta = x_heading[agent_i, 49]
            rot = torch.tensor([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
            x[agent_i] = torch.matmul(x[agent_i] - orig, rot)
            x_heading[agent_i] = (x_heading[agent_i] - theta +
                                  np.pi) % (2 * np.pi) - np.pi

        # norm each lane segment
        lane_ctrs = lane_positions[:, 0, :2].clone()
        lane_positions_orignal = lane_positions.clone()
        lane_thetas = torch.zeros([lane_ctrs.shape[0]])
        for lane_j in range(lane_positions.shape[0]):
            orig = lane_positions[lane_j, 0, :2]
            theta = torch.arctan2(
                lane_positions[lane_j, -1, 1] - lane_positions[lane_j, 0, 1],
                lane_positions[lane_j, -1, 0] - lane_positions[lane_j, 0, 0])
            lane_thetas[lane_j] = (theta + np.pi) % (2 * np.pi) - np.pi
            rot = torch.tensor([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
            lane_positions[lane_j] = torch.matmul(
                lane_positions[lane_j] - orig, rot)

        x_velocity_diff = x_velocity[:, :50].clone()

        # get fut trajectory relative to current frame's position diff
        x[:, 50:] = torch.where(
            (padding_mask[:, 49].unsqueeze(-1)
             | padding_mask[:, 50:]).unsqueeze(-1),
            torch.zeros(num_nodes, 60, 2),
            x[:, 50:] - x[:, 49].unsqueeze(-2),
        )  # if 当前帧或者之后的帧为被mask，填充0，将来的轨迹相对于每个障碍物当前帧的偏差
        # get past trajectory position diff with himselves last frame
        x[:, 1:50] = torch.where(
            (padding_mask[:, :49] | padding_mask[:, 1:50]).unsqueeze(-1),
            torch.zeros(num_nodes, 49, 2),
            x[:, 1:50] - x[:, :49],
        )  # 位移偏差 相对于前一帧
        ## first frame diff is zero.
        x[:, 0] = torch.zeros(num_nodes, 2)

        x_velocity_diff[:, 1:50] = torch.where(
            (padding_mask[:, :49] | padding_mask[:, 1:50]),
            torch.zeros(num_nodes, 49),
            x_velocity_diff[:, 1:50] - x_velocity_diff[:, :49],
        )  # 速度偏差 相对于前一帧
        x_velocity_diff[:, 0] = torch.zeros(num_nodes)

        y = None if self.mode == "test" else x[:, 50:]

        return {
            "x": x[:, :50],  # instance frame
            "y": y,  # instance frame
            "x_attr": x_attr,
            "x_positions": x_positions,  # scene frame
            "x_centers": x_ctrs,  # means distance with AV scene frame
            "x_angles":
            x_heading,  # means heading diff with AV # instance frame
            "x_theta": x_theta,  # scene frame
            "x_velocity":
            x_velocity,  # means absolutely diff with  AV  # scene frame
            "x_velocity_diff":
            x_velocity_diff,  # velocity diff with last frame
            "x_padding_mask": padding_mask,
            "x_scored": scored_agents_mask,
            "lane_positions": lane_positions,  # instance frame
            "lane_positions_orignal": lane_positions_orignal,  # scene frame
            "lane_centers": lane_ctrs,  # scene frame
            "lane_angles": lane_thetas,  # scene frame
            "lane_attr": lane_attr,
            "lane_padding_mask": lane_padding_mask,
            "is_intersections": is_intersections,
            "av_index": torch.tensor(av_idx),
            "origin": origin.view(-1, 2),
            "theta": theta,
            "scenario_id": scenario_id,
            "track_id": actor_ids,
            "city": city,
        }

    @staticmethod
    def get_lane_features(
        am: ArgoverseStaticMap,
        query_pos: torch.Tensor,
        origin: torch.Tensor,
        rotate_mat: torch.Tensor,
        radius: float,
    ):
        lane_segments = am.get_nearby_lane_segments(query_pos.numpy(), radius)

        lane_positions, is_intersections, lane_attrs = [], [], []
        for segment in lane_segments:
            lane_centerline, lane_width = interp_utils.compute_midpoint_line(
                left_ln_boundary=segment.left_lane_boundary.xyz,
                right_ln_boundary=segment.right_lane_boundary.xyz,
                num_interp_pts=20,
            )
            lane_centerline = torch.from_numpy(lane_centerline[:, :2]).float()
            lane_centerline = torch.matmul(lane_centerline - origin,
                                           rotate_mat)
            is_intersection = am.lane_is_in_intersection(segment.id)

            lane_positions.append(lane_centerline)
            is_intersections.append(is_intersection)

            # get lane attrs
            lane_type = LaneTypeMap[segment.lane_type]
            attribute = torch.tensor([lane_type, lane_width, is_intersection],
                                     dtype=torch.float)
            lane_attrs.append(attribute)

        lane_positions = torch.stack(lane_positions)
        lanes_ctr = lane_positions[:, 9:11].mean(dim=1)
        lanes_angle = torch.atan2(
            lane_positions[:, 10, 1] - lane_positions[:, 9, 1],
            lane_positions[:, 10, 0] - lane_positions[:, 9, 0],
        )
        is_intersections = torch.Tensor(is_intersections)
        lane_attrs = torch.stack(lane_attrs, dim=0)

        x_max, x_min = radius, -radius
        y_max, y_min = radius, -radius

        padding_mask = ((lane_positions[:, :, 0] > x_max)
                        | (lane_positions[:, :, 0] < x_min)
                        | (lane_positions[:, :, 1] > y_max)
                        | (lane_positions[:, :, 1] < y_min))

        invalid_mask = padding_mask.all(dim=-1)
        lane_positions = lane_positions[~invalid_mask]
        is_intersections = is_intersections[~invalid_mask]
        lane_attrs = lane_attrs[~invalid_mask]
        lanes_ctr = lanes_ctr[~invalid_mask]
        lanes_angle = lanes_angle[~invalid_mask]
        padding_mask = padding_mask[~invalid_mask]

        lane_positions = torch.where(padding_mask[..., None],
                                     torch.zeros_like(lane_positions),
                                     lane_positions)

        return (
            lane_positions,
            is_intersections,
            lanes_ctr,
            lanes_angle,
            lane_attrs,
            padding_mask,
        )
