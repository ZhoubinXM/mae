import traceback
from pathlib import Path
from typing import List, Tuple

import av2.geometry.interpolate as interp_utils
import numpy as np
import torch
from av2.map.map_api import ArgoverseStaticMap, LaneSegment

from .av2_data_utils import (
    OBJECT_TYPE_MAP,
    OBJECT_TYPE_MAP_COMBINED,
    LaneTypeMap,
    load_av2_df,
)


class Av2Extractor:

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
        # save_dir = Path("/environment-information/jerome.zhou/av2/model-sept") / self.save_path.stem
        save_dir = Path("/data/jerome.zhou/prediction_dataset/av2/model-sept") / self.save_path.stem
        save_file = save_dir / (file.stem + ".pt")
        torch.save(data, save_file)

    def get_data(self, file: Path):
        return self.process(file)

    def process(self, raw_path: str, agent_id=None):
        df, am, scenario_id = load_av2_df(raw_path)
        city = df.city.values[0]
        agent_id = df["focal_track_id"].values[0]

        local_df = df[df["track_id"] == agent_id].iloc
        origin = torch.tensor(
            [local_df[49]["position_x"], local_df[49]["position_y"]],
            dtype=torch.float)
        theta = torch.tensor([local_df[49]["heading"]], dtype=torch.float)
        rotate_mat = torch.tensor([
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)],
        ], )

        timestamps = list(np.sort(df["timestep"].unique()))  # 50 - 60
        cur_df = df[df["timestep"] == timestamps[49]]  # align timestamp
        actor_ids = list(cur_df["track_id"].unique())
        cur_pos = torch.from_numpy(cur_df[["position_x",
                                           "position_y"]].values).float()
        out_of_range = np.linalg.norm(cur_pos - origin, axis=1) > self.radius
        actor_ids = [
            aid for i, aid in enumerate(actor_ids) if not out_of_range[i]
        ]
        actor_ids.remove(agent_id)
        actor_ids = [agent_id
                     ] + actor_ids  # make sure predicted object is first
        num_nodes = len(actor_ids)

        df = df[df["track_id"].isin(
            actor_ids)]  # delete objects which is not in radius.

        # initialization
        x = torch.zeros(num_nodes, 110, 2, dtype=torch.float)
        x_trans = torch.zeros(num_nodes, 110, 2, dtype=torch.float)
        x_attr = torch.zeros(num_nodes, 3, dtype=torch.uint8)
        x_heading = torch.zeros(num_nodes, 110, dtype=torch.float)
        x_velocity = torch.zeros(num_nodes, 110, dtype=torch.float)
        x_track_horizon = torch.zeros(num_nodes, dtype=torch.int)
        padding_mask = torch.ones(num_nodes, 110, dtype=torch.bool)

        for actor_id, actor_df in df.groupby("track_id"):
            node_idx = actor_ids.index(actor_id)
            node_steps = [timestamps.index(ts) for ts in actor_df["timestep"]]
            object_type = OBJECT_TYPE_MAP[actor_df["object_type"].values[0]]
            x_attr[node_idx, 0] = object_type
            x_attr[node_idx, 1] = actor_df["object_category"].values[0]
            x_attr[node_idx, 2] = OBJECT_TYPE_MAP_COMBINED[
                actor_df["object_type"].values[0]]
            x_track_horizon[node_idx] = node_steps[-1] - node_steps[
                0]  # full is 109

            padding_mask[node_idx, node_steps] = False
            if padding_mask[
                    node_idx,
                    49] or object_type in self.ignore_type:  # at least has one history frame and not static object
                padding_mask[node_idx, 50:] = True

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

            x[node_idx, node_steps, :2] = torch.matmul(pos_xy - origin,
                                                       rotate_mat)
            x_heading[node_idx, node_steps] = (heading - theta + np.pi) % (
                2 * np.pi
            ) - np.pi  # bounding heading angle from [0, 2pi] to [-pi, pi]
            x_velocity[node_idx, node_steps] = velocity_norm

        # (
        #     lane_positions,
        #     is_intersections,
        #     lane_ctrs,
        #     lane_angles,
        #     lane_attr,
        #     lane_padding_mask,
        # ) = self.get_lane_features(am, origin, origin, rotate_mat, self.radius)
        (
            lane_positions,
            is_intersections,
            lane_ctrs,
            lane_angles,
            lane_attr,
            lane_padding_mask,
            lane_candidate,
            pad_lane_mask,
        ) = self.get_candidate_lane_features(am, origin, origin, rotate_mat,
                                         self.radius)

        if self.remove_outlier_actors:
            lane_samples = lane_positions[:, ::1, :2].view(-1, 2)
            nearest_dist = torch.cdist(x[:, 49, :2],
                                       lane_samples).min(dim=1).values
            valid_actor_mask = nearest_dist < 5
            valid_actor_mask[0] = True  # always keep the target agent

            x = x[valid_actor_mask]
            x_heading = x_heading[valid_actor_mask]
            x_velocity = x_velocity[valid_actor_mask]
            x_attr = x_attr[valid_actor_mask]
            padding_mask = padding_mask[valid_actor_mask]
            num_nodes = x.shape[0]

        x_trans = x.clone()
        x_ctrs = x[:, 49, :2].clone()
        x_positions = x[:, :50, :2].clone()
        x_velocity_diff = x_velocity[:, :50].clone()

        x[:, 50:] = torch.where(
            (padding_mask[:, 49].unsqueeze(-1)
             | padding_mask[:, 50:]).unsqueeze(-1),
            torch.zeros(num_nodes, 60, 2),
            x[:, 50:] - x[:, 49].unsqueeze(-2),
        )
        x[:, 1:50] = torch.where(
            (padding_mask[:, :49] | padding_mask[:, 1:50]).unsqueeze(-1),
            torch.zeros(num_nodes, 49, 2),
            x[:, 1:50] - x[:, :49],
        )
        x_trans[:, 50:] = torch.where(
            (padding_mask[:, 49].unsqueeze(-1)
             | padding_mask[:, 50:]).unsqueeze(-1),
            torch.zeros(num_nodes, 60, 2),
            x_trans[:, 50:],
        )
        x_trans[:, :50] = torch.where(
            (padding_mask[:, :50] | padding_mask[:, :50]).unsqueeze(-1),
            torch.zeros(num_nodes, 50, 2),
            x_trans[:, :50],
        )
        x[:, 0] = torch.zeros(num_nodes, 2)

        x_velocity_diff[:, 1:50] = torch.where(
            (padding_mask[:, :49] | padding_mask[:, 1:50]),
            torch.zeros(num_nodes, 49),
            x_velocity_diff[:, 1:50] - x_velocity_diff[:, :49],
        )
        x_velocity_diff[:, 0] = torch.zeros(num_nodes)

        y = None if self.mode == "test" else x[:, 50:]

        return {
            "x": x[:, :50],
            "y": y,
            "x_trans": x_trans,
            "x_attr": x_attr,
            "x_positions": x_positions,
            "x_centers": x_ctrs,
            "x_angles": x_heading,
            "x_velocity": x_velocity,
            "x_velocity_diff": x_velocity_diff,
            "x_padding_mask": padding_mask,
            "lane_positions": lane_positions,
            "lane_centers": lane_ctrs,
            "lane_angles": lane_angles,
            "lane_attr": lane_attr,
            "lane_padding_mask": lane_padding_mask,
            "is_intersections": is_intersections,
            "origin": origin.view(-1, 2),
            "theta": theta,
            "scenario_id": scenario_id,
            "track_id": agent_id,
            "city": city,
            "lane_candidate": lane_candidate,
            "pad_lane_mask": pad_lane_mask,
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
        lanes_ctr = lane_positions[:,
                                   9:11].mean(dim=1)  # Get centerline center
        lanes_angle = torch.atan2(
            lane_positions[:, 10, 1] - lane_positions[:, 9, 1],
            lane_positions[:, 10, 0] - lane_positions[:, 9, 0],
        )  # use centerline's center calculate lane angle(has already been normed).
        is_intersections = torch.Tensor(is_intersections)
        lane_attrs = torch.stack(lane_attrs, dim=0)

        x_max, x_min = radius, -radius
        y_max, y_min = radius, -radius

        padding_mask = (
            (lane_positions[:, :, 0] > x_max)
            | (lane_positions[:, :, 0] < x_min)
            | (lane_positions[:, :, 1] > y_max)
            | (lane_positions[:, :, 1] < y_min))  # [num_lanes, 20] bool

        invalid_mask = padding_mask.all(
            dim=-1)  # wheather lane is out of range(150).
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

    @staticmethod
    def get_candidate_lane_features(
        am: ArgoverseStaticMap,
        query_pos: torch.Tensor,
        origin: torch.Tensor,
        rotate_mat: torch.Tensor,
        radius: float,
    ):
        # all lane segment
        lane_segments = am.get_nearby_lane_segments(query_pos.numpy(), radius)
        lane_segments_id = [lane.id for lane in lane_segments]
        # get all lane candidates with a bubble
        manhattan_throld = 2.5
        cur_lane_segments = am.get_nearby_lane_segments(
            query_pos.numpy(), manhattan_throld)

        # keep expanding until at least 1 lane is found
        while len(cur_lane_segments) < 1 and manhattan_throld < radius:
            manhattan_throld *= 2
            cur_lane_segments = am.get_nearby_lane_segments(
                query_pos.numpy(), manhattan_throld)
        assert len(cur_lane_segments) > 0, "No nearby lane was found!!"

        # dfs to get all successor and predecessor candiates
        candidate_segments = []
        for candidate_segment in cur_lane_segments:
            candidate_segments.extend(dfs(candidate_segment, am, False,
                                   lane_segments_id))
            candidate_segments.extend(dfs(candidate_segment, am, True, lane_segments_id))
        candidate_segments_id = [lane.id for lane in candidate_segments]
        candidate_segments_id = set(candidate_segments_id)
        lane_positions, is_intersections, lane_attrs, candidatae = [], [], [], []
        lane_center, lane_angle = [], []
        for segment in lane_segments:
            if segment.id in candidate_segments_id:
                candidatae.append(True)
            else:
                candidatae.append(False)
            lane_centerline, lane_width = interp_utils.compute_midpoint_line(
                left_ln_boundary=segment.left_lane_boundary.xyz,
                right_ln_boundary=segment.right_lane_boundary.xyz,
                num_interp_pts=20,
            )
            # 然后，计算中心线的总长度
            total_length = np.sum(np.sqrt(np.sum(np.diff(lane_centerline, axis=0)**2, axis=1)))
            if total_length < 5:
                # 如果中心线的总长度不足segment_length，只取首尾两个点
                if len(lane_centerline) > 1:
                    lane_centerline = np.array([lane_centerline[0], lane_centerline[-1]])
            else:
                # 计算需要的点的数量
                num_pts = int(np.round(total_length / 5))

                # 对中心线进行插值
                lane_centerline = interp_utils.interp_arc(num_pts, points=lane_centerline)
            lane_center.append(np.mean(lane_centerline, axis=0))
            # 计算中心点
            center_point_index = len(lane_centerline) // 2
            center_point = lane_centerline[center_point_index]

            # 计算中心点前后的点
            prev_point = lane_centerline[center_point_index - 1] if center_point_index - 1 >= 0 else center_point
            next_point = lane_centerline[center_point_index + 1] if center_point_index + 1 < len(lane_centerline) else center_point

            # 计算两点之间的差值
            delta_y = next_point[1] - prev_point[1]
            delta_x = next_point[0] - prev_point[0]

            # 计算并返回朝向
            orientation = np.arctan2(delta_y, delta_x)
            lane_angle.append(orientation)

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
        # pad tensor
        from torch.nn.utils.rnn import pad_sequence
        pad_lane_positions = pad_sequence(lane_positions, batch_first=True)  # [R, L, 2]
        pad_lane_mask = torch.zeros_like(pad_lane_positions, dtype=bool)  # [R, L, 2]
        for i, tensor in enumerate(lane_positions):
            pad_lane_mask[i, :tensor.shape[0]] = 1
        # lane_positions = torch.stack(lane_positions)
        # lanes_ctr = lane_positions[:,
        #                            9:11].mean(dim=1)  # Get centerline center
        # lanes_angle = torch.atan2(
        #     lane_positions[:, 10, 1] - lane_positions[:, 9, 1],
        #     lane_positions[:, 10, 0] - lane_positions[:, 9, 0],
        # )  # use centerline's center calculate lane angle(has already been normed).
        lanes_ctr = torch.Tensor(lane_center)
        lanes_angle = torch.Tensor(lane_angle)
        is_intersections = torch.Tensor(is_intersections)
        lane_attrs = torch.stack(lane_attrs, dim=0)
        candidatae = torch.Tensor(candidatae)
        lane_positions = pad_lane_positions
        x_max, x_min = radius, -radius
        y_max, y_min = radius, -radius

        padding_mask = (
            (lane_positions[:, :, 0] > x_max)
            | (lane_positions[:, :, 0] < x_min)
            | (lane_positions[:, :, 1] > y_max)
            | (lane_positions[:, :, 1] < y_min))  # [num_lanes, 20] bool

        invalid_mask = padding_mask.all(
            dim=-1)  # wheather lane is out of range(150).
        lane_positions = lane_positions[~invalid_mask]
        is_intersections = is_intersections[~invalid_mask]
        lane_attrs = lane_attrs[~invalid_mask]
        lanes_ctr = lanes_ctr[~invalid_mask]
        lanes_angle = lanes_angle[~invalid_mask]
        padding_mask = padding_mask[~invalid_mask]
        candidatae = candidatae[~invalid_mask]
        pad_lane_mask = pad_lane_mask[~invalid_mask]

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
            candidatae,
            pad_lane_mask,
        )


def dfs(lane_segment: LaneSegment,
        am: ArgoverseStaticMap,
        extend_along_predecessor: bool = False,
        lane_segments_id: List[int] = None) -> List[LaneSegment]:
    """Perform depth first search over lane graph.

        Args:
            lane_segment (LaneSegment): _description_
            extend_along_predecessor (bool, optional): _description_. Defaults to False.
            lane_segments (List[LaneSegment], optional): _description_. Defaults to None.
        """
    # break condition
    # if lane_segment.id not in lane_segments_id:
    traversed_lanes, traversed_id = [], []

    def helper(lane: LaneSegment):
        if lane.id not in lane_segments_id or lane.id in traversed_id:
            return
        traversed_lanes.append(lane)
        traversed_id.append(lane.id)
        child_lanes = []
        if extend_along_predecessor:
            for id in lane.predecessors:
                try:
                    child_lanes.append(am.vector_lane_segments[id])
                except:
                    pass
        else:
            for id in lane.successors:
                try:
                    child_lanes.append(am.vector_lane_segments[id])
                except:
                    pass

        if len(child_lanes):
            for child in child_lanes:
               helper(child)

    helper(lane_segment)
    return traversed_lanes
