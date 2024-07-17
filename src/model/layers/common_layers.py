# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi
# All Rights Reserved

import torch
import torch.nn as nn


def build_mlps(c_in,
               mlp_channels=None,
               ret_before_act=False,
               without_norm=False):
    layers = []
    num_layers = len(mlp_channels)

    for k in range(num_layers):
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k], bias=True))
        else:
            if without_norm:
                layers.extend(
                    [nn.Linear(c_in, mlp_channels[k], bias=True),
                     nn.ReLU()])
            else:
                layers.extend([
                    nn.Linear(c_in, mlp_channels[k], bias=False),
                    nn.BatchNorm1d(mlp_channels[k]),
                    nn.ReLU()
                ])
            c_in = mlp_channels[k]

    return nn.Sequential(*layers)


import torch
import torch.nn.functional as F


def calculate_relative_positions_angles(points1, angles1, points2, angles2):
    """计算points2 相对于 points1的位置和角度
       @input1: points1: [B, N, 2]
       @input2: angles1: [B, N]
       @input3: points2: [B, M, 2]
       @input4: angles2: [B, M]

       @return: point and angle diff: [B, N, M, 3]
    """

    B, N, _ = points1.size()
    _, M, _ = points2.size()

    # Expand angles to match the size of points
    angles1_expanded = angles1
    angles2_expanded = angles2

    # Create rotation matrix for both sets of points
    cos_matrix1 = angles1_expanded.cos()
    sin_matrix1 = angles1_expanded.sin()
    rotation_matrix1 = torch.stack(
        [cos_matrix1, -sin_matrix1, sin_matrix1, cos_matrix1], dim=-1)
    rotation_matrix1 = rotation_matrix1.view(B, N, 2, 2)

    cos_matrix2 = angles2_expanded.cos()
    sin_matrix2 = angles2_expanded.sin()
    rotation_matrix2 = torch.stack(
        [cos_matrix2, -sin_matrix2, sin_matrix2, cos_matrix2], dim=-1)
    rotation_matrix2 = rotation_matrix2.view(B, M, 2, 2)

    # Subtract the origin point from all points [B,N,M,2]
    points_diff = points2.view(B, 1, M, 2) - points1.view(B, N, 1, 2)

    # Apply rotation
    rotated_points = torch.matmul(points_diff, rotation_matrix1)
    # rotated_points = torch.einsum('bijk,bilk->bijl', points_diff, rotation_matrix1)
    rotated_points = rotated_points

    # Calculate relative angles
    angles_diff = angles2.view(B, 1, M) - angles1.view(B, N, 1)

    relative_pose = torch.cat(
        [rotated_points, angles_diff.unsqueeze(-1)], dim=-1)

    return relative_pose


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import math
    # Set batch size to 1
    B = 1
    N = 1  # number of points in set 1
    M = 1  # number of points in set 2

    # Define simple points and angles
    points1 = torch.tensor([[[1.0, 1.0]]])  # point at (1,1)
    angles1 = torch.tensor([[[math.radians(30)]]])  # angle 30 degrees
    points2 = torch.tensor([[[2.0, 2.0]]])  # point at (2,2)
    angles2 = torch.tensor([[[math.radians(30)]]])  # angle 30 degrees

    # Calculate relative positions and angles
    rotated_points, relative_angles = calculate_relative_positions_angles(
        points1, angles1, points2, angles2)

    # Set up the figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot original points and their directions
    for i in range(N):
        ax.plot(points1[0, i, 0], points1[0, i, 1], 'ro')
        ax.arrow(points1[0, i, 0],
                 points1[0, i, 1],
                 angles1[0, i, 0].cos(),
                 angles1[0, i, 0].sin(),
                 color='r',
                 width=0.05)
    for i in range(M):
        ax.plot(points2[0, i, 0], points2[0, i, 1], 'bo')
        ax.arrow(points2[0, i, 0],
                 points2[0, i, 1],
                 angles2[0, i, 0].cos(),
                 angles2[0, i, 0].sin(),
                 color='b',
                 width=0.05)

    # Plot rotated points and their directions
    for i in range(M):
        ax.plot(rotated_points[0, i, 0, 0], rotated_points[0, i, 0, 1], 'go')
        ax.arrow(rotated_points[0, i, 0, 0],
                 rotated_points[0, i, 0, 1],
                 relative_angles[0, i, 0].cos(),
                 relative_angles[0, i, 0].sin(),
                 color='g',
                 width=0.05)

    # Plot the origin
    ax.plot(0, 0, 'ko', label='Origin')

    # Set title, legend, and grid
    ax.set_title('Transformation')
    ax.legend()
    ax.grid(True)

    # Display the plot
    plt.show()
