import torch

def angle_between_2d_vectors(
        ctr_vector: torch.Tensor,
        nbr_vector: torch.Tensor) -> torch.Tensor:
    return torch.atan2(ctr_vector[..., 0] * nbr_vector[..., 1] - ctr_vector[..., 1] * nbr_vector[..., 0],
                       (ctr_vector[..., :2] * nbr_vector[..., :2]).sum(dim=-1))