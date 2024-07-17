import torch


def sort_predictions(predictions, probability, k=6):
    """Sort the predictions based on the probability of each mode.
    Args:
        predictions (torch.Tensor): The predicted trajectories [b, k, t, 2].
        probability (torch.Tensor): The probability of each mode [b, k].
    Returns:
        torch.Tensor: The sorted predictions [b, k', t, 2].
    """
    indices = torch.argsort(probability, dim=-1, descending=True)
    sorted_prob = probability[torch.arange(probability.size(0))[:, None],
                              indices]
    sorted_predictions = predictions[torch.arange(predictions.size(0))[:,
                                                                       None],
                                     indices]
    return sorted_predictions[:, :k], sorted_prob[:, :k]


def sort_multi_predictions(predictions, probability, k=6):
    """Sort the predictions based on the probability of each mode. descending high->low
    Args:
        predictions (torch.Tensor): The predicted trajectories [b, a, k, t, 2].
        probability (torch.Tensor): The probability of each mode [b, k].
    Returns:
        torch.Tensor: The sorted predictions [b, a, k', t, 2].
        torch.Tensor: The sorted probability [b, k'].
    """
    B, A, _, _, _ = predictions.shape
    probability = probability.squeeze(-1)
    indices = torch.argsort(probability, dim=-1, descending=True)
    sorted_prob = torch.gather(probability, -1, indices)
    sorted_predictions = predictions[torch.arange(B)[..., None, None],
                                     torch.arange(A)[None, ..., None],
                                     indices, :]
    return sorted_predictions[:, :, :k], sorted_prob[:, :, :k]
