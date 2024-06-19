import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

from torchvision.utils import _log_api_usage_once

from collections.abc import Sequence

from monai.losses.dice import DiceLoss


def softmax_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    gamma: float = 2,
    eps: float = 1e-6,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example. Softmax() is applied on this tensor
                to convert the raw logits to class probabilities. Expected shape is
                (N, C, *).
        targets (Tensor): Must be a long tensor similar to the one expected by
                PyTorch's CrossEntropyLoss.
                https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
                The class dimension is expected to be absent, and each
                element is the class value in the range [0, C).
        alpha (Tensor[float]): Weighting factor in range (0,1) to balance
                positive vs negative examples or None for no weighting. The elements
                of this alpha should sum up to 1.0. Default: ``None``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        eps (float): Small value to check if the sum of elements in alpha adds
                up to 1.0.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed.
                ``'instance-sum-batch-mean'``: The output will be summed for each
                        value in the batch, and then averaged across the entire
                        batch. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Adapted from this version by Thomas V.
    # https://discuss.pytorch.org/t/focal-loss-for-imbalanced-multi-class-classification-in-pytorch/61289/2
    # Referenced from this github issue:
    # https://github.com/pytorch/vision/issues/3250
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(softmax_focal_loss)

    assert (
        targets.dtype == torch.long
    ), f"Expected a long tensor for 'targets', but got {targets.dtype}"

    logits = inputs
    weight = None
    if alpha is not None:
        num_classes = logits.size(1)
        assert isinstance(
            alpha, torch.Tensor
        ), f"Expected alpha to be torch.Tensor, got {type(alpha)}"
        assert (
            alpha.size(0) == num_classes
        ), f"Expected alpha (weights) to have {num_classes} elements, but got {alpha.size(0)} elements"
        assert (
            abs(alpha.sum() - 1.0) <= eps
        ), f"Expected elements of alpha to sum 1.0, instead they sum to {alpha.sum()}"
        weight = alpha
    weight = weight.to(logits.device)
    log_p = F.log_softmax(input=logits, dim=1)
    ce_loss = F.nll_loss(input=log_p, target=targets, weight=weight)

    pt = torch.exp(log_p.take_along_dim(indices=targets.unsqueeze(dim=1), dim=1))
    focal_loss = ((1 - pt) ** gamma) * ce_loss
    if reduction == "none":
        return focal_loss
    elif reduction == "sum":
        return focal_loss.sum()
    elif reduction == "mean":
        return focal_loss.mean()
    elif reduction == "instance-sum-batch-mean":
        return focal_loss.sum() / logits.size(0)
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum', 'instance-sum-batch-mean'"
        )


def object_fill_loss(
    inputs: torch.Tensor,
    objects: torch.Tensor,
) -> torch.Tensor:

    batch_loss = torch.zeros(inputs.shape[0])

    for i,object in enumerate(objects):
        uniques = object.unique()
        individual_losses = torch.zeros(uniques.shape)
        for j,object_num in enumerate(uniques):
            _, count = inputs[i:i+1][object == object_num].unique(return_counts=True)
            individual_losses[j] = 1 - (count.max() / count.sum())
        batch_loss[i] = individual_losses.mean()
        
    batch_loss = batch_loss.mean()

    return batch_loss


















    
