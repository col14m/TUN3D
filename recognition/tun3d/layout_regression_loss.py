from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
from mmdet3d.registry import MODELS
from mmdet.models.losses.utils import weighted_loss


@weighted_loss
def l1_loss_bev(pred_layout, gt_layout):
    """Compute L1 loss for layout quads.

    Args:
        pred_layout (Tensor): Predicted layout, shape (..., 5).
        gt_layout (Tensor): Ground-truth layout, same shape as pred_layout.

    Returns:
        Tensor: Element-wise L1 loss.
    """

    coord_error = torch.sum(
        torch.abs(pred_layout[..., :4] - gt_layout[..., :4]), dim=-1)
    height_error = torch.abs(pred_layout[..., 4] - gt_layout[..., 4])
    return coord_error + height_error


@MODELS.register_module()
class L1Loss(nn.Module):
    """Normalized L1 loss wrapper for layout regression.

    Args:
        reduction (str): Reduction method ('mean', 'sum', 'none'). Default: 'mean'.
        normalization (bool): Whether to normalize the loss. Default: False.
        loss_weight (float): Scaling factor for the loss. Default: 1.0.

    """

    def __init__(self,
                 reduction: str = 'mean',
                 normalization: bool = False,
                 loss_weight: float = 1.0):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.normalization = normalization
        self.loss = l1_loss_bev

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                reduction_override: Optional[str] = None,
                **kwargs):

        if weight is not None and not torch.any(weight > 0):
            return pred.sum() * weight.sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            weight = weight.mean(-1)
        loss = self.loss_weight * self.loss(
            pred,
            target,
            weight,
            reduction=reduction,
            **kwargs)

        return loss
