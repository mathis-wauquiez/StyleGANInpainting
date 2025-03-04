import torch
import torch.nn.functional as F
import numpy as np


def attachment_loss(x, y, mask):
    """
    Compute the attachment loss.

    Args:
        x: The input tensor.
        y: The target tensor.
        mask: The mask tensor.

    Returns:
        The attachment loss.
    """
    return F.mse_loss(x * mask, y * mask, reduction='sum') / mask.sum()
