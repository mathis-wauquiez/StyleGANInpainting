import torch
import torch.nn.functional as F
import numpy as np


def attachment_loss(x, y, mask, factor=1):
    """
    Compute the attachment loss.

    Args:
        x: The input tensor.
        y: The target tensor.
        mask: The mask tensor.

    Returns:
        The attachment loss.
    """
    loss = F.mse_loss(x * mask, y * mask, reduction='mean') * factor**2
    return loss

import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg').to('cuda')
def lpips_loss(x, y, mask):
    """
    Compute the LPIPS loss.

    Args:
        x: The input tensor.
        y: The target tensor.
        mask: The mask tensor.
        model: The LPIPS model.

    Returns:
        The LPIPS loss.
    """
    # x and y are in [0, 255]
    x = x * mask
    y = y * mask

    # Normalize to [-1, 1]
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    # Compute the loss
    loss = loss_fn_vgg(x, y) # [1, 1, 1, 1]
    loss = loss.mean()
    return loss
