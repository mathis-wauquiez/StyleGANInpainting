import torch
import torch.nn.functional as F
import numpy as np

from .utils import get_stylegan_discriminator


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



import clip

import torch
import clip
from torchvision import transforms


from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode


class CLIP:
    """
    A wrapper around the CLIP model that computes the CLIP loss
    and allows differentiation with respect to the synthesized image.
    """

    def __init__(self, caption, model='ViT-B/32', device='cuda'):
        self.device = device
        self.model, _ = clip.load(model, device=device, jit=False)
        self.model.eval()  # Set the model to evaluation mode

        # Modified from clip/clip.py:_transform
        self.preprocess = transforms.Compose([
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # Tokenize and encode the caption
        self.text = clip.tokenize([caption]).to(device)

        with torch.no_grad():
            self.text_features = self.model.encode_text(self.text)

    def __call__(self, synth, target, mask):
        """
        Computes the CLIP loss between the synthesized image and the target description.
        This function allows differentiation with respect to 'synth'.
        """
        # Normalize the image tensor
        # synth = (synth + 1) / 2  # Convert from [-1, 1] to [0, 1]
        # synth = self.preprocess(synth)

        synth = (synth - synth.mean(dim=(0, 2, 3), keepdim=True)) / synth.std(dim=(0, 2, 3), keepdim=True)

        # Interpolate to (224, 224)
        synth = F.interpolate(synth, (224, 224), mode='bicubic', align_corners=False)

        # print(synth.mean(axis=(0, 2, 3)), synth.std(axis=(0, 2, 3)))

        # Add a batch dimension if necessary
        if synth.dim() == 3:
            synth = synth.unsqueeze(0)

        # Encode the image
        synth_features = self.model.encode_image(synth)

        # Compute cosine similarity loss
        loss = 1 - torch.cosine_similarity(synth_features, self.text_features, dim=-1).mean()
        # use the l2 loss instead
        # loss = F.mse_loss(synth_features, self.text_features, reduction='mean')
        return loss  # Loss ranges from 0 to 2


class DiscriminatorLoss:
    """
    A wrapper around the StyleGAN2 discriminator that computes the discriminator loss.
    """

    def __init__(self, device='cuda'):
        self.device = device
        self.model = get_stylegan_discriminator().to(device)
        # print(self.model.c_dim) # >> 0
        self.model.eval()  # Set the model to evaluation mode

    def __call__(self, synth, target_images, masks):
        """
        Computes the discriminator loss for the synthesized image.
        """
        # Add a batch dimension if necessary
        if synth.dim() == 3:
            synth = synth.unsqueeze(0)

        # Compute the discriminator predictions
        # make the sys stdout shut up to avoid the warnings
        import sys
        stdout = sys.stdout
        sys.stdout = open('trash', 'w')
        preds = self.model(synth, None)
        sys.stdout = stdout

        # Compute the generator loss
        loss = F.softplus(-preds).mean()
        return loss