import torch
import torch.nn.functional as F
import numpy as np
import clip
from torchvision import transforms
from torchvision.transforms import Normalize
from classifier.classifier import ClassifierWrapper
from .utils import get_stylegan_discriminator

import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg').to('cuda')


def _stylegan2_to_255(x):
    """
    Convert the input tensor from [-1, 1] to [0, 255].
    """
    return (x + 1) * 127.5


def _stylegan2_to_01(x):
    """
    Convert the input tensor from [-1, 1] to [0, 1].
    """
    return (x + 1) / 2



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

    # Normalize to [0, 1]
    x = _stylegan2_to_01(x)
    y = _stylegan2_to_01(y)

    # Compute the loss
    loss = loss_fn_vgg(x, y) # [1, 1, 1, 1]
    loss = loss.mean()
    return loss


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
        # Add a batch dimension if necessary
        if synth.dim() == 3:
            synth = synth.unsqueeze(0)

        # Preprocess the image
        synth = self.preprocess(_stylegan2_to_01(synth))

        # Encode the image
        synth_features = self.model.encode_image(synth)

        # Compute cosine similarity loss
        loss = 1 - torch.cosine_similarity(synth_features, self.text_features, dim=-1).mean()
        # use the l2 loss instead
        # loss = F.mse_loss(synth_features, self.text_features, reduction='mean')
        return loss  # Loss ranges from 0 to 2


    # def __call__(self, synth, target, mask):
    #     """
    #     Computes the CLIP loss between the synthesized image and the target description.
    #     This function allows differentiation with respect to 'synth'.
    #     """
    #     # Normalize the image tensor
    #     # synth = (synth + 1) / 2  # Convert from [-1, 1] to [0, 1]
    #     # synth = self.preprocess(synth)

    #     synth = (synth - synth.mean(dim=(0, 2, 3), keepdim=True)) / synth.std(dim=(0, 2, 3), keepdim=True)

    #     # Interpolate to (224, 224)
    #     synth = F.interpolate(synth, (224, 224), mode='bicubic', align_corners=False)

    #     # print(synth.mean(axis=(0, 2, 3)), synth.std(axis=(0, 2, 3)))

    #     # Add a batch dimension if necessary
    #     if synth.dim() == 3:
    #         synth = synth.unsqueeze(0)

    #     # Encode the image
    #     synth_features = self.model.encode_image(synth)

    #     # Compute cosine similarity loss
    #     loss = 1 - torch.cosine_similarity(synth_features, self.text_features, dim=-1).mean()
    #     # use the l2 loss instead
    #     # loss = F.mse_loss(synth_features, self.text_features, reduction='mean')
    #     return loss  # Loss ranges from 0 to 2


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
    


class ClassifierLoss:
    """
    A wrapper around the classifier that computes the classifier loss
    for a specific class.
    """

    def __init__(self, class_index, value, device='cuda'):
        """
        Args:
            class_index (int): The target class index.
            value (int): The expected probability for the target class (0 or 1).
            device (str): Device to run the classifier on.
        """
        assert value in [0, 1], "Value must be 0 or 1."

        self.device = device
        self.class_index = class_index
        self.target_value = float(value)  # Ensure it's a float for loss computation
        self.model = ClassifierWrapper().to(device)
        self.model.eval()  # Set the model to evaluation mode

    def __call__(self, synth, target_images, masks):
        """
        Computes the classifier loss for the synthesized image.

        Args:
            synth (Tensor): The synthesized image (shape: CxHxW or BxCxHxW).
            target_images (Tensor): Not used but kept for interface consistency.
            masks (Tensor): Not used but kept for interface consistency.

        Returns:
            Tensor: The classifier loss.
        """
        # Ensure batch dimension
        if synth.dim() == 3:
            synth = synth.unsqueeze(0)

        # Normalize to [0,1] range
        synth = _stylegan2_to_01(synth).clamp(1e-3, 1-1e-3)

        # Get classifier predictions (logits)
        preds = self.model(synth)  # Shape: (batch_size, num_classes)

        # Extract the logits corresponding to the target class
        class_logits = preds[:, self.class_index]  # Shape: (batch_size,)

        # Compute binary cross-entropy loss
        target_probs = torch.full_like(class_logits, self.target_value, device=self.device)
        loss = F.binary_cross_entropy_with_logits(class_logits, target_probs)

        return loss
