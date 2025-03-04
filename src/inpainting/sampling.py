
import torch
import torch.nn.functional as F
import numpy as np
from .losses import attachment_loss

def sample(generator, n_samples, device="cuda", c=None, seed=None, trunc=None):
    """
    Sample from the generator.

    Args:
        generator: The generator model.
        n_samples: The number of samples to generate.
        device: The device to use.

    Returns:
        The generated samples.
    """
    
    if type(device) == str:
        device = torch.device(device)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    with torch.no_grad():
        if trunc is None:
            z = torch.randn(n_samples, generator.z_dim).to(device)
        else:
            z = torch.randn(n_samples, generator.z_dim).to(device)
            z = trunc * (z / z.norm(dim=1, keepdim=True).clamp(min=1e-8))

        if c is None:
            samples = generator(z, None)
        else:
            c = c.repeat(n_samples, 1)
            samples = generator(z, c)

    return samples

def sample_silent(*args, **kwargs):
    """
    Sample from the generator, silencing the output.
    This is a monkey-patch for the sampling function of StyleGAN2, that tries to build a plugin and fails every time.

    Args:
        generator: The generator model.
        n_samples: The number of samples to generate.
        device: The device to use.

    Returns:
        The generated samples.
    """
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    tmp = sys.stdout
    sys.stdout = open('logs.txt', 'w')
    samples = sample(*args, **kwargs)
    sys.stdout = tmp
    warnings.filterwarnings("default")
    return samples