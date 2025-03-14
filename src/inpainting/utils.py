import matplotlib.pyplot as plt
import torch
import torchvision

def plot_images(images, nrow=8):
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2, pad_value=1)
    grid = grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(nrow, len(images) // nrow))
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

import torch
import torch.nn.functional as F
import numpy as np

import pickle

def get_stylegan_generator():
    with open('models/stylegan2/ffhq.pkl', 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()
    return G

def get_stylegan_discriminator():
    with open('models/stylegan2/ffhq.pkl', 'rb') as f:
        D = pickle.load(f)['D'].cuda()
    return D