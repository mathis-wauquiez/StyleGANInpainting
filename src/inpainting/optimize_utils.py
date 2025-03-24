import torch
from .losses import lpips_loss, attachment_loss, CLIP, DiscriminatorLoss, ClassifierLoss
from omegaconf.listconfig import ListConfig

losses_dict = {
    'lpips': lpips_loss,
    'mse': attachment_loss,
}

def get_criterion(losses, device='cuda'):
    loss_fns = []

    if type(losses) == ListConfig:
        losses_ = {}
        for loss_dict in losses:
            losses_.update(loss_dict)
        losses = losses_

    # add the loss functions to the list
    for loss_fn, weight in losses.items():
        if type(loss_fn) == str:
            if loss_fn == 'clip':
                args = dict(weight)
                weight = args.pop('weight')
                loss_fn = CLIP(caption=args.pop('caption'), model=args.pop('model'), device=device)
            elif loss_fn == 'disc':
                loss_fn = DiscriminatorLoss(device=device)
            elif loss_fn == 'classification':
                args = dict(weight)
                weight = args.pop('weight')
                loss_fn = ClassifierLoss(class_index=args.pop('class_index'), value=args.pop('target'), device=device)
            else:
                loss_fn = losses_dict[loss_fn]

        loss_fns.append((loss_fn, weight))

    criterion = lambda synth_images, target_images, masks: sum([loss_fn(synth_images, target_images, masks) * weight for loss_fn, weight in loss_fns])

    return criterion
