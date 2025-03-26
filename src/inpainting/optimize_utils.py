import torch
from .losses import lpips_loss, attachment_loss, CLIP, DiscriminatorLoss, ClassifierLoss
from omegaconf.listconfig import ListConfig

from classifier.constant import ATTRIBUTES

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
        do_not_add = False
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
                class_name = ATTRIBUTES[args['class_index']]
                target_value = args.pop('target')
                print(f'Classification loss: {class_name} should be {target_value}')
                loss_fn = ClassifierLoss(class_index=args.pop('class_index'), value=target_value, margin=args.pop('margin'), device=device)
            else:
                if loss_fn in losses_dict:
                    loss_fn = losses_dict[loss_fn]
                else:
                    print('Loss function not found in image losses:', loss_fn)
                    do_not_add = True

        if not do_not_add:
            loss_fns.append((loss_fn, weight))

    criterion = lambda synth_images, target_images, masks: sum([loss_fn(synth_images, target_images, masks) * weight for loss_fn, weight in loss_fns])

    return criterion
