import numpy as np
import torch
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

from torch.amp import autocast, GradScaler
import sys
sys_stdout = sys.stdout

scaler = GradScaler()


from .losses import lpips_loss, attachment_loss, CLIP, DiscriminatorLoss

losses_dict = {
    'lpips': lpips_loss,
    'mse': attachment_loss,
}



def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [-1,2515], W & H must match G output resolution
    mask: torch.Tensor, # [1,H,W] with 1 for known pixels and 0 for unknown pixels
    losses: dict, # {loss_fn: weight} where loss_fn is a function that takes (synth_images, target_images, masks) and returns a scalar loss
    device: torch.device,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    learning_rate              = 0.1,
    verbose                    = False,
    visualize_progress         = True,  # Enable progress visualization
    visualize_frequency        = 50,     # Visualize every N stepsi
    use_encoder                = False,  # Use encoder to optimize w
):

    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    def get_total_loss(losses, synth_images, target_images, masks):
        """
        Get the total loss.
        Loss functions should take in synth_images, target_images, and masks.
        """
        total_loss = 0
        from omegaconf.listconfig import ListConfig
        if type(losses) == ListConfig:
            losses_ = {}
            for loss_dict in losses:
                losses_.update(loss_dict)
            losses = losses_

        for loss_fn, weight in losses.items():
            if type(loss_fn) == str:
                if loss_fn == 'clip':
                    args = dict(weight)
                    weight = args.pop('weight')
                    loss_fn = CLIP(caption=args.pop('caption'), model=args.pop('model'), device=device)
                elif loss_fn == 'disc':
                    loss_fn = DiscriminatorLoss(device=device)
                else:

                    loss_fn = losses_dict[loss_fn]

            loss = loss_fn(synth_images, target_images, masks)
            total_loss += loss * weight

        return total_loss
    
    def visualize_step(step, current_loss, synth_img, target_img, mask_img):
        """
        Visualize the current optimization progress
        """
        if not visualize_progress or step % visualize_frequency != 0:
            return
            
        # Convert tensors to numpy for visualization
        synth_img = (synth_img + .5) * 127.5
        target_img = (target_img + .5) * 127.5
        synth_np = synth_img.detach().cpu().numpy()[0].transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
        target_np = target_img[0].cpu().numpy().transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
        mask_np = mask_img[0].cpu().numpy().transpose(1, 2, 0).repeat(3, axis=2) * 255
        
        # Create a composite image showing masked target
        masked_target = target_np * mask_np.astype(np.uint8) / 255
        
        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 3, 1)
        plt.imshow(synth_np)
        plt.title(f'Step {step}: Current Synthesis')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(masked_target.astype(np.uint8))
        plt.title('Target (Masked)')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(target_np)
        plt.title('Target (Full)')
        plt.axis('off')
        
        plt.suptitle(f'Optimization Progress - Loss: {current_loss:.4f}')
        
        plt.tight_layout()
                    
        display(plt.gcf())
        

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w_avg, the initial latent code for optimization.
    z_samples = torch.randn(w_avg_samples, G.z_dim).to(device) # [N, Z]
    w_samples = G.mapping(z_samples, None)  # [N, L, C]
    w_avg = torch.mean(w_samples, axis=0, keepdims=True)      # [1, L, C]

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    masks = mask.unsqueeze(0).to(device).to(torch.float32)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([w_opt], lr=learning_rate)
    
    try:
        # Optimize latent code.
        for step in range(num_steps):

            # Generate images from w_opt.
            # We temporarily disable the output stream to prevent
            # the warnings in the generator from cluttering the logs.
            sys.stdout = open('logs.txt', 'w')

            # with autocast(device_type=device):

            synth_images = G.synthesis(w_opt)
            
            # Re-enable stdout
            sys.stdout = sys_stdout

            # Get the loss.
            loss = get_total_loss(losses, synth_images, target_images, masks)

            # Step
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_([w_opt], max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            logprint(f'step {step+1:>4d}/{num_steps}: loss {float(loss):<5.4f}')
            
            # Visualize progress
            visualize_step(step+1, float(loss), synth_images, target_images, masks)

    except KeyboardInterrupt:
        logprint('Interrupted')
        

    # Final visualization
    visualize_step(num_steps, float(loss), synth_images, target_images, masks)
    

    return w_opt.detach().cpu().numpy(), synth_images.detach().cpu().numpy()