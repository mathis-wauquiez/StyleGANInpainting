import numpy as np
import torch
import torch.nn.functional as F
import copy
from IPython.display import clear_output, display
in_notebook = False
from torch.amp import autocast, GradScaler
import sys
from tqdm import tqdm
from .optimize_utils import get_criterion

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' 
import matplotlib.pyplot as plt

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
    learning_rate              = None,
    verbose                    = False,
    visualize_progress         = True,  # Enable progress visualization
    visualize_frequency        = 1,     # Visualize every N stepsi
    use_encoder                = False,  # Use encoder to optimize w
    scheduler                  = None,   # Learning rate scheduler
):


    if visualize_progress:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        img_synth = axes[0].imshow(np.zeros((256, 256, 3), dtype=np.uint8))
        img_masked_target = axes[1].imshow(np.zeros((256, 256, 3), dtype=np.uint8))
        img_target = axes[2].imshow(np.zeros((256, 256, 3), dtype=np.uint8))

        axes[0].set_title('Step: - Current Synthesis')
        axes[1].set_title('Target (Masked)')
        axes[2].set_title('Target (Full)')

        for ax in axes:
            ax.axis('off')

        plt.ion()  # Interactive mode
        plt.tight_layout()
        plt.show(block=False)  # Show the figure non-blockingly

    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    def visualize_step(step, current_loss, synth_img, target_img, mask_img):
        """
        Visualize the current optimization progress
        """
        if not visualize_progress or step % visualize_frequency != 0:
            return

        # Convert tensors to numpy for visualization
        synth_img = (synth_img + 0.5) * 127.5
        target_img = (target_img + 0.5) * 127.5
        synth_np = synth_img.detach().cpu().numpy()[0].transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
        target_np = target_img[0].cpu().numpy().transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
        mask_np = mask_img[0].cpu().numpy().transpose(1, 2, 0).repeat(3, axis=2) * 255

        # Create a composite image showing masked target
        masked_target = (target_np * (mask_np.astype(float)/255).astype(np.uint8)).astype(np.uint8)

        # Mise à jour des images affichées
        img_synth.set_data(synth_np)
        img_masked_target.set_data(masked_target)
        img_target.set_data(target_np)

        # Update titles
        axes[0].set_title(f'Step {step}: Current Synthesis')
        fig.suptitle(f'Optimization Progress - Loss: {current_loss:.4f}')

        # Redraw the figure and force update
        fig.canvas.draw_idle()  # redraw the figure
        fig.canvas.flush_events()  # process GUI events
        plt.pause(0.01)  # small pause to update


    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w_avg, the initial latent code for optimization.
    z_samples = torch.randn(w_avg_samples, G.z_dim).to(device) # [N, Z]
    w_samples = G.mapping(z_samples, None)  # [N, L, C]
    w_avg = torch.mean(w_samples, axis=0, keepdims=True)      # [1, L, C]

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    masks = mask.unsqueeze(0).to(device).to(torch.float32)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True)
    
    optimizer = torch.optim.Adam([w_opt], lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

    if scheduler is not None:
        print(scheduler)
        from hydra.utils import instantiate
        scheduler = instantiate(scheduler, optimizer=optimizer)

    criterion = get_criterion(losses, device=device)
    
    try:
        # Optimize latent code.
        with tqdm(total=num_steps, desc="Optimization", dynamic_ncols=True) as pbar:
            for step in range(num_steps):

                # Temporarily disable stdout to suppress generator warnings
                sys.stdout = open('logs.txt', 'w')

                # Generate images
                synth_images = G.synthesis(w_opt)

                # Re-enable stdout
                sys.stdout = sys_stdout

                # Compute loss
                loss = criterion(synth_images, target_images, masks)
                if 'w2' in losses:
                    loss += losses['w2'] * F.mse_loss(w_opt, w_avg)

                # Step
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_([w_opt], max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                # Update tqdm bar with loss value
                pbar.set_postfix(loss=f"{float(loss):.4f}")
                pbar.update(1)
                
                # Visualize progress
                visualize_step(step+1, float(loss), synth_images, target_images, masks)

                if scheduler is not None:
                    scheduler.step()


    except KeyboardInterrupt:
        logprint('Interrupted - Saving progress...')

    plt.ioff()  # Disable interactive mode
    # plt.show(block=True)  # Keep the final figure open

    # Final visualization
    visualize_step(num_steps, float(loss), synth_images, target_images, masks)
    plt.show()
    

    return w_opt.detach().cpu().numpy(), synth_images.detach().cpu().numpy()