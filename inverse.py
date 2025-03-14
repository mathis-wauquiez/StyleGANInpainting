import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

root_path = Path.cwd()
sys.path.append(str(root_path / "src"))
sys.path.append(str(root_path / "src/stylegan2"))



import torch
import torchvision.transforms as transforms

from inpainting.utils import get_stylegan_generator
from inpainting.optimize import project
from inpainting.losses import attachment_loss, lpips_loss


from PIL import Image


# G = get_stylegan_generator()

# size = G.img_resolution

# transform = transforms.Compose([
#     transforms.Resize((size, size)),
#     transforms.ToTensor(),
# ])


# image = transform(image)
# mask = transform(mask)
# mask = (~(mask > 0.5)).float()


# losses = {lpips_loss: 1, attachment_loss: 1}

# w, synths = project(
#     G,
#     target=image*2-1,
#     mask=mask,
#     losses=losses,
#     device='cuda',
#     verbose=True,
#     num_steps=1000,
# )

# generated_image = synths[0].detach().cpu().numpy().squeeze().transpose((1, 2, 0))


import hydra

import time

@hydra.main(config_path='configs', config_name='inverse', version_base=None)
def main(cfg):
    """ Inverse an image using StyleGAN2. """

    start_time = time.time()

    image = Image.open(cfg.image_path)
    image_name = Path(cfg.image_path).stem
    mask = Image.open(cfg.mask_path)

    seed = cfg.seed if cfg.seed is not None else torch.randint(0, 100000, (1,)).item()
    torch.manual_seed(seed)

    losses = cfg.losses

    save_folder = Path(cfg.save_folder)

    verbose = cfg.verbose
    device = cfg.device
    num_steps = cfg.num_steps

    # Preprocessing

    G = get_stylegan_generator()

    size = G.img_resolution

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    image = transform(image)
    mask = transform(mask)
    mask = (~(mask > 0.5)).float()

    # Inversion

    w, synths = project(
        G,
        target=image*2-1, # [-1, 1]
        mask=mask,
        losses=losses,
        device=device,
        verbose=verbose,
        num_steps=num_steps,
        learning_rate=cfg.learning_rate,
        visualize_progress=False
    )

    generated_image = (synths[0].clip(-1, 1)*127.5 + 127.5).transpose((1, 2, 0)).astype('uint8')

    # Save the w and the generated image to the file save_folder/w_[image_name]_[seed].npy or save_folder/image_[image_name]_[seed].png

    save_folder.mkdir(parents=True, exist_ok=True)

    w_save_path = save_folder / f'w_{image_name}_{seed}_{num_steps}.npy'
    image_save_path = save_folder / f'image_{image_name}_{seed}_{num_steps}.png'

    torch.save(w, w_save_path)
    Image.fromarray(generated_image).save(image_save_path)

    end_time = time.time()

    # log the results to a file
    with open(save_folder / f'log_{image_name}_{seed}_{num_steps}.txt', 'w') as f:
        f.write(f'Image: {image_name}\n')
        f.write(f'Seed: {seed}\n')
        f.write(f'Losses: {losses}\n')
        f.write(f'Num steps: {num_steps}\n')
        f.write(f'Time taken: {end_time - start_time} seconds\n')
        f.write(f'W saved to: {w_save_path}\n')
        f.write(f'Image saved to: {image_save_path}\n')

    # log the time taken to 'logs.txt' with the format [image_name]_[seed]_[num_steps]: [HH:MM:SS]
    import datetime
    time_taken = datetime.timedelta(seconds=end_time - start_time)
    time_taken = str(time_taken)

    with open(save_folder / 'logs.txt', 'a') as f:
        f.write(f'{image_name}_{seed}_{num_steps}: {time_taken}\n')

if __name__ == "__main__":
    main()
