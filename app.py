import os
import sys
import torch
import numpy as np
import gradio as gr
from pathlib import Path
import time
import datetime
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import yaml

# Add necessary paths
root_path = Path.cwd()
sys.path.append(str(root_path / "src"))
sys.path.append(str(root_path / "src/stylegan2"))

from inpainting.utils import get_stylegan_generator
from inpainting.optimize import project
from inpainting.losses import attachment_loss, lpips_loss, CLIP, DiscriminatorLoss

# Global variables
generator = None
optimization_running = True
optimization_paused = False

def load_generator():
    global generator
    if generator is None:
        generator = get_stylegan_generator()
    return generator

def process_image_and_mask(image, mask):
    G = load_generator()
    size = G.img_resolution
    
    # Convert to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Handle mask from Gradio sketch tool which returns a dict
    if isinstance(mask, dict):
        # Extract the mask image from the dictionary
        if "mask" in mask:
            mask_array = mask["mask"]
            # Convert to PIL Image
            mask = Image.fromarray(mask_array)
        else:
            # Create a blank mask if no mask is drawn
            mask = Image.new("RGB", image.size, (0, 0, 0))
    elif not isinstance(mask, Image.Image):
        mask = Image.fromarray(mask)
    
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image)
    mask_tensor = transform(mask)
    mask_tensor = (~(mask_tensor > 0.5)).float()  # Invert mask
    
    return image_tensor, mask_tensor, G

def parse_losses(losses_config):
    """Parse losses from the configuration string"""
    losses = []
    
    try:
        # Try parsing as YAML
        config = yaml.safe_load(losses_config)
        if isinstance(config, list):
            losses = config
        else:
            # If it's a dictionary, convert to list format
            for loss_type, weight in config.items():
                if isinstance(weight, dict):
                    losses.append({loss_type: weight})
                else:
                    losses.append({loss_type: weight})
    except Exception as e:
        print(f"Error parsing losses: {e}")
        # Default losses if parsing fails
        losses = [
            {"mse": 0.4},
            {"lpips": 1.0}
        ]
    
    return losses

def custom_visualize_step(step, current_loss, synth_img, target_img, mask_img):
    """Custom visualization function that returns the image instead of displaying it"""
    # Convert tensors to numpy for visualization
    synth_img = (synth_img + .5) * 127.5
    target_img = (target_img + .5) * 127.5
    synth_np = synth_img.detach().cpu().numpy()[0].transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
    target_np = target_img[0].cpu().numpy().transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
    mask_np = mask_img[0].cpu().numpy().transpose(1, 2, 0) * 255
    
    # Create a composite image showing masked target
    masked_target = target_np * mask_np.astype(np.uint8) / 255
    
    # Create a figure
    fig = plt.figure(figsize=(18, 6))
    
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
    
    # Convert figure to image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return img, synth_np

def optimize_with_updates(image, mask, num_steps, learning_rate, losses_config, caption, device, progress=gr.Progress()):
    global optimization_running, optimization_paused
    optimization_running = True
    optimization_paused = False
    
    # Process image and mask
    image_tensor, mask_tensor, G = process_image_and_mask(image, mask)
    
    # Parse losses
    losses = parse_losses(losses_config)
    
    # Add CLIP loss if caption is provided
    if caption and caption.strip():
        clip_config = {
            "model": "RN101",
            "weight": 1.0,
            "caption": caption
        }
        losses.append({"clip": clip_config})
    
    # Custom project function with yield for updates
    target = image_tensor * 2 - 1  # [-1, 1]
    
    # Setup for optimization
    G = G.eval().requires_grad_(False).to(device)
    w_avg_samples = 10000
    z_samples = torch.randn(w_avg_samples, G.z_dim).to(device)
    w_samples = G.mapping(z_samples, None)
    w_avg = torch.mean(w_samples, axis=0, keepdims=True)
    
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    masks = mask_tensor.unsqueeze(0).to(device).to(torch.float32)
    
    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([w_opt], lr=learning_rate)
    
    # For storing results
    result_images = []
    final_w = None
    final_synth = None
    
    try:
        # Optimize latent code
        for step in range(num_steps):
            if not optimization_running:
                break
                
            while optimization_paused:
                time.sleep(0.1)
                if not optimization_running:
                    break
            
            # Generate images from w_opt
            import sys
            tmp = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            synth_images = G.synthesis(w_opt)
            sys.stdout = tmp
            
            # Get the loss
            from inpainting.optimize import get_total_loss
            loss = get_total_loss(losses, synth_images, target_images, masks, device=device)
            
            # Step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([w_opt], max_norm=1.0)
            optimizer.step()

            print(f'step {step+1:>4d}/{num_steps}: loss {float(loss):<5.4f}')
            
            # Visualize progress
            if step % 5 == 0 or step == num_steps - 1:  # Update every 5 steps
                progress(step / num_steps, f"Step {step+1}/{num_steps}")
                vis_img, synth_np = custom_visualize_step(step+1, float(loss), synth_images, target_images, masks)
                result_images.append(vis_img)
                final_synth = synth_np
                final_w = w_opt.detach().cpu()
                
                # Yield intermediate result
                yield vis_img, result_images
    
    except Exception as e:
        raise
    
    # Save final results if available
    if final_w is not None and final_synth is not None:
        save_folder = Path("output/interactive")
        save_folder.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        w_save_path = save_folder / f'w_{timestamp}.pt'
        image_save_path = save_folder / f'image_{timestamp}.png'
        
        torch.save(final_w, w_save_path)
        Image.fromarray(final_synth).save(image_save_path)
        
        print(f"Results saved to {save_folder}")
    
    # Return the final visualization and all intermediate visualizations
    if result_images:
        return result_images[-1], result_images
    else:
        return None, []

def pause_optimization():
    global optimization_paused
    optimization_paused = not optimization_paused
    return "Resume" if optimization_paused else "Pause"

def stop_optimization():
    global optimization_running
    optimization_running = False
    return "Stopped"

def create_app():
    with gr.Blocks() as app:
        gr.Markdown("# StyleGAN Inpainting App")
        gr.Markdown("Upload an image, draw a mask on the areas to inpaint, configure parameters, and run the optimization.")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Input Image", type="pil")
                mask_input = gr.Image(
                    label="Draw Mask (white areas will be inpainted)", 
                    type="numpy",  # Use numpy instead of pil
                    tool="sketch", 
                    brush_radius=20
                )

                
                with gr.Accordion("Advanced Configuration", open=False):
                    losses_config = gr.Textbox(
                        label="Losses Configuration (YAML format)",
                        value="""
- mse: 0.4
- lpips: 1.0
- disc: 0.1
""",
                        lines=5
                    )
                    
                    caption = gr.Textbox(
                        label="CLIP Text Prompt (optional)",
                        placeholder="Describe the content you want in the inpainted region..."
                    )
                    
                    with gr.Row():
                        num_steps = gr.Slider(label="Number of Steps", minimum=50, maximum=500, value=250, step=10)
                        learning_rate = gr.Slider(label="Learning Rate", minimum=0.01, maximum=0.5, value=0.1, step=0.01)
                    
                    device = gr.Radio(label="Device", choices=["cuda", "cpu"], value="cuda")
                
                with gr.Row():
                    run_btn = gr.Button("Run Optimization", variant="primary")
                    pause_btn = gr.Button("Pause")
                    stop_btn = gr.Button("Stop")
            
            with gr.Column(scale=1):
                output_image = gr.Image(label="Current Result")
                gallery = gr.Gallery(label="Optimization Progress", show_label=True, elem_id="gallery").style(grid=2, height="auto")
        
        # Set up event handlers
        run_btn.click(
            fn=optimize_with_updates,
            inputs=[input_image, mask_input, num_steps, learning_rate, losses_config, caption, device],
            outputs=[output_image, gallery]
        )
        
        pause_btn.click(
            fn=pause_optimization,
            inputs=[],
            outputs=[pause_btn]
        )
        
        stop_btn.click(
            fn=stop_optimization,
            inputs=[],
            outputs=[stop_btn]
        )
        
        # Example images
        gr.Markdown("## Examples")
        examples = gr.Examples(
            examples=[
                ["data/pairs/sample_0_image.png", "data/pairs/sample_0_mask.png", 250, 0.1, 
                 "- mse: 0.4\n- lpips: 1.0\n- disc: 0.1", 
                 "A close-up portrait of a young woman with long wavy red hair smiles warmly at the camera.", "cuda"]
            ],
            inputs=[input_image, mask_input, num_steps, learning_rate, losses_config, caption, device]
        )
        
    return app

if __name__ == "__main__":
    app = create_app()
    app.queue().launch(share=True)
