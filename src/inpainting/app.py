# 100% AI generated code. Most features don't work. Use at your own risk.

import numpy as np
import torch
import gradio as gr
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import tempfile
import os
import threading
import cv2

class BinaryImageMaskingGradio:
    def __init__(self):
        # Initialize variables
        self.orig_image = None
        self.mask = None
        self.brush_size = 20
        self.is_eraser = False
        self.mask_history = []
        self.result_ready = threading.Event()
        self.result_image = None
        self.result_mask = None
        self.show_overlay = True
        self.overlay_color = [255, 0, 0, 128]  # RGBA: Semi-transparent red
        self.fill_mode = False
        self.tolerance = 10
        
    def set_image(self, image):
        """Set the image from various input types (tensor, numpy, path, etc.)"""
        if image is None:
            return None, None, None
        
        try:
            if isinstance(image, str):
                # Path to image
                self.orig_image = Image.open(image).convert("RGB")
            elif isinstance(image, torch.Tensor):
                # PyTorch tensor
                if len(image.shape) == 4:  # batch, channels, height, width
                    image = image.squeeze(0)  # Remove batch dimension if present
                if image.shape[0] == 3:  # channels, height, width
                    # Convert to numpy and transpose to height, width, channels
                    image_np = image.detach().cpu().numpy().transpose(1, 2, 0)
                    # Normalize if needed
                    if image_np.max() <= 1.0:
                        image_np = (image_np * 255).astype(np.uint8)
                    self.orig_image = Image.fromarray(image_np)
                else:
                    raise ValueError("Unsupported tensor format")
            elif isinstance(image, np.ndarray):
                # Numpy array
                if len(image.shape) == 3 and image.shape[2] == 3:  # height, width, channels
                    # Normalize if needed
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    self.orig_image = Image.fromarray(image)
                else:
                    raise ValueError("Unsupported numpy array format")
            elif isinstance(image, Image.Image):
                # PIL Image
                self.orig_image = image.convert("RGB")
            else:
                raise ValueError("Unsupported image format")
        except Exception as e:
            return None, None, f"Error loading image: {str(e)}"
        
        # Create a new binary mask
        self.mask = Image.new('L', self.orig_image.size, 0)
        self.mask_history = [self.mask.copy()]
        
        # Create overlay preview
        overlay = self.create_overlay(np.array(self.orig_image), np.array(self.mask))
        
        # Convert to numpy arrays for Gradio
        return np.array(self.orig_image), np.array(self.mask), overlay
    
    def create_overlay(self, image, mask):
        """Create an overlay of the mask on the original image"""
        if not self.show_overlay or image is None or mask is None:
            return image
            
        # Create a copy of the image
        overlay = image.copy()
        
        # Create colored mask (red semi-transparent)
        color_mask = np.zeros_like(overlay)
        mask_bool = mask > 127  # Binary threshold
        
        # Apply the color to masked areas
        color_mask[mask_bool] = self.overlay_color[:3]
        
        # Blend with original image
        alpha = self.overlay_color[3] / 255.0
        overlay = cv2.addWeighted(color_mask, alpha, overlay, 1, 0)
        
        return overlay
    
    def painting(self, image, mask, tool, brush_size, evt: gr.SelectData):
        """Handle drawing on the mask"""

        # Convert sketch event data to coordinates
        y, x = evt.index[0], evt.index[1]  # Gradio uses (y,x) format

        if image is None or mask is None:
            return None, None, None
        
        # Convert numpy arrays back to PIL
        if self.orig_image is None:
            self.orig_image = Image.fromarray(image).convert("RGB")
        
        if self.mask is None or self.mask.size != (mask.shape[1], mask.shape[0]):
            self.mask = Image.fromarray(mask).convert("L")
        
        # Update brush size
        self.brush_size = brush_size
        
        # Update tool
        self.is_eraser = (tool == "Unmask")
        
        # Check if we're in fill mode
        if self.fill_mode:
            # Perform fill operation
            return self.perform_fill(image, mask, tool, evt)
        
        # Get coordinates
        x, y = evt.index
        
        # Create a mask draw object
        mask_draw = ImageDraw.Draw(self.mask)
        
        # Draw on the mask - binary values only (0 or 255)
        value = 0 if self.is_eraser else 255
        mask_draw.ellipse(
            [
                x - self.brush_size, y - self.brush_size,
                x + self.brush_size, y + self.brush_size
            ],
            fill=value
        )
        
        # Save state for undo
        self.mask_history.append(self.mask.copy())
        
        # Create overlay
        mask_np = np.array(self.mask)
        image_np = np.array(self.orig_image)
        overlay = self.create_overlay(image_np, mask_np)
        
        # Return updated mask and overlay
        return image_np, mask_np, overlay
    
    def perform_fill(self, image, mask, tool, evt: gr.SelectData):
        """Perform fill operation"""
        # Convert to numpy for processing
        mask_np = np.array(mask)
        
        # Get coordinates
        x, y = evt.index
        
        # Determine fill value based on tool
        fill_value = 0 if tool == "Unmask" else 255
        
        # Convert mask to binary if it's not already
        binary_mask = (mask_np > 127).astype(np.uint8) * 255
        
        # Use OpenCV floodFill for the operation
        fill_mask = binary_mask.copy()
        # Create a mask for floodFill that is 2 pixels larger in each dimension
        h, w = fill_mask.shape[:2]
        flood_mask = np.zeros((h+2, w+2), np.uint8)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        cv2.floodFill(hsv_image, flood_mask, (x, y), fill_value,
                    loDiff=(self.tolerance,)*3,  # Apply to all channels
                    upDiff=(self.tolerance,)*3, 
                    flags=4)
        
        # Update mask
        self.mask = Image.fromarray(fill_mask)
        
        # Save state for undo
        self.mask_history.append(self.mask.copy())
        
        # Reset fill mode
        self.fill_mode = False
        
        # Create overlay
        overlay = self.create_overlay(image, fill_mask)
        
        return image, fill_mask, overlay
    
    def set_fill_mode(self, active, tolerance=10):
        """Set fill mode active or inactive"""
        self.fill_mode = active
        self.tolerance = tolerance
        return "Fill tool active. Click on the image to fill area." if active else "Fill tool deactivated."
    
    def toggle_overlay(self, image, mask, show_overlay):
        """Toggle the overlay visualization"""
        self.show_overlay = show_overlay
        
        # Create and return overlay
        overlay = self.create_overlay(image, mask)
        return overlay
    
    def undo(self, image, mask):
        """Undo the last drawing operation"""
        if len(self.mask_history) > 1:
            self.mask_history.pop()  # Remove current state
            self.mask = self.mask_history[-1].copy()  # Get previous state
            
            # Create overlay
            mask_np = np.array(self.mask)
            image_np = np.array(self.orig_image) if self.orig_image else image
            overlay = self.create_overlay(image_np, mask_np)
            
            return image_np, mask_np, overlay
        
        # Create overlay without changes
        overlay = self.create_overlay(image, mask)
        return image, mask, overlay
    
    def clear_mask(self, image, mask):
        """Clear the entire mask"""
        if self.orig_image is not None:
            self.mask = Image.new('L', self.orig_image.size, 0)
            self.mask_history = [self.mask.copy()]
            
            # Create overlay
            mask_np = np.array(self.mask)
            image_np = np.array(self.orig_image)
            overlay = self.create_overlay(image_np, mask_np)
            
            return image_np, mask_np, overlay
            
        # Create overlay without changes
        overlay = self.create_overlay(image, mask)
        return image, mask, overlay
    
    def invert_mask(self, image, mask):
        """Invert the binary mask"""
        if mask is not None:
            # Invert the mask (255 becomes 0, 0 becomes 255)
            inverted_mask = 255 - mask
            
            # Update the mask
            self.mask = Image.fromarray(inverted_mask)
            
            # Save state for undo
            self.mask_history.append(self.mask.copy())
            
            # Create overlay
            overlay = self.create_overlay(image, inverted_mask)
            
            return image, inverted_mask, overlay
            
        # Create overlay without changes
        overlay = self.create_overlay(image, mask)
        return image, mask, overlay
    
    def get_mask_stats(self, mask):
        """Calculate and return stats about the binary mask"""
        if mask is None:
            return "No mask available"
            
        total_pixels = mask.size
        masked_pixels = np.sum(mask > 127)
        percentage = (masked_pixels / total_pixels) * 100
        
        stats = f"Mask statistics:\n"
        stats += f"Total image pixels: {total_pixels:,}\n"
        stats += f"Masked pixels: {masked_pixels:,}\n"
        stats += f"Percentage masked: {percentage:.2f}%"
        
        return stats
    
    def ensure_binary(self, image, mask):
        """Ensure the mask is strictly binary (0 or 255)"""
        if mask is None:
            return image, mask, "No mask to process"
            
        # Apply threshold to ensure binary values
        binary_mask = (mask > 127).astype(np.uint8) * 255
        
        # Update the mask
        self.mask = Image.fromarray(binary_mask)
        
        # Add to history only if changed
        if not np.array_equal(mask, binary_mask):
            self.mask_history.append(self.mask.copy())
            msg = "Mask was thresholded to ensure binary values (0 or 255 only)"
        else:
            msg = "Mask is already binary (0 or 255 values only)"
        
        # Create overlay
        overlay = self.create_overlay(image, binary_mask)
        
        return image, binary_mask, overlay, msg
    
    def get_result(self, image, mask):
        """Store the result and signal completion"""
        if image is None or mask is None:
            self.result_ready.set()
            return "No image or mask to process."
        
        # Ensure binary mask
        binary_mask = (mask > 127).astype(np.uint8) * 255
        
        # Convert to PIL Images if they're numpy arrays
        if isinstance(image, np.ndarray):
            self.result_image = Image.fromarray(image).convert("RGB")
        else:
            self.result_image = image
            
        if isinstance(binary_mask, np.ndarray):
            self.result_mask = Image.fromarray(binary_mask).convert("L")
        else:
            self.result_mask = binary_mask
        
        # Create binary numpy array (0s and 1s) for ML use
        self.binary_numpy_mask = (binary_mask > 127).astype(np.uint8)
        
        # Signal that we're done
        self.result_ready.set()
        
        return "Masking complete. Binary mask created successfully."

def run_app(image=None):
    """
    Run the binary image masking application using Gradio
    
    Parameters:
    image : torch.Tensor, numpy.ndarray, str, or PIL.Image.Image
        The input image. Can be:
        - A path to an image file (str)
        - A PyTorch tensor (CxHxW or BxCxHxW, values between 0-1 or 0-255)
        - A numpy array (HxWxC, values between 0-1 or 0-255)
        - A PIL Image object
        - None, in which case the user will be prompted to upload an image
    
    Returns:
    tuple : (PIL.Image.Image, PIL.Image.Image, numpy.ndarray)
        A tuple containing (original_image, binary_mask, binary_numpy_array)
        If only two return values are expected, it will return (original_image, binary_mask)
    """
    # Create an instance of the masking tool
    masking_tool = BinaryImageMaskingGradio()
    
    # Define Gradio interface functions
    def process_upload(uploaded_image):
        if uploaded_image is not None:
            return masking_tool.set_image(uploaded_image)
        return None, None, "No image uploaded"
    
    def process_input_image():
        if image is not None:
            return masking_tool.set_image(image)
        return None, None, "No image provided"
    
    def handle_finish(image, mask):
        # Ensure binary mask and process results
        image, mask = image['image'], image['mask']
        # print(image, mask)
        print(np.unique(mask))
        binary_mask = (mask > 127).astype(np.uint8) * 255
        return masking_tool.get_result(image, binary_mask)
    
    def update_overlay(show_overlay, image, mask):
        masking_tool.show_overlay = show_overlay
        return masking_tool.create_overlay(image, mask)
    
    def update_stats(mask):
        return masking_tool.get_mask_stats(mask)
    
    def toggle_fill_tool(active, tolerance):
        return masking_tool.set_fill_mode(active, tolerance)
    
    # CSS for keyboard shortcuts
    css = """
    #keyboardShortcuts {
        margin: 10px 0;
        padding: 8px;
        background-color: #f8f9fa;
        border-radius: 4px;
    }
    #keyboardShortcuts kbd {
        background-color: #eee;
        border-radius: 3px;
        border: 1px solid #b4b4b4;
        box-shadow: 0 1px 1px rgba(0,0,0,.2);
        color: #333;
        display: inline-block;
        font-size: 0.85em;
        font-weight: 700;
        line-height: 1;
        padding: 2px 4px;
        white-space: nowrap;
    }
    """
    
    # Define the Gradio interface
    with gr.Blocks(title="Binary Image Masking Tool", css=css) as interface:
        status_text = gr.Textbox(label="Status", value="Ready to mask. Only binary values (0/255) will be used.", interactive=False)
        
        with gr.Row():
            with gr.Column(scale=3):
                # Original image with drawing capability
                input_image = gr.Image(label="Draw on Image", type="numpy", tool="sketch", interactive=True)
                
                # Mask image and overlay
                with gr.Row():
                    mask_image = gr.Image(label="Binary Mask (Black/White)", type="numpy", image_mode="L")
                    overlay_image = gr.Image(label="Overlay Preview", type="numpy")
            
            with gr.Column(scale=1):
                # Controls
                if image is None:
                    upload_button = gr.UploadButton("Upload Image", file_types=["image"])
                else:
                    # Load the provided image automatically
                    gr.HTML("<p>Image provided programmatically</p>")
                
                with gr.Accordion("Tools", open=True):
                    tool_select = gr.Radio(["Mask", "Unmask"], label="Tool", value="Mask")
                    brush_size_slider = gr.Slider(minimum=1, maximum=50, value=20, step=1, label="Brush Size")
                    
                with gr.Accordion("Fill Tool", open=True):
                    tolerance_slider = gr.Slider(minimum=0, maximum=50, value=10, step=1, label="Fill Tolerance")
                    fill_active = gr.Checkbox(label="Fill Tool Active", value=False)
                
                with gr.Accordion("Visualization", open=True):
                    show_overlay = gr.Checkbox(label="Show Mask Overlay", value=True)
                
                with gr.Accordion("Actions", open=True):
                    undo_button = gr.Button("Undo (Z)")
                    clear_button = gr.Button("Clear Mask")
                    invert_button = gr.Button("Invert Mask")
                    ensure_binary_button = gr.Button("Ensure Binary Values")
                
                with gr.Accordion("Mask Statistics", open=True):
                    stats_text = gr.Textbox(label="Mask Stats", value="No mask yet", interactive=False)
                
                with gr.Accordion("Keyboard Shortcuts", open=False):
                    gr.HTML("""
                    <div id="keyboardShortcuts">
                        <p><kbd>Z</kbd> - Undo last action</p>
                        <p><kbd>M</kbd> - Switch to Mask tool</p>
                        <p><kbd>U</kbd> - Switch to Unmask tool</p>
                        <p><kbd>F</kbd> - Toggle Fill tool</p>
                        <p><kbd>C</kbd> - Clear mask</p>
                        <p><kbd>I</kbd> - Invert mask</p>
                        <p><kbd>O</kbd> - Toggle overlay</p>
                        <p><kbd>+</kbd>/<kbd>-</kbd> - Increase/decrease brush size</p>
                    </div>
                    """)
                
                finish_button = gr.Button("Finish & Create Binary Mask", variant="primary")
        
        # Set up event handlers
        if image is None and 'upload_button' in locals():
            upload_button.upload(
                process_upload, 
                upload_button, 
                [input_image, mask_image, overlay_image]
            )
        
        # Drawing handlers
        # input_image.select(
        #     masking_tool.painting, 
        #     inputs=[input_image, mask_image, tool_select, brush_size_slider], 
        #     outputs=[input_image, mask_image, overlay_image]
        # )

        input_image.select(
            fn=masking_tool.painting,
            inputs=[input_image, mask_image, tool_select, brush_size_slider],
            outputs=[input_image, mask_image, overlay_image]
        )

        brush_size_slider.change(
            fn=lambda size: gr.update(label=f"Brush Size ({size}px)"),
            inputs=brush_size_slider,
            outputs=brush_size_slider
        )

        
        # Fill tool handler
        fill_active.change(
            toggle_fill_tool,
            inputs=[fill_active, tolerance_slider],
            outputs=[status_text]
        )
        
        # Other button handlers
        show_overlay.change(
            update_overlay,
            inputs=[show_overlay, input_image, mask_image],
            outputs=[overlay_image]
        )
        
        undo_button.click(
            masking_tool.undo,
            inputs=[input_image, mask_image],
            outputs=[input_image, mask_image, overlay_image]
        )
        
        clear_button.click(
            masking_tool.clear_mask,
            inputs=[input_image, mask_image],
            outputs=[input_image, mask_image, overlay_image]
        )
        
        invert_button.click(
            masking_tool.invert_mask,
            inputs=[input_image, mask_image],
            outputs=[input_image, mask_image, overlay_image]
        )
        
        ensure_binary_button.click(
            masking_tool.ensure_binary,
            inputs=[input_image, mask_image],
            outputs=[input_image, mask_image, overlay_image, status_text]
        )
        
        # Update stats when mask changes
        mask_image.change(
            update_stats,
            inputs=[mask_image],
            outputs=[stats_text]
        )
        
        finish_button.click(
            handle_finish,
            inputs=[input_image, mask_image],
            outputs=[status_text]
        )
        
        # Keyboard handler
        def handle_keyboard(key):
            if key == "z":
                undo_button.click()
            elif key == "m":
                return "Mask"
            elif key == "u":
                return "Unmask"
            elif key == "f":
                return not fill_active.value
            elif key == "c":
                clear_button.click()
            elif key == "i":
                invert_button.click()
            elif key == "o":
                return not show_overlay.value
            elif key == "plus" or key == "=":
                return min(50, brush_size_slider.value + 5)
            elif key == "minus" or key == "-":
                return max(1, brush_size_slider.value - 5)
            return tool_select.value
        
        # If an image was provided, load it automatically
        if image is not None:
            interface.load(
                process_input_image,
                inputs=None,
                outputs=[input_image, mask_image, overlay_image]
            )
    
    # Launch the interface in a separate thread to allow our code to continue
    interface.queue()  # Enable queuing to improve responsiveness
    interface.launch(share=False, inbrowser=True, prevent_thread_lock=True)
    
    # Wait for the user to complete masking
    masking_tool.result_ready.wait()
    
    # Close the interface
    interface.close()
    
    # Return the results
    if masking_tool.binary_numpy_mask is not None:
        # Return either 2 or 3 values depending on what the calling code expects
        try:
            return masking_tool.result_image, masking_tool.result_mask
        except:
            return masking_tool.result_image, masking_tool.result_mask, masking_tool.binary_numpy_mask
    else:
        return None, None

if __name__ == "__main__":
    # Example usage:
    # 1. With no image (user uploads)
    image, mask = run_app()
    
    # 2. With a file path
    # image, mask = run_app("path/to/image.jpg")
    
    # 3. With a numpy array
    # import cv2
    # img = cv2.imread("path/to/image.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    # image, mask = run_app(img)
    
    # 4. With a PyTorch tensor
    # import torch
    # from torchvision import transforms
    # from PIL import Image
    # img = Image.open("path/to/image.jpg")
    # tensor = transforms.ToTensor()(img)  # Creates a tensor with values 0-1
    # image, mask = run_app(tensor)
    
    if image is not None and mask is not None:
        # Show results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(np.array(image))
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(np.array(mask), cmap='gray')
        plt.title("Binary Mask")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        # Create overlay for visualization
        overlay = np.array(image).copy()
        mask_np = np.array(mask)
        mask_bool = mask_np > 127
        overlay[mask_bool] = np.array([255, 0, 0])  # Red overlay
        plt.imshow(overlay)
        plt.title("Overlay Result")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()