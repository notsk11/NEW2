import random
import sys
import numpy as np
from PIL import Image
import gradio as gr
import torch
from diffusers import DiffusionPipeline
from modules import pipeline
from modules import pipeline as pipe_module
from modules.pipeline import load_pipeline

# Define the function that ensures a number is divisible by 8 and close to the input value
def closest_divisible_by_8(number: int) -> int:
    remainder = number % 8
    if remainder == 0:
        return number
    lower_nearest = number - remainder
    higher_nearest = number + (8 - remainder)
    # Return the nearest number divisible by 8
    if abs(number - lower_nearest) <= abs(number - higher_nearest):
        return lower_nearest
    else:
        return higher_nearest
# Define inpaint function
def inpaint(prompt_i2i, negative_prompt_i2i, image_input_inpaint, height_inpaint, width_inpaint, num_inference_steps_inpaint, guidance_scale_inpaint, strength_inpaint, batch_count_inpaint, seed_int="", scheduler_inpaint=None):
    global pipeline

    if pipe_module.pipeline is None:
        print("Pipeline is not loaded. Please click 'Load Pipeline' first.")
        return None, None, None  # Check if the pipeline is loaded

    # If seed is not provided, generate a random seed
    if seed_int == "":
        seed = random.randint(0, sys.maxsize)
    else:
        try:
            seed = int(seed_int)
        except ValueError:
            return None, None, "Invalid input. Please enter a valid number for seed."

    # Set the seed
    torch.manual_seed(seed)

    # Convert input images to RGB
    image_input_inpaint = image_input_inpaint.convert("RGB")
    mask_img = image_input_inpaint.convert("RGB")

    try:
        # Perform inpainting using the pipeline
        results = pipe_module.pipeline(
            prompt=prompt_i2i,
            negative_prompt=negative_prompt_i2i,
            image=image_input_inpaint,
            mask_image=mask_img,
            height=height_inpaint,
            width=width_inpaint,
            num_inference_steps=num_inference_steps_inpaint,
            guidance_scale=guidance_scale_inpaint,
            strength=strength_inpaint,
            num_images_per_prompt=batch_count_inpaint,
            scheduler=scheduler_inpaint  # Pass the scheduler if defined
        )

        # Convert results to numpy arrays and PIL images
        images_np = [np.array(img) for img in results.images]
        images_pil = [Image.fromarray(img_np) for img_np in images_np]
        
        # Create metadata string
        metadata_str = (
            f"Seed: {seed}, Prompt: {prompt_i2i}, Negative Prompt: {negative_prompt_i2i}, "
            f"Height: {height_inpaint}, Width: {width_inpaint}, Num Inference Steps: {num_inference_steps_inpaint}, "
            f"Guidance Scale: {guidance_scale_inpaint}, Strength: {strength_inpaint}"
        )
        
        # Return the images and metadata
        return images_pil, images_pil, metadata_str
    
    except Exception as e:
        # Print error message and return None values in case of exception
        print(f"Error during image generation: {e}")
        return None, None, None
