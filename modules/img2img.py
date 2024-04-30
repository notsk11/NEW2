# /content/modules/img2img.py

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
from diffusers.utils import load_image

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

def img2img(prompt_i2i, negative_prompt_i2i, image_input_i2i, height_i2i, width_i2i, num_inference_steps_i2i, guidance_scale_i2i, strength_i2i, batch_count_i2i, seed_int="", scheduler_i2i=None):
    if seed_int == "":
        seed = random.randint(0, sys.maxsize)
    else:
        try:
            seed = int(seed_int)
        except ValueError:
            print("Invalid seed input. Please enter a valid number.")
            return None, None, None
    torch.manual_seed(seed)
    images = []
    try:
        if image_input_i2i is None:
            print("Image input is None.")
            return None, None, None

        # Handling single image or list of images
        if isinstance(image_input_i2i, list):
            for img_input in image_input_i2i:
                # Handle tuple input, if applicable
                if isinstance(img_input, tuple):
                    img_input = img_input[0]  # Use the first element in the tuple (URL or file path)

                if isinstance(img_input, str):
                    # Load image from URL or file path
                    if img_input.startswith(('http://', 'https://')):
                        image = load_image(img_input)
                    else:
                        image = Image.open(img_input)
                    images.append(image)
                elif isinstance(img_input, Image.Image):
                    images.append(img_input)
                else:
                    print(f"Invalid input format: {img_input}")
                    return None, None, None
        else:
            # Handle single input (URL, file path, or PIL image)
            if isinstance(image_input_i2i, str):
                if image_input_i2i.startswith(('http://', 'https://')):
                    image = load_image(image_input_i2i)
                else:
                    image = Image.open(image_input_i2i)
            elif isinstance(image_input_i2i, Image.Image):
                image = image_input_i2i
            else:
                print("Invalid input format for image. Provide a valid URL, file path, or PIL image.")
                return None, None, None
    except Exception as e:
        print(f"Error handling image input: {e}")
        return None, None, None

    global pipeline

    if pipe_module.pipeline is None:
        print("Pipeline is not loaded. Please click 'Load Pipeline' first.")
        return None, None, None

    try:
        initial_image = images[0] if images else None
        if initial_image is None:
            print("Initial image is None. Aborting generation.")
            return None, None, None
        # Ensure height and width are divisible by 8
        height_i2i = closest_divisible_by_8(height_i2i)
        width_i2i = closest_divisible_by_8(width_i2i)
        results = pipe_module.pipeline(
            prompt=prompt_i2i,
            negative_prompt=negative_prompt_i2i,
            image=initial_image,
            height=height_i2i,
            width=width_i2i,
            num_inference_steps=num_inference_steps_i2i,
            guidance_scale=guidance_scale_i2i,
            strength=strength_i2i,
            num_images_per_prompt=batch_count_i2i
        )

        images_np = [np.array(img) for img in results.images]
        images_pil = [Image.fromarray(img_np) for img_np in images_np]

        metadata_str = (
            f"Seed: {seed}, Prompt: {prompt_i2i}, Negative Prompt: {negative_prompt_i2i}, "
            f"Height: {height_i2i}, Width: {width_i2i}, Num Inference Steps: {num_inference_steps_i2i}, "
            f"Guidance Scale: {guidance_scale_i2i}, Strength: {strength_i2i}"
        )

        return images_pil, images_pil, metadata_str
    except Exception as e:
        print(f"Error during image generation: {e}")
        return None, None, None
