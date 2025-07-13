import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import os
from PIL import Image
import numpy as np
import io
import torch

class HGemini25ProNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": ([
                    "gemini-2.5-pro",
                    "gemini-2.5-pro-preview-05-06",
                    "gemini-2.5-flash",
                    "gemini-2.5-flash-lite-preview-06-17",
                    "gemini-2.5-flash-preview-tts",
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-preview-image-generation",
                    "gemini-2.0-flash-lite",
                    "gemma-3-12b-it",
                    "gemma-3-27b-it"
                ],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "control_after_generate": (["randomize", "increment", "fixed"],),
                "api_key": ("STRING", {"multiline": False, "default": "", "secret": True}),
            },
            "optional": {
                "images": ("IMAGE", {"multiple": True}),
                "audio": ("AUDIO", {"multiple": True}),
                "video": ("VIDEO", {"multiple": True}),
                "files": ("FILE", {"multiple": True}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_output_tokens": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "generate_content_with_gemini"
    CATEGORY = "My AI/Google Gemini"

    def tensor_to_pil(self, tensor):
        """Convert ComfyUI tensor to PIL Image"""
        # ComfyUI tensors are usually in format [batch, height, width, channels]
        # and values are typically in range [0, 1]
        if isinstance(tensor, torch.Tensor):
            # Convert to numpy and ensure it's in the right format
            np_image = tensor.cpu().numpy()
            
            # Handle different tensor shapes
            if len(np_image.shape) == 4:  # [batch, height, width, channels]
                np_image = np_image[0]  # Take first image from batch
            elif len(np_image.shape) == 3 and np_image.shape[0] == 3:  # [channels, height, width]
                np_image = np_image.transpose(1, 2, 0)  # Convert to [height, width, channels]
            
            # Convert from [0, 1] to [0, 255]
            if np_image.max() <= 1.0:
                np_image = (np_image * 255).astype(np.uint8)
            else:
                np_image = np_image.astype(np.uint8)
            
            # Handle grayscale
            if len(np_image.shape) == 2:
                np_image = np.stack([np_image] * 3, axis=-1)
            
            # Ensure we have 3 channels (RGB)
            if np_image.shape[-1] == 4:  # RGBA
                np_image = np_image[:, :, :3]  # Remove alpha channel
            
            return Image.fromarray(np_image)
        else:
            raise ValueError(f"Unsupported tensor type: {type(tensor)}")

    def generate_content_with_gemini(self, prompt, model, seed, control_after_generate, api_key,
                                      images=None, audio=None, video=None, files=None,
                                      temperature=0.7, max_output_tokens=1024, top_p=1.0, top_k=40):
        # 1. API Key Configuration
        if not api_key:
            print("HGemini25ProNode: API Key is missing. Please provide it in the node settings.")
            return ("Error: API Key is required.",)

        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            print(f"HGemini25ProNode: Error configuring Google Generative AI: {e}")
            return (f"Error configuring GenAI: {e}",)

        # 2. Map Display Model Name to Actual API Model Name
        api_model_name = model
        if model == "gemma-3-and-3n":
            api_model_name = "models/gemma-3b"

        # 3. Initialize the Model
        try:
            generation_model = genai.GenerativeModel(model_name=api_model_name)
            print(f"HGemini25ProNode: Initialized model: {api_model_name}")
        except Exception as e:
            print(f"HGemini25ProNode: Error initializing model '{api_model_name}': {e}")
            return (f"Error initializing model '{api_model_name}': {e}",)

        # 4. Prepare Content for the Model
        contents = [prompt]

        # --- IMPLEMENTED IMAGE HANDLING ---
        if images is not None:
            print("HGemini25ProNode: Processing image inputs...")
            try:
                # Handle single image or multiple images
                if not isinstance(images, list):
                    images = [images]
                
                for img_tensor in images:
                    # Convert ComfyUI tensor to PIL Image
                    img_pil = self.tensor_to_pil(img_tensor)
                    
                    # Convert PIL Image to bytes in PNG format
                    byte_arr = io.BytesIO()
                    img_pil.save(byte_arr, format='PNG')
                    byte_arr.seek(0)
                    
                    # Add image to contents in the format expected by Gemini API
                    contents.append({
                        'mime_type': 'image/png',
                        'data': byte_arr.getvalue()
                    })
                    
                print(f"HGemini25ProNode: Successfully processed {len(images)} image(s)")
                
            except Exception as e:
                print(f"HGemini25ProNode: Error processing images: {e}")
                return (f"Error processing images: {e}",)

        # Audio, video, and file handling remain as placeholders
        if audio is not None:
            print("HGemini25ProNode: Detected audio input. Audio handling not implemented yet.")

        if video is not None:
            print("HGemini25ProNode: Detected video input. Video handling not implemented yet.")

        if files is not None:
            print("HGemini25ProNode: Detected generic file input. File handling not implemented yet.")

        # 5. Set up Generation Configuration
        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
        )

        # 6. Call the Gemini API
        try:
            print(f"HGemini25ProNode: Sending content to {api_model_name}...")
            response = generation_model.generate_content(
                contents,
                generation_config=generation_config,
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
            )
            generated_text = response.text
            print("HGemini25ProNode: Content generation successful.")
        except Exception as e:
            generated_text = f"Error generating content: {e}"
            print(f"HGemini25ProNode: Error calling Gemini API: {e}")

        # 7. Return Result to ComfyUI
        return (generated_text,)
