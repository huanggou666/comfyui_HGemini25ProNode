import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import os

# You might need to import other ComfyUI specific modules if you fully implement
# media handling, e.g., for image processing:
# from PIL import Image
# import numpy as np
# import io
# import base64 # If base64 encoding is needed for API

class HGemini25ProNode: # Node class name updated
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                # Updated model list based on the provided free tier image from Google AI rates
                "model": ([
                    "gemini-2.5-pro",
                    "gemini-2.5-flash",
                    "gemini-2.5-flash-lite-preview-06-17",
                    "gemini-2.5-flash-preview-tts",
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-preview-image-generation",
                    "gemini-2.0-flash-lite",
                    "gemma-3-and-3n"
                ],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "control_after_generate": (["randomize", "increment", "fixed"],),
                "api_key": ("STRING", {"multiline": False, "default": "", "secret": True}),
            },
            "optional": { # Inputs here are not strictly required for the node to run
                "images": ("IMAGE", {"multiple": True}), # ComfyUI typically passes torch.Tensor for images
                "audio": ("AUDIO", {"multiple": True}),
                "video": ("VIDEO", {"multiple": True}),
                "files": ("FILE", {"multiple": True}),
                # Optional generation configuration parameters
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_output_tokens": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING",) # The node will output a single string
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "generate_content_with_gemini" # The method that ComfyUI will call to execute the node
    CATEGORY = "My AI/Google Gemini" # Category in ComfyUI's node menu

    def generate_content_with_gemini(self, prompt, model, seed, control_after_generate, api_key,
                                      images=None, audio=None, video=None, files=None, # Ensure optional inputs default to None
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
            # !!! IMPORTANT !!!
            # Verify the exact API model name for Gemma 3 & 3n from Google's official documentation.
            # This is a placeholder and might need to be adjusted (e.g., "models/gemma-3b-it").
            api_model_name = "models/gemma-3b" # PLACEHOLDER - VERIFY THIS STRING
        # Add more mappings here if any other display name doesn't match the API name directly
        # elif model == "some-other-display-name":
        #     api_model_name = "actual-api-model-name"


        # 3. Initialize the Model
        try:
            generation_model = genai.GenerativeModel(model_name=api_model_name)
            print(f"HGemini25ProNode: Initialized model: {api_model_name}")
        except Exception as e:
            print(f"HGemini25ProNode: Error initializing model '{api_model_name}': {e}")
            return (f"Error initializing model '{api_model_name}': {e}",)

        # 4. Prepare Content for the Model
        contents = [prompt]

        # --- CONCEPTUAL MEDIA HANDLING ---
        # The following blocks are placeholders. You will need to implement the actual logic
        # to convert ComfyUI's input data (e.g., PyTorch Tensors for images) into
        # a format compatible with the Google Gemini API (e.g., bytes with MIME type).

        if images is not None:
            print("HGemini25ProNode: Detected image input. Actual image processing logic needs to be implemented here.")
            # Example conceptual conversion:
            # for img_tensor in images:
            #     # Convert ComfyUI tensor to PIL Image (requires PIL, numpy)
            #     i = 255. * img_tensor.cpu().numpy()
            #     img_pil = Image.fromarray(i.astype(np.uint8))
            #
            #     # Convert PIL Image to bytes in a suitable format (e.g., JPEG or PNG)
            #     byte_arr = io.BytesIO()
            #     img_pil.save(byte_arr, format='PNG') # Or 'JPEG'
            #     contents.append({'mime_type': 'image/png', 'data': byte_arr.getvalue()})
            #
            # You might need to check if the model supports image input (e.g., gemini-2.0-flash-preview-image-generation)
            # or if the image content is part of a multi-turn conversation.

        if audio is not None:
            print("HGemini25ProNode: Detected audio input. Audio handling not implemented yet.")
            # Similar to images, you'd read audio data and format it for the API.

        if video is not None:
            print("HGemini25ProNode: Detected video input. Video handling not implemented yet.")
            # Similar to images, you'd read video data and format it for the API.

        if files is not None:
            print("HGemini25ProNode: Detected generic file input. File handling not implemented yet.")
            # Similar to images, you'd read file data and format it for the API.
        # --- END OF CONCEPTUAL MEDIA HANDLING ---


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
                safety_settings=[ # These settings try to reduce content blocking
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
            )
            # Access the generated text
            generated_text = response.text
            print("HGemini25ProNode: Content generation successful.")
        except Exception as e:
            # Handle API errors (rate limits, invalid API key, content moderation, etc.)
            generated_text = f"Error generating content: {e}"
            print(f"HGemini25ProNode: Error calling Gemini API: {e}")

        # 7. Return Result to ComfyUI
        return (generated_text,)