# __init__.py

# Import your node class(es) from the new file name
from .hgemini_nodes import HGemini25ProNode # Updated import

# A dictionary that contains all nodes you want to export with this package
NODE_CLASS_MAPPINGS = {
    "HGemini25ProNode": HGemini25ProNode # Updated class name in mapping
}

# A dictionary that contains the friendly names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "HGemini25ProNode": "H Gemini 2.5 Pro (Multi-Model)" # Updated display name for clarity
}

print("HGemini25ProNode: Custom Google Gemini AI node package loaded!")