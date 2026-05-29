import os

# Detect Render environment
IS_RENDER = os.environ.get("RENDER") == "true"

def is_render():
    return IS_RENDER