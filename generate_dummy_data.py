import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def create_synthetic_xray(label, size=(224, 224)):
    img = Image.new("L", size, color=20)
    draw = ImageDraw.Draw(img)
    # Lungs
    draw.ellipse([30,40,100,200], fill=40)
    draw.ellipse([124,40,194,200], fill=40)
    # Heart/mediastinum
    draw.ellipse([90,60,134,190], fill=70)
    # Ribs
    for y in range(60, 180, 20):
        draw.line([25,y,95,y+5], fill=120, width=4)
        draw.line([129,y,199,y+5], fill=120, width=4)
    # Disease features
    if label == "BACTERIA":
        opacity = Image.new("L", size, 0)
        op_draw = ImageDraw.Draw(opacity)
        op_draw.ellipse([135,110,185,160], fill=180)
        opacity = opacity.filter(ImageFilter.GaussianBlur(12))
        img = Image.blend(img, opacity, 0.4)
    elif label == "VIRUS":
        opacity = Image.new("L", size, 0)
        op_draw = ImageDraw.Draw(opacity)
        op_draw.ellipse([45,90,75,120], fill=140)
        op_draw.ellipse([50,140,80,170], fill=120)
        op_draw.ellipse([140,80,170,110], fill=130)
        op_draw.ellipse([135,130,165,160], fill=140)
        opacity = opacity.filter(ImageFilter.GaussianBlur(8))
        img = Image.blend(img, opacity, 0.4)
    # Noise
    arr = np.array(img, dtype=np.float32)
    arr = np.clip(arr + np.random.normal(0, 5, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def setup_dataset():
    base = "dataset"
    splits = ["train", "val", "test"]
    classes = ["NORMAL", "BACTERIA", "VIRUS"]
    counts = {"train": 40, "val": 15, "test": 15}
    for split in splits:
        for cls in classes:
            dir_path = os.path.join(base, split, cls)
            os.makedirs(dir_path, exist_ok=True)
            for i in range(counts[split]):
                img = create_synthetic_xray(cls)
                img.save(os.path.join(dir_path, f"{cls.lower()}_synth_{i+1}.jpeg"))
    print("✅ Synthetic dataset created in 'dataset/'")

if __name__ == "__main__":
    setup_dataset()