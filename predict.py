import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from explainable_ai import get_gradcam_heatmap, save_and_display_gradcam

def run_prediction(image_path, model_path="model/pneumonia_model.h5", encoder_path="model/label_encoder.json", output_cam=None):
    if not os.path.exists(image_path):
        print("Image not found"); sys.exit(1)
    model = tf.keras.models.load_model(model_path)
    with open(encoder_path, "r") as f:
        label_map = json.load(f)
    img = Image.open(image_path).convert('RGB').resize((224,224))
    img_arr = np.array(img, dtype=np.float32)
    img_pre = tf.keras.applications.resnet50.preprocess_input(img_arr)
    batch = np.expand_dims(img_pre, axis=0)
    preds = model.predict(batch)[0]
    idx = np.argmax(preds)
    print(f"Prediction: {label_map[str(idx)]}")
    print(f"Confidence: {preds[idx]*100:.2f}%")
    if output_cam:
        heatmap = get_gradcam_heatmap(batch, model, 'conv5_block3_out', pred_index=idx)
        save_and_display_gradcam(image_path, heatmap, cam_path=output_cam)
        print(f"Grad-CAM saved to {output_cam}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="model/pneumonia_model.h5")
    parser.add_argument("--encoder", default="model/label_encoder.json")
    parser.add_argument("--gradcam", default=None)
    args = parser.parse_args()
    run_prediction(args.image, args.model, args.encoder, args.gradcam)