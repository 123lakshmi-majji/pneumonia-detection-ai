import tensorflow as tf

print("Loading model with TF", tf.__version__)
model = tf.keras.models.load_model("model/pneumonia_model.h5")

# Save in H5 format (older, more compatible)
model.save("model/pneumonia_model_fixed.h5", save_format='h5')
print("Model saved as H5 format")

# Replace the old file
import os
os.replace("model/pneumonia_model_fixed.h5", "model/pneumonia_model.h5")
print("✅ Model re-saved in H5 format. Now commit and push this file.")