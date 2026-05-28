import os
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from PIL import Image
plt.switch_backend('agg')

CLASS_MAP = {0: "Normal", 1: "Bacterial Pneumonia", 2: "Viral Pneumonia"}

def parse_dataset_directory(base_dir):
    images, labels = [], []
    if not os.path.exists(base_dir):
        return images, labels
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            file_path = os.path.join(root, file)
            parent = os.path.basename(root).upper()
            if parent == "NORMAL":
                label = 0
            elif "BACTERIA" in parent or "bacteria" in file.lower():
                label = 1
            elif "VIRUS" in parent or "virus" in file.lower():
                label = 2
            else:
                continue
            images.append(file_path)
            labels.append(label)
    return images, labels

def preprocess_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, tf.one_hot(label, 3)

def create_dataset(file_paths, labels, batch_size=16, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.shuffle(len(file_paths)).map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.15), tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomTranslation(0.1, 0.1), tf.keras.layers.RandomFlip("horizontal")
        ])
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def build_model():
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    out = Dense(3, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    return model

def plot_training_history(history, save_dir="static/images"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.plot(history.history['accuracy'], label='Train Acc', color='#3b82f6')
    plt.plot(history.history['val_accuracy'], label='Val Acc', color='#10b981')
    plt.title('Accuracy')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_history.png"))
    plt.close()
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Train Loss', color='#ef4444')
    plt.plot(history.history['val_loss'], label='Val Loss', color='#f59e0b')
    plt.title('Loss')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_history.png"))
    plt.close()

def main(epochs=10, batch_size=16, dataset_dir="dataset"):
    os.makedirs("model", exist_ok=True)
    train_paths, train_labels = parse_dataset_directory(os.path.join(dataset_dir, "train"))
    val_paths, val_labels = parse_dataset_directory(os.path.join(dataset_dir, "val"))
    test_paths, test_labels = parse_dataset_directory(os.path.join(dataset_dir, "test"))
    if not train_paths:
        raise ValueError("No images found. Run generate_dummy_data.py first.")
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    with open("model/label_encoder.json", "w") as f:
        json.dump(CLASS_MAP, f)
    
    train_ds = create_dataset(train_paths, train_labels, batch_size, augment=True)
    val_ds = create_dataset(val_paths, val_labels, batch_size, augment=False)
    test_ds = create_dataset(test_paths, test_labels, batch_size, augment=False)
    
    model = build_model()
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint("model/pneumonia_model.h5", monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    plot_training_history(history)
    
    # Evaluation
    model = tf.keras.models.load_model("model/pneumonia_model.h5")
    y_true, y_pred, y_pred_prob = [], [], []
    for xb, yb in test_ds:
        probs = model.predict(xb)
        y_pred_prob.extend(probs)
        y_pred.extend(np.argmax(probs, axis=1))
        y_true.extend(np.argmax(yb.numpy(), axis=1))
    y_true = np.array(y_true); y_pred = np.array(y_pred); y_pred_prob = np.array(y_pred_prob)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.savefig("static/images/confusion_matrix.png")
    plt.close()
    
    # ROC curves
    plt.figure(figsize=(8,6))
    for i in range(3):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred_prob[:, i])
        plt.plot(fpr, tpr, label=f"{CLASS_MAP[i]} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curves'); plt.legend()
    plt.savefig("static/images/roc_curve.png")
    plt.close()
    
    print(classification_report(y_true, y_pred, target_names=list(CLASS_MAP.values())))
    print("✅ Training complete. Model saved to model/pneumonia_model.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dataset", type=str, default="dataset")
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.dataset)