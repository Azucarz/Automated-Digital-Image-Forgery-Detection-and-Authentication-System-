import os
import json
import cv2
import numpy as np
import tensorflow as tf
import pywt
from skimage.feature import local_binary_pattern
from scipy.fftpack import fft2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.decomposition import PCA

# Paths
DATASET_PATH = "dataset"
REAL_PATH = os.path.join(DATASET_PATH, "Real")
FAKE_PATH = os.path.join(DATASET_PATH, "Fake")
JSON_TRACKER = "Trained/EfficientNetV2L_trained_images.json"
MODEL_PATH = "Trained/EfficientNetV2L.h5"

# Load trained image records
if os.path.exists(JSON_TRACKER):
    with open(JSON_TRACKER, "r") as file:
        trained_images = json.load(file)
else:
    trained_images = {}

# ELA (Error Level Analysis)
def apply_ela(img_path, quality=90):
    temp_path = "temp_ela.jpg"
    img = cv2.imread(img_path)
    cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])

    compressed = cv2.imread(temp_path)
    ela = cv2.absdiff(img, compressed)

    gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0

# Extract features
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (380, 380))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply FFT
    fft_features = np.abs(fft2(gray)).astype(np.float32)

    # Apply Wavelet Transform
    coeffs2 = pywt.dwt2(gray, 'haar')
    cA, (cH, cV, cD) = coeffs2
    wavelet_features = np.concatenate([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()])

    # Apply LBP
    lbp = local_binary_pattern(gray, P=24, R=3, method="uniform")

    # Apply ELA
    ela_image = apply_ela(img_path)

    # Combine features into a single vector
    features = np.concatenate([fft_features.flatten(), wavelet_features, lbp.flatten(), ela_image.flatten()])

    return features

# Load images and labels
def load_dataset():
    images = []
    labels = []

    for category, label in [("Real", 0), ("Fake", 1)]:
        path = os.path.join(DATASET_PATH, category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            if img_name not in trained_images:
                features = preprocess_image(img_path)
                images.append(features)
                labels.append(label)
                trained_images[img_name] = {"path": img_path, "label": label}

    return images, np.array(labels)

# Load new images for retraining
X, y = load_dataset()

if len(X) == 0:
    print("No new images to train. Exiting...")
    exit()

# Convert list to NumPy array
X = np.array(X, dtype=object)

# Ensure all feature vectors have the same length
max_feature_size = max(len(features) for features in X)
X_padded = np.array([np.pad(features, (0, max_feature_size - len(features))) for features in X])

# Apply PCA only if at least 2 samples exist
if X_padded.shape[0] > 1 and X_padded.shape[1] > 1:
    pca_components = min(300, X_padded.shape[0], X_padded.shape[1])  # Ensure valid PCA size
    pca = PCA(n_components=pca_components)
    X_padded = pca.fit_transform(X_padded)
else:
    print("Skipping PCA (not enough samples). Using raw features.")

# Define fully connected neural network
model = Sequential([
    Dense(512, activation="relu", input_shape=(X_padded.shape[1],)),
    Dropout(0.5),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit(X_padded, y, batch_size=16, epochs=10)

# Save updated model
model.save(MODEL_PATH)

# Save updated training records
with open(JSON_TRACKER, "w") as file:
    json.dump(trained_images, file, indent=4)

print("Training completed and model updated successfully.")
