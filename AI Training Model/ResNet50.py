import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from scipy.fftpack import fft2
import pywt
from skimage.feature import local_binary_pattern

# Paths
DATASET_PATH = "dataset"
REAL_PATH = os.path.join(DATASET_PATH, "Real")
FAKE_PATH = os.path.join(DATASET_PATH, "Fake")
TRAINED_JSON = "Trained/ResNet50_trained_images.json"
MODEL_PATH = "Trained/ResNet50.h5"

# Load previously trained images
if os.path.exists(TRAINED_JSON):
    with open(TRAINED_JSON, "r") as f:
        trained_images = json.load(f)
else:
    trained_images = {}

# ================== ADVANCED FEATURE EXTRACTION ==================

def apply_fft(image):
    """ Apply Fast Fourier Transform (FFT) for frequency analysis """
    f_transform = np.log(1 + np.abs(fft2(image)))
    return cv2.normalize(f_transform, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

def apply_wavelet(image):
    """ Apply Wavelet Transform (WT) and ensure a single-channel output """
    coeffs2 = pywt.dwt2(image, 'haar')
    cA, (cH, cV, _) = coeffs2
    wavelet_image = (cA + cH + cV) / 3  # Convert to single-channel grayscale
    return cv2.normalize(wavelet_image, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

def apply_lbp(image):
    """ Apply Local Binary Pattern (LBP) for texture analysis """
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    return cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

# ================== LOAD IMAGES & APPLY ENHANCED PROCESSING ==================

def load_images():
    """ Load images and apply advanced feature extraction """
    X, y, new_images = [], [], {}

    for label, folder in enumerate([REAL_PATH, FAKE_PATH]):
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            if filename in trained_images:
                continue  # Skip already trained images

            try:
                # Load image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Skipping {filename}: Unable to read image.")
                    continue

                # Resize to ensure consistent shape
                img = cv2.resize(img, (224, 224))

                # Apply feature extraction
                fft_img = apply_fft(img)
                wavelet_img = apply_wavelet(img)
                lbp_img = apply_lbp(img)

                # Ensure all extracted features are (224, 224, 1)
                fft_img = np.expand_dims(cv2.resize(fft_img, (224, 224)), axis=-1)
                wavelet_img = np.expand_dims(cv2.resize(wavelet_img, (224, 224)), axis=-1)
                lbp_img = np.expand_dims(cv2.resize(lbp_img, (224, 224)), axis=-1)

                # Stack features into 3-channel image
                combined_features = np.concatenate([fft_img, wavelet_img, lbp_img], axis=-1)

                # Normalize pixel values
                combined_features = combined_features.astype("float32") / 255.0

                X.append(combined_features)
                y.append(label)
                new_images[filename] = {"trained": False, "classified_as": "Unknown", "prediction": 0.0}

            except Exception as e:
                print(f"Skipping {filename}: {e}")

    return np.array(X), np.array(y), new_images

# ================== TRAINING FUNCTION ==================

def train_model():
    """ Train ResNet50 model with enhanced features """
    X, y, new_images = load_images()

    if len(X) == 0:
        print("No new images to train.")
        return

    # Convert labels to categorical
    y = tf.keras.utils.to_categorical(y, num_classes=2)

    # Define ResNet50 model
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(2, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output)

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Train model
    model.fit(X, y, epochs=10, batch_size=8, validation_split=0.1)

    # Save model
    model.save(MODEL_PATH)

    # Mark images as trained with classification
    for filename, label in zip(new_images.keys(), y):
        class_label = "Real" if np.argmax(label) == 0 else "Fake"
        new_images[filename] = {"trained": True, "classified_as": class_label, "prediction": float(label[1])}

    # Update JSON file
    trained_images.update(new_images)
    with open(TRAINED_JSON, "w") as f:
        json.dump(trained_images, f, indent=4)

    print("Training complete. Model saved.")

# ================== IMAGE PREDICTION FUNCTION ==================

def predict_image(image_path):
    """ Predict if an image is Real or Deepfake """
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Train the model first.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)

    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Skipping {image_path}: Unable to read image.")
        return

    img = cv2.resize(img, (224, 224))

    # Extract features
    fft_img = apply_fft(img)
    wavelet_img = apply_wavelet(img)
    lbp_img = apply_lbp(img)

    # Ensure all features are (224, 224, 1)
    fft_img = np.expand_dims(cv2.resize(fft_img, (224, 224)), axis=-1)
    wavelet_img = np.expand_dims(cv2.resize(wavelet_img, (224, 224)), axis=-1)
    lbp_img = np.expand_dims(cv2.resize(lbp_img, (224, 224)), axis=-1)

    # Stack features
    img_array = np.concatenate([fft_img, wavelet_img, lbp_img], axis=-1)
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    result = "Real" if np.argmax(prediction) == 0 else "Fake"

    # Save results
    trained_images[os.path.basename(image_path)] = {
        "trained": True,
        "classified_as": result,
        "prediction": float(prediction[1])
    }

    with open(TRAINED_JSON, "w") as f:
        json.dump(trained_images, f, indent=4)

    print(f"Image {os.path.basename(image_path)} classified as: {result}")

# ================== RUN TRAINING ==================

if __name__ == "__main__":
    train_model()
