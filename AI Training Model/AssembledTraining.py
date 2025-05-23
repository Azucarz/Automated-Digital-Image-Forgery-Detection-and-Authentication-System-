import os
import json
import numpy as np
import cv2
import tensorflow as tf
import pywt
from skimage.feature import local_binary_pattern
from scipy.fftpack import fft2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, Input, Conv2D, BatchNormalization, Activation, Lambda, Add, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# **Paths**
DATASET_PATH = "dataset"
REAL_PATH = os.path.join(DATASET_PATH, "Real")
FAKE_PATH = os.path.join(DATASET_PATH, "Fake")
TRAINED_IMAGES_JSON = "trained_images.json"
MODEL_PATH = "Trained/model_advanced.h5"

# **Model Parameters**
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20  # Increased for better training stability
LEARNING_RATE = 0.0001

# **Load Trained Image Tracking**
if os.path.exists(TRAINED_IMAGES_JSON):
    with open(TRAINED_IMAGES_JSON, "r") as f:
        trained_images = json.load(f)
else:
    trained_images = {}

def preprocess_image(img_path):
    """Preprocess image using ELA, FFT, Wavelet Transform, and LBP."""
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0  # Normalize

    # **1. Error Level Analysis (ELA)**
    ela_img = cv2.absdiff(img, cv2.GaussianBlur(img, (5, 5), 0))

    # **2. Fourier Transform (FFT)**
    fft_image = np.log(np.abs(fft2(img[:, :, 0])) + 1)
    fft_image = cv2.resize(fft_image, (IMG_SIZE, IMG_SIZE))

    # **3. Wavelet Transform (WT)**
    coeffs2 = pywt.dwt2(img[:, :, 0], 'haar')
    cA, (cH, cV, cD) = coeffs2  # Approximation and detail coefficients
    cA_resized = cv2.resize(cA, (IMG_SIZE, IMG_SIZE))

    # **4. Local Binary Patterns (LBP)**
    lbp = local_binary_pattern(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), P=8, R=1, method="uniform")
    lbp = cv2.resize(lbp, (IMG_SIZE, IMG_SIZE))

    # **Fix shape: Keep only 3 channels**
    processed_img = np.dstack([img[:, :, 0], ela_img[:, :, 0], fft_image])

    return processed_img

def load_dataset():
    """Load images, process them, and return dataset for training"""
    X_train, y_train = [], []

    # Load images from both Real and Fake folders
    for label, folder in enumerate([REAL_PATH, FAKE_PATH]):
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            if img_name in trained_images:
                continue  # Skip already trained images

            processed_img = preprocess_image(img_path)
            if processed_img is not None:
                X_train.append(processed_img)
                y_train.append(label)
                trained_images[img_name] = {"trained": True}

    # Save updated trained images list
    with open(TRAINED_IMAGES_JSON, "w") as f:
        json.dump(trained_images, f, indent=4)

    return np.array(X_train), np.array(y_train)

# **SE-Block (Squeeze-and-Excitation)**
def se_block(input_tensor, ratio=16):
    """Squeeze-and-Excitation Block for Attention Enhancement"""
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Multiply()([input_tensor, se])
    return se

# **Fixed Transformer Block**
def transformer_block(input_tensor):
    """Lightweight Transformer Block with Matching Shape"""
    filters = input_tensor.shape[-1]  # Ensure same number of channels

    conv1 = Conv2D(filters, (3, 3), padding="same")(input_tensor)
    norm1 = BatchNormalization()(conv1)
    act1 = Activation("relu")(norm1)

    conv2 = Conv2D(filters, (3, 3), padding="same")(act1)
    norm2 = BatchNormalization()(conv2)
    act2 = Activation("relu")(norm2)

    shortcut = Add()([input_tensor, act2])  # Shapes now match
    return shortcut

# **Build Advanced Model**
def build_advanced_model():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for layer in base_model.layers:
        layer.trainable = False  # Freeze base model

    x = base_model.output
    x = se_block(x)  # Apply SE Block
    x = transformer_block(x)  # Apply Transformer Block
    x = GlobalAveragePooling2D()(x)

    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model

# **Load dataset**
X_train, y_train = load_dataset()

if len(X_train) > 0:
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Build and compile model
    model = build_advanced_model()
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="binary_crossentropy", metrics=["accuracy"])

    # **Dynamic Learning Rate Scheduling**
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # Train the model
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=[lr_scheduler])

    # Save trained model
    model.save(MODEL_PATH)
    print("✅ Training complete. Model saved as 'Trained/model_advanced.h5'.")
else:
    print("⚠️ No new images found for training. Skipping training.")

### **🔹 Classify and Save Results**
def predict_image(image_path):
    """Classify image and save result"""
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found. Train first!")
        return

    model = load_model(MODEL_PATH)
    img = preprocess_image(image_path)
    if img is None:
        return

    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0]
    result = "Real" if prediction < 0.5 else "Fake"

    trained_images[os.path.basename(image_path)] = {
        "trained": True,
        "prediction": float(prediction),
        "classified_as": result
    }

    with open(TRAINED_IMAGES_JSON, "w") as f:
        json.dump(trained_images, f, indent=4)
    print(f"🟢 {os.path.basename(image_path)} classified as: {result} (Score: {prediction:.4f})")

# **Test all images**
for folder in [REAL_PATH, FAKE_PATH]:
    for img_name in os.listdir(folder):
        predict_image(os.path.join(folder, img_name))

print("✅ Deepfake detection completed.")
