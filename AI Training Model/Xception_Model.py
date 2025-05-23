import os
import json
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.utils.class_weight import compute_class_weight
from skimage.feature import local_binary_pattern
from scipy.fftpack import fft2
import pywt

# Constants
IMG_SIZE = 299
BATCH_SIZE = 16
EPOCHS = 10
MODEL_PATH = "Trained/Xception.h5"
TRAINED_IMAGES_FILE = "Trained/Xception_trained_images.json"
REAL_PATH = "Dataset/Real"
FAKE_PATH = "Dataset/Fake"

os.makedirs("Trained", exist_ok=True)

if os.path.exists(TRAINED_IMAGES_FILE):
    with open(TRAINED_IMAGES_FILE, "r") as f:
        trained_images = set(json.load(f))
else:
    trained_images = set()


def preprocess_image(image_path):
    """Load image and generate feature channels dynamically"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image could not be read")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0

        channels = [img]  # Start with RGB

        # ELA
        temp_path = "temp_ela.jpg"
        cv2.imwrite(temp_path, img * 255, [cv2.IMWRITE_JPEG_QUALITY, 90])
        ela_img = cv2.absdiff(img, cv2.imread(temp_path).astype(np.float32) / 255.0)
        channels.append(ela_img)

        # FFT
        fft_img = np.abs(fft2(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)))
        fft_img = np.log1p(fft_img)
        fft_img = cv2.resize(fft_img, (IMG_SIZE, IMG_SIZE))
        fft_img = np.expand_dims(fft_img, axis=-1)  # Convert to 3D
        channels.append(fft_img)

        # Wavelet
        coeffs2 = pywt.dwt2(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 'haar')
        cA, (_, _, _) = coeffs2
        wavelet_img = cv2.resize(cA, (IMG_SIZE, IMG_SIZE))
        wavelet_img = np.expand_dims(wavelet_img, axis=-1)  # Convert to 3D
        channels.append(wavelet_img)

        # LBP
        lbp_img = local_binary_pattern(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), P=8, R=1, method="uniform")
        lbp_img = cv2.resize(lbp_img, (IMG_SIZE, IMG_SIZE))
        lbp_img = np.expand_dims(lbp_img, axis=-1)  # Convert to 3D
        channels.append(lbp_img)

        feature_map = np.dstack(channels)  # Stack dynamically

        return feature_map
    except Exception as e:
        print(f"Skipping corrupted image: {image_path} - {str(e)}")
        return None


def load_dataset():
    """Load dataset while skipping already trained images"""
    X_train, y_train = [], []
    new_images = set()
    num_channels = None  # Keep track of consistent channel count

    for label, folder in enumerate([REAL_PATH, FAKE_PATH]):
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)

            if img_name in trained_images:
                continue

            img_array = preprocess_image(img_path)
            if img_array is not None:
                if num_channels is None:
                    num_channels = img_array.shape[-1]  # Set channel count
                elif img_array.shape[-1] != num_channels:
                    print(f"Skipping {img_name} due to mismatched channel count")
                    continue  # Skip inconsistent images

                X_train.append(img_array)
                y_train.append(label)
                new_images.add(img_name)

    if not X_train:
        return None, None, None, None

    return np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32), new_images, num_channels


def build_or_load_model(input_channels):
    """Load or create Xception model with dynamic input channels"""
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        return load_model(MODEL_PATH)

    print(f"Creating new Xception model with {input_channels} channels...")

    input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, input_channels))
    base_model = Xception(include_top=False, weights=None, input_tensor=input_layer)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model():
    """Train or update the model"""
    global trained_images

    X_train, y_train, new_images, input_channels = load_dataset()

    if X_train is None or len(X_train) == 0:
        print("No new images found. Training aborted.")
        return

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

    model = build_or_load_model(input_channels)

    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weights, validation_split=0.2)

    model.save(MODEL_PATH)

    trained_images.update(new_images)
    with open(TRAINED_IMAGES_FILE, "w") as f:
        json.dump(list(trained_images), f)

    print(f"Training complete. Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
