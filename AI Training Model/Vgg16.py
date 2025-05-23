import os
import json
import numpy as np
import cv2
import tensorflow as tf
import pywt
from skimage.feature import local_binary_pattern
from scipy.fftpack import fft2, dct
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt


DATASET_PATH = "dataset"
REAL_PATH = os.path.join(DATASET_PATH, "Real")
FAKE_PATH = os.path.join(DATASET_PATH, "Fake")
TRAINED_IMAGES_JSON = "Trained/VGG16_trained_images.json"
STORED_ANALYSIS_JSON = "Trained/Analysis_Data.json"
MODEL_PATH = "Trained/VGG16.h5"
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001


if os.path.exists(STORED_ANALYSIS_JSON):
    with open(STORED_ANALYSIS_JSON, "r") as f:
        analysis_data = json.load(f)
else:
    analysis_data = {}


face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

def detect_face(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    return cv2.resize(face, (IMG_SIZE, IMG_SIZE))

def preprocess_image(img_path):

    img = cv2.imread(img_path)
    if img is None:
        return None, None

    face_img = detect_face(img)
    if face_img is None:
        return None, None

    face_img = face_img.astype("float32") / 255.0  # Normalize


    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray = np.uint8(gray * 255)


    ela_img = cv2.absdiff(face_img, cv2.GaussianBlur(face_img, (5, 5), 0))


    fft_image = np.log(np.abs(fft2(gray)) + 1)
    fft_image = cv2.resize(fft_image, (IMG_SIZE, IMG_SIZE))


    coeffs2 = pywt.dwt2(gray, 'haar')
    cA, _ = coeffs2
    cA_resized = cv2.resize(cA, (IMG_SIZE, IMG_SIZE))


    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp = cv2.resize(lbp, (IMG_SIZE, IMG_SIZE))


    dct_image = dct(gray, type=2)
    dct_image = cv2.resize(dct_image, (IMG_SIZE, IMG_SIZE))


    fft_image = (fft_image - np.min(fft_image)) / (np.max(fft_image) - np.min(fft_image) + 1e-8)
    dct_image = (dct_image - np.min(dct_image)) / (np.max(dct_image) - np.min(dct_image) + 1e-8)
    lbp = (lbp - np.min(lbp)) / (np.max(lbp) - np.min(lbp) + 1e-8)


    processed_img = np.dstack([gray, fft_image, dct_image])


    feature_data = {
        "FFT_mean": float(np.mean(fft_image)),
        "WT_mean": float(np.mean(cA_resized)),
        "LBP_mean": float(np.mean(lbp)),
        "DCT_mean": float(np.mean(dct_image))
    }

    return processed_img, feature_data

def load_dataset(force_retrain=False):

    X, y = [], []

    for label, (folder, class_name) in enumerate([(REAL_PATH, "Real"), (FAKE_PATH, "Fake")]):
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)

            if not force_retrain and img_name in analysis_data:
                continue

            processed_img, feature_data = preprocess_image(img_path)
            if processed_img is not None:
                X.append(processed_img)
                y.append(label)

                analysis_data[img_name] = {
                    "class": class_name,
                    "features": feature_data
                }


    with open(STORED_ANALYSIS_JSON, "w") as f:
        json.dump(analysis_data, f, indent=4)

    return np.array(X), np.array(y)


X, y = load_dataset(force_retrain=False)

if len(X) > 0:
    X = np.array(X)
    y = np.array(y)


    split_1 = int(0.7 * len(X))  # 70% Train
    split_2 = int(0.85 * len(X)) # 15% Validation, 15% Test

    X_train, y_train = X[:split_1], y[:split_1]
    X_val, y_val = X[split_1:split_2], y[split_1:split_2]
    X_test, y_test = X[split_2:], y[split_2:]


    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                                 height_shift_range=0.1, horizontal_flip=True)


    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}


    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE), epochs=EPOCHS, validation_data=(X_val, y_val), class_weight=class_weight_dict)

    model.save(MODEL_PATH)
    print("Training complete. Model saved.")
else:
    print("No new images found for training.")


#Dharshan A/L Harikrishnan
#TP056582
#Apu assignment 2023 and 2025 . Continued study in 2025 due to postpone for studies in 2024