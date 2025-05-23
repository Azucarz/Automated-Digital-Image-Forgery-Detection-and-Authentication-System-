import os
import json
import numpy as np
import cv2
import tensorflow as tf
import pywt
from scipy.fftpack import fft2
from skimage.feature import local_binary_pattern
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# Paths
DATASET_PATH = "dataset"
REAL_PATH = os.path.join(DATASET_PATH, "Real")
FAKE_PATH = os.path.join(DATASET_PATH, "Fake")
TRAINED_IMAGES_JSON = "Trained/VGG19_trained_images.json"
MODEL_PATH = "Trained/VGG19.h5"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load trained images record
if os.path.exists(TRAINED_IMAGES_JSON):
    with open(TRAINED_IMAGES_JSON, "r") as f:
        trained_images = json.load(f)
else:
    trained_images = {}

# Apply Error Level Analysis (ELA)
def apply_ela(image_path, quality=90):
    original = cv2.imread(image_path)
    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, original, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    recompressed = cv2.imread(temp_path)
    ela = cv2.absdiff(original, recompressed)
    return ela

# Apply Fourier Transform (FFT)
def apply_fft(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f_transform = np.log(np.abs(fft2(gray)) + 1)
    return cv2.normalize(f_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Apply Local Binary Patterns (LBP)
def apply_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=24, R=3, method='uniform')
    return lbp.astype(np.uint8)

# Apply Wavelet Transform (WT)
def apply_wavelet(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coeffs2 = pywt.dwt2(gray, 'haar')
    _, (_, _, HH) = coeffs2  # Extract high-frequency components
    return np.abs(HH).astype(np.uint8)

# Apply Spatial Rich Model (SRM)
def apply_srm(image):
    kernel = np.array([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1]
    ])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    srm_output = cv2.filter2D(gray, -1, kernel)
    return srm_output.astype(np.uint8)

# Get new images
def get_new_images():
    new_images = []
    for label, path in [("Real", REAL_PATH), ("Fake", FAKE_PATH)]:
        for img in os.listdir(path):
            if img not in trained_images:
                new_images.append((label, img))
    return new_images

# Get new images
new_images = get_new_images()
if not new_images:
    print("No new images to train. Exiting...")
    exit()

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Training & Validation Data Generators
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# Compute Class Weights
class_labels = train_generator.classes
class_weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=class_labels)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Load Pretrained VGG19 Model
base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Unfreeze Last Few Layers for Fine-Tuning
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Custom Model Architecture
x = Flatten()(base_model.output)
x = BatchNormalization()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output_layer = Dense(1, activation="sigmoid", dtype='float32')(x)

# Compile Model
model = Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
]

# Train Model with Class Weights
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# Save Model
os.makedirs("Trained", exist_ok=True)
model.save(MODEL_PATH)

# Predict and Update Trained Images JSON
for label, img in new_images:
    img_path = os.path.join(DATASET_PATH, label, img)

    # Load and preprocess image
    image = cv2.imread(img_path)
    image = cv2.resize(image, IMG_SIZE)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict using trained model
    prediction = model.predict(image)[0][0]

    # Classify based on threshold
    classified_as = "Real" if prediction < 0.5 else "Fake"

    # Save results
    trained_images[img] = {
        "trained": True,
        "prediction": float(prediction),
        "classified_as": classified_as
    }

# Save JSON File
with open(TRAINED_IMAGES_JSON, "w") as f:
    json.dump(trained_images, f, indent=4)

print(f"Training complete. Model saved to {MODEL_PATH}")
