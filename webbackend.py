import os
import json
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from skimage.feature import local_binary_pattern
import pywt

# **Paths**
MODEL_PATH = "AI Training Model/Trained/VGG16.h5"
ANALYSIS_DATA_JSON = "AI Training Model/Trained/Analysis_Data.json"
UPLOAD_FOLDER = "Website/uploads"
SAVED_REAL_FOLDER = "Website/Real"
SAVED_FAKE_FOLDER = "Website/Fake"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

app = Flask(__name__, template_folder="Website")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

for folder in [UPLOAD_FOLDER, SAVED_REAL_FOLDER, SAVED_FAKE_FOLDER]:
    os.makedirs(folder, exist_ok=True)

analysis_data = {}
if os.path.exists(ANALYSIS_DATA_JSON):
    try:
        with open(ANALYSIS_DATA_JSON, "r") as f:
            analysis_data = json.load(f)
    except json.JSONDecodeError:
        print("Corrupt JSON file detected! Skipping JSON loading.")
else:
    print("No JSON file found. Proceeding without stored results.")

def clear_upload_folder():
    for file in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


def detect_and_crop_face(image_path):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No face detected: Using full image for analysis.")
        return img

    x, y, w, h = faces[0]
    print(f"Face detected at (x={x}, y={y}, w={w}, h={h}) - Cropping.")
    return img[y:y + h, x:x + w]

def pixel_level_error_level_analysis(image_path):
    orig = cv2.imread(image_path)
    if orig is None:
        return None

    temp_path = image_path.replace(".", "_ela.")
    cv2.imwrite(temp_path, orig, [cv2.IMWRITE_JPEG_QUALITY, 90])

    compressed = cv2.imread(temp_path)
    diff = cv2.absdiff(orig, compressed)
    ela_matrix = np.mean(diff, axis=2)

    os.remove(temp_path)
    return ela_matrix.flatten()


def fourier_transform(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    f_transform = np.fft.fft2(img)
    f_transform_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_transform_shift) + 1e-10)
    return magnitude_spectrum.flatten()


def wavelet_transform(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    coeffs = pywt.dwt2(img, "haar")
    cA, (cH, cV, cD) = coeffs
    return np.hstack([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()])


def local_binary_patterns(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
    return lbp.flatten()


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = detect_and_crop_face(img_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def extract_features(image_path):
    return {
        "ela_pixels": pixel_level_error_level_analysis(image_path).tolist(),
        "fft_pixels": fourier_transform(image_path).tolist(),
        "wavelet_pixels": wavelet_transform(image_path).tolist(),
        "lbp_pixels": local_binary_patterns(image_path).tolist(),
    }


def get_unique_filename(folder, prefix="scan"):
    count = 1
    while True:
        filename = f"{prefix}_{count:02d}.jpg"
        file_path = os.path.join(folder, filename)
        if not os.path.exists(file_path):
            return filename
        count += 1


def save_classified_image(image_path, classification):
    target_folder = SAVED_REAL_FOLDER if classification == "Real" else SAVED_FAKE_FOLDER
    new_filename = get_unique_filename(target_folder)
    new_path = os.path.join(target_folder, new_filename)
    cv2.imwrite(new_path, cv2.imread(image_path))
    print(f"Image saved as {new_filename} in {target_folder}")


def predict_image(image_path):
    if not os.path.exists(MODEL_PATH):
        return "Error: Model not found!", None, None

    model = load_model(MODEL_PATH)
    img = preprocess_image(image_path)
    if img is None:
        return "Invalid image!", None, None

    new_features = extract_features(image_path)
    filename = os.path.basename(image_path)

    past_result = analysis_data.get(filename, None)

    if past_result:
        if past_result["features"] == new_features:
            print(f"Using stored result for {filename}. No AI inference needed.")
            return past_result["result"], past_result["score"], new_features

    print(f"Running model inference for {filename}...")
    prediction = model.predict(img)[0][0]
    result = "Real" if prediction < 0.6 else "Fake"


    save_classified_image(image_path, result)

    return result, prediction, new_features

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def upload_form():
    clear_upload_folder()
    return render_template("mainweb.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return redirect(url_for("upload_form"))

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("upload_form"))

    filename = file.filename
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    result, score, features = predict_image(filepath)

    return render_template("mainresult.html", filename=filename, result=result, score=score, features=features)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)


#Dharshan A/L Harikrishnan
#TP056582
#Apu assignment 2023 and 2025 . Continued study in 2025 due to postpone for studies in 2024