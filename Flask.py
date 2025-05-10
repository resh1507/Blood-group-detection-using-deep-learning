import os
import numpy as np
import cv2
import xgboost as xgb
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Paths to models and data

XGB_MODEL_PATH = "xgb_model.json"
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Preprocessed data paths
X_ORIG_PATH = "X_fingerprint.npy"
Y_ORIG_PATH = "y_fingerprint.npy"
X_AUG_PATH = "X_fingerprint_aug.npy"
Y_AUG_PATH = "y_fingerprint_aug.npy"
X_SEG_PATH = "X_fingerprint_segmented.npy"
Y_SEG_PATH = "y_fingerprint_segmented.npy"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load models

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_MODEL_PATH)

# Load preprocessed data
X_fingerprint = np.load(X_ORIG_PATH)
y_fingerprint = np.load(Y_ORIG_PATH)
X_aug = np.load(X_AUG_PATH)
y_aug = np.load(Y_AUG_PATH)
X_segment = np.load(X_SEG_PATH)
y_segment = np.load(Y_SEG_PATH)

# Blood group mapping
blood_groups = ["O+","O-","A+","A-","B+","B-","AB+","AB-"]

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format. Use PNG, JPG, JPEG."})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Load and preprocess image
    image = Image.open(filepath).convert("RGB").resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Segment the image manually if needed (optional)
    image_segmented = image  # assume already segmented for demo

    # Extract features
    
    features = features.reshape(1, -1)

    # Predict with XGBoost
    prediction_proba = xgb_model.predict_proba(features)
    predicted_class = np.argmax(prediction_proba)
    prediction = blood_groups[predicted_class] if predicted_class < len(blood_groups) else "Unknown"

    return jsonify({"prediction": prediction})

@app.route("/test-batch/<stage>")
def test_batch(stage):
    """
    Test prediction from pre-saved .npy files:
    - /test-batch/original
    - /test-batch/augmented
    - /test-batch/segmented
    """
    if stage == "original":
        data = X_fingerprint
        labels = y_fingerprint
    elif stage == "augmented":
        data = X_aug
        labels = y_aug
    elif stage == "segmented":
        data = X_segment
        labels = y_segment
    else:
        return jsonify({"error": "Invalid stage name"})

    results = []
    for i in range(min(5, len(data))):  # Just testing on first 5 samples
        x = np.expand_dims(data[i], axis=0)
        # Assuming the model expects a 1D array of features
        features = x.flatten()
        features = features.reshape(1, -1)
        pred = np.argmax(xgb_model.predict_proba(features))
        true_label = int(labels[i])
        results.append({
            "sample": i,
            "predicted": blood_groups[pred] if pred < len(blood_groups) else "Unknown",
            "actual": blood_groups[true_label] if true_label < len(blood_groups) else "Unknown"
        })

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)

