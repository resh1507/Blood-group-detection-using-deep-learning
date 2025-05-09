import os
import numpy as np
import cv2
import onnxruntime as ort
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import torch
from torchvision import models

app = Flask(__name__)

# Path to the ONNX model
MODEL_PATH = r"C:\Users\reshh\OneDrive\Desktop\NEW PROJECT\xgboost_model.onnx"  # Update this with your model path
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

# Load ONNX model
onnx_session = ort.InferenceSession(MODEL_PATH)

# Load a pre-trained model (e.g., ResNet) for feature extraction
feature_extractor = models.resnet18(pretrained=True)
feature_extractor.eval()  # Set to evaluation mode

# Image Data Generator for augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Function to check allowed file types
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the uploaded image
def preprocess_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    img = np.array(img) / 255.0  # Normalize
    return img

# Function to augment the image
def augment_image(image):
    return next(datagen.flow(np.expand_dims(image, axis=0), batch_size=1))[0]

# Function to segment the image
def segment_image(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary thresholding to create a binary segmented image
    _, segmented = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Expand dimensions to create a 3D array (H, W, 1) to (H, W, 3)
    segmented_colored = np.stack((segmented,) * 3, axis=-1)  # Duplicate the single channel

    return segmented_colored  # This should now be (H, W, 3)

# Route to serve frontend
@app.route("/")
def index():
    return render_template("code.html")

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

    # Preprocess the uploaded image
    preprocessed_image = preprocess_image(filepath)

    # Augment the image
    augmented_image = augment_image(preprocessed_image)

    # Segment the augmented image
    segmented_image = segment_image(augmented_image)

    # Check the shape of the segmented image
    print("Shape of segmented_image:", segmented_image.shape)

    # Ensure the segmented image is a 3D array
    if segmented_image.ndim != 3:
        return jsonify({"error": "Segmented image does not have the expected dimensions."})

    # Convert the segmented image to float32 and resize for the feature extractor
    segmented_image_float = segmented_image.astype(np.float32)  # Ensure the type is float32
    segmented_image_float = cv2.resize(segmented_image_float, (224, 224))  # Resize to match ResNet input

    # Add batch dimension and change the channel order
    segmented_image_float = np.expand_dims(segmented_image_float, axis=0)  # Add batch dimension
    segmented_image_float = np.transpose(segmented_image_float, (0, 3, 1, 2))  # Change to (N, C, H, W)

    # Extract features using the ResNet model
    with torch.no_grad():
        features = feature_extractor(torch.tensor(segmented_image_float))  # No need to permute again

    # Check the shape of the features
    print("Shape of extracted features:", features.shape)

    # Ensure the features are of the expected shape for the ONNX model
    if features.shape[1] != 512:
        # If the output is not 512, you may need to adjust it
        features = features[:, :512]  # Adjust to match the expected input shape

    # Convert features to numpy and reshape for the ONNX model
    features_np = features.numpy().flatten().reshape(1, -1)  # Flatten to 1D array

    # Prepare input for the ONNX model
    inputs = {onnx_session.get_inputs()[0].name: features_np}

    # Make prediction using the ONNX model
    outputs = onnx_session.run(None, inputs)

    # Assuming the output is a classification probability vector
    predicted_class = np.argmax(outputs[0])

    # Map class index to blood group (Update according to your model)
    blood_groups = ["O+", "A-", "B+", "B-", "AB+", "AB-", "A+", "O-"]
    prediction = blood_groups[predicted_class] if predicted_class < len(blood_groups) else "Unknown"

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)


