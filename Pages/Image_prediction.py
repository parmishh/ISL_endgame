import streamlit as st
import joblib
import os
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from tensorflow.keras.models import load_model as keras_load_model

MODEL_DIR = "saved_models"
DATASET_DIR = "dataset"
NUM_LANDMARKS = 21 * 2 * 3

# Class label mapping from dataset folder names
CLASS_LABELS = sorted([folder for folder in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, folder))])

# Load model
@st.cache_resource
def load_model(model_name):
    path = os.path.join(MODEL_DIR, model_name)
    if model_name.endswith(".pkl"):
        model_data = joblib.load(path)
        model = model_data[0] if isinstance(model_data, tuple) else model_data
    else:
        model = keras_load_model(path)
    return model

# Extract hand landmarks
def extract_landmarks_from_image(image):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            landmarks = np.zeros((2, 21, 3), dtype=np.float32)
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                for j, lm in enumerate(hand_landmarks.landmark):
                    landmarks[i, j] = [lm.x, lm.y, lm.z]
            return landmarks.flatten().reshape(1, -1)
    return None

# UI
st.set_page_config(page_title="Upload & Predict", layout="centered")
st.title("📤 Upload ISL Image and Predict")

model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl") or f.endswith(".h5")])
model_name = st.selectbox("Select a Model", model_files)
model = load_model(model_name)

uploaded_image = st.file_uploader("Upload Hand Sign Image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

    landmarks = extract_landmarks_from_image(img)

    if landmarks is not None and landmarks.shape[1] == NUM_LANDMARKS:
        if model_name.endswith(".h5"):
            pred = np.argmax(model.predict(landmarks), axis=1)[0]
        else:
            pred = model.predict(landmarks)[0]
            if hasattr(pred, "__iter__"):
                pred = pred[0]

        if isinstance(pred, (int, np.integer)) and pred < len(CLASS_LABELS):
            label = CLASS_LABELS[pred]
            st.success(f"✅ Predicted ISL Symbol: `{label}`")
        else:
            st.warning(f"⚠️ Model predicted unknown class: {pred}")
    else:
        st.error("❌ Could not detect valid hand landmarks in the uploaded image.")
