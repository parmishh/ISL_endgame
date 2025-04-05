import streamlit as st
import joblib
import os
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import base64
import pyttsx3
import time
import threading
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# CONFIG
MODEL_DIR = "saved_models"
DATASET_DIR = "dataset"
NUM_LANDMARKS = 21 * 2 * 3

# Background
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Load dataset folder names as class labels
CLASS_LABELS = sorted([folder for folder in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, folder))])

@st.cache_resource
def load_model(model_name):
    model_path = os.path.join(MODEL_DIR, model_name)
    if model_name.endswith(".pkl"):
        model_data = joblib.load(model_path)
        if isinstance(model_data, tuple) and len(model_data) == 2:
            model, label_encoder = model_data
        else:
            model = model_data
            label_encoder = None
    else:
        from tensorflow.keras.models import load_model as keras_load_model
        model = keras_load_model(model_path)
        label_encoder = None
    return model, label_encoder

def get_combined_hand_box(frame, hand_landmarks_list):
    h, w, _ = frame.shape
    boxes = []
    for hand_landmarks in hand_landmarks_list:
        x_list = [lm.x * w for lm in hand_landmarks.landmark]
        y_list = [lm.y * h for lm in hand_landmarks.landmark]
        xmin, xmax = int(min(x_list)) - 20, int(max(x_list)) + 20
        ymin, ymax = int(min(y_list)) - 20, int(max(y_list)) + 20
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)
        boxes.append((xmin, ymin, xmax, ymax))
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    return (x1, y1, x2, y2)

def extract_landmarks(results):
    landmarks = np.zeros((2, 21, 3), dtype=np.float32)
    for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
        if i >= 2:
            break
        for j, lm in enumerate(hand_landmarks.landmark):
            landmarks[i, j] = [lm.x, lm.y, lm.z]
    return landmarks.flatten().reshape(1, -1)

# Init
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# UI
st.set_page_config(page_title="ISL Detection App", layout="centered")
set_background("Assets/Background.jpg")
st.title("🧠 Indian Sign Language Recognition")
st.markdown("Uses MediaPipe for hand detection and ML models trained for ISL (A-Z, 1–9).")

# Session state
if "sentence" not in st.session_state:
    st.session_state["sentence"] = ""
if "last_append_time" not in st.session_state:
    st.session_state["last_append_time"] = 0

# Model
model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl") or f.endswith(".h5")])
model_name = st.selectbox("Select a Trained Model", model_files)
model, label_encoder = load_model(model_name)
st.success(f"✅ Loaded model: {model_name}")

# Buttons
col1, col2 = st.columns(2)

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

with col1:
    if st.button("🔊 Speak"):
        if st.session_state["sentence"]:
            threading.Thread(target=speak_text, args=(st.session_state["sentence"],)).start()
        else:
            st.warning("Nothing to speak.")

with col2:
    if st.button("🧹 Clear Sentence"):
        st.session_state["sentence"] = ""
        st.success("Sentence cleared!")

prediction_placeholder = st.empty()
info_placeholder = st.empty()

# Streamlit WebRTC integration
class ISLTransformer(VideoTransformerBase):
    def __init__(self):
        self.model, _ = load_model(model_name)
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
        self.last_prediction_time = 0
        self.current_sentence = st.session_state.get("sentence", "")

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        display_text = "No hands detected."
        prediction = ""

        if results.multi_hand_landmarks:
            try:
                x1, y1, x2, y2 = get_combined_hand_box(img, results.multi_hand_landmarks)
                input_data = extract_landmarks(results)

                if input_data.shape[1] == NUM_LANDMARKS:
                    if model_name.endswith(".h5"):
                        raw_pred = np.argmax(self.model.predict(input_data), axis=1)[0]
                    else:
                        raw_pred = self.model.predict(input_data)[0]
                        if hasattr(raw_pred, "__iter__"):
                            raw_pred = raw_pred[0]

                    if isinstance(raw_pred, (int, np.integer)) and raw_pred < len(CLASS_LABELS):
                        prediction = CLASS_LABELS[raw_pred]
                        current_time = time.time()
                        if current_time - self.last_prediction_time > 3:
                            self.current_sentence += prediction
                            self.last_prediction_time = current_time

                        display_text = f"Prediction: {prediction}"
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    else:
                        display_text = f"Unknown class ({raw_pred})"
                else:
                    display_text = "Landmark data incomplete."

            except Exception as e:
                display_text = f"Prediction error: {str(e)}"

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(img, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Safely update Streamlit components outside transform
        prediction_placeholder.markdown(f"### 🔤 {display_text}")
        info_placeholder.markdown(f"### ✏️ Current Sentence: `{self.current_sentence}`")
        st.session_state["sentence"] = self.current_sentence

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.markdown("🖐️ Show your ISL sign in front of the webcam.")
webrtc_streamer(
    key="isl-stream",
    video_transformer_factory=ISLTransformer,
    media_stream_constraints={"video": True, "audio": False},
)

# Reference
st.markdown("---")
st.subheader("🧾 Reference Image")
st.image("Assets/Reference.png", caption="Use this as a guide for signs", use_container_width=True)


































# import streamlit as st
# import joblib
# import os
# import cv2
# import numpy as np
# import mediapipe as mp
# from PIL import Image
# import base64
# import pyttsx3
# import time
# import threading

# # CONFIG
# MODEL_DIR = "saved_models"
# DATASET_DIR = "dataset"
# NUM_LANDMARKS = 21 * 2 * 3

# # Background
# def set_background(image_file):
#     with open(image_file, "rb") as f:
#         encoded = base64.b64encode(f.read()).decode()
#     css = f"""
#     <style>
#     .stApp {{
#         background-image: url("data:image/jpg;base64,{encoded}");
#         background-size: cover;
#         background-position: center;
#         background-repeat: no-repeat;
#     }}
#     </style>
#     """
#     st.markdown(css, unsafe_allow_html=True)

# # Load dataset folder names as class labels
# CLASS_LABELS = sorted([folder for folder in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, folder))])

# @st.cache_resource
# def load_model(model_name):
#     model_path = os.path.join(MODEL_DIR, model_name)
#     if model_name.endswith(".pkl"):
#         model_data = joblib.load(model_path)
#         if isinstance(model_data, tuple) and len(model_data) == 2:
#             model, label_encoder = model_data
#         else:
#             model = model_data
#             label_encoder = None
#     else:
#         from tensorflow.keras.models import load_model as keras_load_model
#         model = keras_load_model(model_path)
#         label_encoder = None
#     return model, label_encoder

# def get_combined_hand_box(frame, hand_landmarks_list):
#     h, w, _ = frame.shape
#     boxes = []
#     for hand_landmarks in hand_landmarks_list:
#         x_list = [lm.x * w for lm in hand_landmarks.landmark]
#         y_list = [lm.y * h for lm in hand_landmarks.landmark]
#         xmin, xmax = int(min(x_list)) - 20, int(max(x_list)) + 20
#         ymin, ymax = int(min(y_list)) - 20, int(max(y_list)) + 20
#         xmin, ymin = max(0, xmin), max(0, ymin)
#         xmax, ymax = min(w, xmax), min(h, ymax)
#         boxes.append((xmin, ymin, xmax, ymax))
#     x1 = min(box[0] for box in boxes)
#     y1 = min(box[1] for box in boxes)
#     x2 = max(box[2] for box in boxes)
#     y2 = max(box[3] for box in boxes)
#     return (x1, y1, x2, y2)

# def extract_landmarks(results):
#     landmarks = np.zeros((2, 21, 3), dtype=np.float32)
#     for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
#         if i >= 2:
#             break
#         for j, lm in enumerate(hand_landmarks.landmark):
#             landmarks[i, j] = [lm.x, lm.y, lm.z]
#     return landmarks.flatten().reshape(1, -1)

# # Init
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# # UI
# st.set_page_config(page_title="ISL Detection App", layout="centered")
# set_background("Assets/Background.jpg")
# st.title("🧠 Indian Sign Language Recognition")
# st.markdown("Uses MediaPipe for hand detection and ML models trained for ISL (A-Z, 1–9).")

# # Session state
# if "sentence" not in st.session_state:
#     st.session_state["sentence"] = ""
# if "last_append_time" not in st.session_state:
#     st.session_state["last_append_time"] = 0

# # Model
# model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl") or f.endswith(".h5")])
# model_name = st.selectbox("Select a Trained Model", model_files)
# model, label_encoder = load_model(model_name)
# st.success(f"✅ Loaded model: {model_name}")

# # Buttons (ALWAYS visible)
# col1, col2 = st.columns(2)

# def speak_text(text):
#     engine = pyttsx3.init()
#     engine.say(text)
#     engine.runAndWait()

# with col1:
#     if st.button("🔊 Speak"):
#         if st.session_state["sentence"]:
#             threading.Thread(target=speak_text, args=(st.session_state["sentence"],)).start()
#         else:
#             st.warning("Nothing to speak.")

# with col2:
#     if st.button("🧹 Clear Sentence"):
#         st.session_state["sentence"] = ""
#         st.success("Sentence cleared!")

# # Camera
# start_camera = st.checkbox("📷 Start Webcam")
# prediction_placeholder = st.empty()
# info_placeholder = st.empty()

# if start_camera:
#     stframe = st.empty()
#     cap = cv2.VideoCapture(0)

#     with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7) as hands:
#         st.markdown("🖐️ Show your ISL sign in front of the webcam.")

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 st.warning("⚠️ Failed to grab frame.")
#                 break

#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = hands.process(frame_rgb)
#             prediction = ""
#             display_text = ""

#             if results.multi_hand_landmarks:
#                 num_hands = len(results.multi_hand_landmarks)
#                 x1, y1, x2, y2 = get_combined_hand_box(frame, results.multi_hand_landmarks)
#                 input_data = extract_landmarks(results)

#                 if input_data.shape[1] == NUM_LANDMARKS:
#                     try:
#                         raw_pred = None
#                         if model_name.endswith(".h5"):
#                             raw_pred = np.argmax(model.predict(input_data), axis=1)[0]
#                         else:
#                             raw_pred = model.predict(input_data)[0]
#                             if hasattr(raw_pred, "__iter__"):
#                                 raw_pred = raw_pred[0]

#                         if isinstance(raw_pred, (int, np.integer)) and raw_pred < len(CLASS_LABELS):
#                             prediction = CLASS_LABELS[raw_pred]

#                             # Wait only before appending, not predicting
#                             current_time = time.time()
#                             if current_time - st.session_state["last_append_time"] > 3:
#                                 st.session_state["sentence"] += prediction
#                                 st.session_state["last_append_time"] = current_time

#                         else:
#                             prediction = f"Unknown class ({raw_pred})"
#                     except Exception as e:
#                         prediction = f"Prediction error: {e}"

#                     hand_type = "🖐️ One hand" if num_hands == 1 else "👐 Two hands"
#                     display_text = f"{hand_type} detected — Prediction: `{prediction}`"
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 else:
#                     display_text = "⚠️ Landmark data incomplete for prediction."

#                 for hand_landmarks in results.multi_hand_landmarks:
#                     mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             else:
#                 display_text = "🙋 No hands detected. Please show your sign."

#             stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
#             prediction_placeholder.markdown(f"### 🔤 {display_text}")
#             info_placeholder.markdown(f"### ✏️ Current Sentence: `{st.session_state['sentence']}`")

#         cap.release()

# # Reference
# st.markdown("---")
# st.subheader("🧾 Reference Image")
# st.image("Assets/Reference.png", caption="Use this as a guide for signs", use_container_width=True)




