
#Note : if models already exists the code will skip those models and will not train them again.

import os
import cv2
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import mediapipe as mp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Dropout, LSTM, MaxPooling1D, GlobalAveragePooling2D, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate

# CONFIG
DATA_PATH = "dataset"
MODEL_DIR = "saved_models"
CACHE_PATH = "landmark_cache"
NUM_LANDMARKS = 2 * 21 * 3

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)


def extract_landmarks_from_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if not result.multi_hand_landmarks:
        return None
    landmarks = np.zeros((2, 21, 3), dtype=np.float32)
    for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
        if i >= 2:
            break
        for j, lm in enumerate(hand_landmarks.landmark):
            landmarks[i, j] = [lm.x, lm.y, lm.z]
    return landmarks.flatten()


def load_landmark_data(path):
    os.makedirs(CACHE_PATH, exist_ok=True)
    x_path, y_path = os.path.join(CACHE_PATH, "X.npy"), os.path.join(CACHE_PATH, "y.npy")
    if os.path.exists(x_path) and os.path.exists(y_path):
        print("Loaded landmark data from cache.")
        return np.load(x_path), np.load(y_path)

    print("Generating landmark data...")
    X, y = [], []
    labels = sorted(os.listdir(path))
    for label in labels:
        folder = os.path.join(path, label)
        if not os.path.isdir(folder):
            continue
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            features = extract_landmarks_from_image(img_path)
            if features is not None:
                X.append(features)
                y.append(label)
    X, y = np.array(X), np.array(y)
    np.save(x_path, X)
    np.save(y_path, y)
    return X, y


def save_model(model, name, is_keras=False):
    os.makedirs(MODEL_DIR, exist_ok=True)
    ext = "h5" if is_keras else "pkl"
    filepath = os.path.join(MODEL_DIR, f"{name.replace(' ', '_').lower()}.{ext}")
    if is_keras:
        model.save(filepath)
    else:
        joblib.dump(model, filepath)


def model_exists(name, is_keras=False):
    ext = "h5" if is_keras else "pkl"
    return os.path.exists(os.path.join(MODEL_DIR, f"{name.replace(' ', '_').lower()}.{ext}"))


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(y_pred[0], '__iter__'):
        y_pred = np.argmax(y_pred, axis=1)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }


def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Dropout(0.3),
        Conv1D(128, 3, activation='relu'),
        GlobalAveragePooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_lstm(input_shape, num_classes):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_mobilenet(input_shape, num_classes):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=preds)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def prepare_mobilenet_data(path, labels):
    X, y = [], []
    for label in labels:
        folder = os.path.join(path, label)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (224, 224))
                img = img.astype("float32") / 255.0
                X.append(img)
                y.append(label)
    return np.array(X), np.array(y)


def main():
    print("Loading landmark data...", flush=True)
    X, y = load_landmark_data(DATA_PATH)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(),
        "SVM (RBF)": SVC(kernel='rbf', probability=True),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    }

    for name, model in models.items():
        if model_exists(name):
            continue
        print(f"Training {name}...", flush=True)
        model.fit(X_train, y_train)
        save_model(model, name)

    # CNN
    if not model_exists("CNN", is_keras=True):
        try:
            print("Training CNN...", flush=True)
            y_cat = to_categorical(y_enc)
            X_cnn = X.reshape(-1, NUM_LANDMARKS, 1)
            X_train_cnn, _, y_train_cnn, _ = train_test_split(X_cnn, y_cat, test_size=0.2, random_state=42)
            model = build_cnn((NUM_LANDMARKS, 1), y_cat.shape[1])
            model.fit(X_train_cnn, y_train_cnn, epochs=20, batch_size=32, validation_split=0.2, callbacks=[EarlyStopping(patience=3)])
            save_model(model, "CNN", is_keras=True)
        except Exception as e:
            print(f"Error training CNN: {e}", flush=True)

    # LSTM
    if not model_exists("LSTM", is_keras=True):
        try:
            print("Training LSTM...", flush=True)
            y_cat = to_categorical(y_enc)
            X_lstm = X.reshape(-1, NUM_LANDMARKS, 1)
            X_train_lstm, _, y_train_lstm, _ = train_test_split(X_lstm, y_cat, test_size=0.2, random_state=42)
            model = build_lstm((NUM_LANDMARKS, 1), y_cat.shape[1])
            model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32, validation_split=0.2, callbacks=[EarlyStopping(patience=3)])
            save_model(model, "LSTM", is_keras=True)
        except Exception as e:
            print(f"Error training LSTM: {e}", flush=True)

    # # MobileNet
    # if not model_exists("MobileNetV2", is_keras=True):
    #     try:
    #         print("Training MobileNetV2...", flush=True)
    #         labels = sorted(os.listdir(DATA_PATH))
    #         Xm, ym = prepare_mobilenet_data(DATA_PATH, labels)
    #         le_mobilenet = LabelEncoder()
    #         ym_enc = le_mobilenet.fit_transform(ym)
    #         ym_cat = to_categorical(ym_enc)
    #         X_train_m, _, y_train_m, _ = train_test_split(Xm, ym_cat, test_size=0.2, random_state=42)
    #         model = build_mobilenet((224, 224, 3), ym_cat.shape[1])
    #         model.fit(X_train_m, y_train_m, epochs=2, batch_size=32, validation_split=0.2, callbacks=[EarlyStopping(patience=3)])
    #         save_model(model, "MobileNetV2", is_keras=True)
    #     except Exception as e:
    #         print(f"Error training MobileNetV2: {e}", flush=True)

    print("\nEvaluating all saved models...", flush=True)
    eval_results = []

    for model_file in os.listdir(MODEL_DIR):
        model_path = os.path.join(MODEL_DIR, model_file)
        model_name = model_file.rsplit('.', 1)[0].replace('_', ' ').title()

        try:
            if model_file.endswith(".pkl"):
                model = joblib.load(model_path)
                scores = evaluate_model(model, X_test, y_test)

            elif model_file.endswith(".h5"):
                model = load_model(model_path)
                if "mobilenetv2" in model_file:
                    labels = sorted(os.listdir(DATA_PATH))
                    X_mob, y_mob = prepare_mobilenet_data(DATA_PATH, labels)
                    le_m = LabelEncoder()
                    y_mob_enc = le_m.fit_transform(y_mob)
                    y_mob_cat = to_categorical(y_mob_enc)
                    _, X_eval, _, y_eval = train_test_split(X_mob, y_mob_cat, test_size=0.2, random_state=42)
                    scores = evaluate_model(model, X_eval, np.argmax(y_eval, axis=1))
                else:
                    X_eval = X.reshape(-1, NUM_LANDMARKS, 1)
                    y_cat = to_categorical(y_enc)
                    _, X_eval, _, y_eval = train_test_split(X_eval, y_cat, test_size=0.2, random_state=42)
                    scores = evaluate_model(model, X_eval, np.argmax(y_eval, axis=1))

            scores["Model"] = model_name
            eval_results.append(scores)

        except Exception as e:
            print(f"Failed to evaluate {model_name}: {e}")

    print("\nEvaluation Results:")
    print(tabulate(eval_results, headers="keys", tablefmt="fancy_grid"))


if __name__ == "__main__":
    main()
