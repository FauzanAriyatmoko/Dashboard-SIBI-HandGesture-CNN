# dashboard.py
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import mediapipe as mp
import av

# --- Configuration ---
st.set_page_config(
    page_title="SIBI Hand Gesture Recognition",
    page_icon="👋",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Define the class labels for SIBI gestures
CLASS_LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Path to the saved Keras model
model_path = 'model/sibi_model.h5'

# --- Model Loading ---
@st.cache_resource
def load_sibi_model(model_path):
    """
    Loads the SIBI CNN model from the specified path.
    The @st.cache_resource decorator ensures the model is loaded only once,
    improving performance significantly.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model and cache it
model = load_sibi_model(model_path)

# --- MediaPipe Initialization ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=4,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
mp_drawing = mp.solutions.drawing_utils

# --- WebRTC Video Transformer ---
class SIBITransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.hands_detector = hands

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        if self.model is None:
            # If the model failed to load, return the original frame with an error message
            img = frame.to_ndarray(format="bgr24")
            cv2.putText(img, "Model not loaded!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return img

        img = frame.to_ndarray(format="bgr24")
        
        # Flip the image horizontally for a later selfie-view display
        # This makes it feel more like a mirror.
        img = cv2.flip(img, 1)
        
        # Convert the BGR image to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image and find hands
        results = self.hands_detector.process(rgb_img)
        
        # Draw hand landmarks and process prediction if a hand is detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the landmarks on the image
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                
                # --- Bounding Box Calculation ---
                h, w, _ = img.shape
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]
                
                x_min = int(min(x_coords) * w)
                x_max = int(max(x_coords) * w)
                y_min = int(min(y_coords) * h)
                y_max = int(max(y_coords) * h)
                
                # Add padding to the bounding box
                padding = 30
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                # --- Preprocessing for CNN ---
                # Crop the hand region
                hand_roi = rgb_img[y_min:y_max, x_min:x_max]
                
                if hand_roi.size > 0:
                    # Dynamically get the model's expected input size
                    input_shape = self.model.input_shape
                    # input_shape is (None, height, width, channels)
                    height, width = input_shape[1], input_shape[2]
                    resized_roi = cv2.resize(hand_roi, (width, height))

                    # Normalize pixel values to [0, 1]
                    normalized_roi = resized_roi / 255.0

                    # Expand dimensions to create a batch of 1
                    input_data = np.expand_dims(normalized_roi, axis=0)

                    # --- Prediction ---
                    prediction = self.model.predict(input_data)
                    predicted_class_index = np.argmax(prediction)
                    predicted_class_label = CLASS_LABELS[predicted_class_index]
                    confidence = np.max(prediction)

                    # --- Display Prediction on Frame ---
                    # Draw the bounding box
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Create the label text with class and confidence
                    label_text = f"{predicted_class_label} ({confidence:.2f})"

                    # Put the label text above the bounding box
                    cv2.putText(
                        img, label_text, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
                    )

        return img

# --- Streamlit UI ---
st.title("👋SIBI Real-Time Hand Gesture Recognition")
st.markdown(
    "Dashboard ini menggunakan Convolutional Neural Network (CNN) untuk mengenali gerakan tangan Bahasa Isyarat Indonesia (SIBI) secara real-time. "
    "Taruh tangan Anda dalam bingkai camera pada dashboard untuk melihat prediksi."
)

# --- Developer Info Footer (Bottom-Right) ---
footer_style = """
    <style>
    .custom-footer {
        position: fixed;
        right: 16px;
        bottom: 8px;
        z-index: 100;
        padding: 8px 18px;
        border-radius: 8px;
        font-size: 0.95em;
        text-align: right;
    }
    </style>
    <div class="custom-footer">
        Developed by <b>Muhammad Fauzan Ariyatmoko, Fisika ITS 2021</b><br>
    </div>
"""
st.markdown(footer_style, unsafe_allow_html=True)

if model is not None:
    # RTCConfiguration is used to configure STUN/TURN servers for WebRTC
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    webrtc_streamer(
        key="sibi-recognizer",
        video_transformer_factory=SIBITransformer,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
else:
    st.warning("The model could not be loaded. Please check the model path and file integrity.")

# Display an image from a local file
import os
st.text("Ikuti Aturan Tangan SIBI pada bawah berikut:")
image_path = "gambar/aturan_tangan.jpg"
if os.path.exists(image_path):
    st.image(image_path, use_container_width=True)
else:
    st.warning(f"Gambar aturan tangan tidak ditemukan di: {image_path}")

st.text("Accuracy Curve Model:")
image_path = "gambar/model_accuracy.png"
if os.path.exists(image_path):
    st.image(image_path, use_container_width=True)
else:
    st.warning(f"Gambar akurasi model tidak ditemukan di: {image_path}")

st.text("Loss Curve Model:")
image_path = "gambar/model_loss.png"
if os.path.exists(image_path):
    st.image(image_path, use_container_width=True)
else:
    st.warning(f"Gambar loss model tidak ditemukan di: {image_path}")

st.markdown("Confusion Matrix Model:")
image_path = "gambar/confusion_matrix.png"
if os.path.exists(image_path):
    st.image(image_path, use_container_width=True)
else:
    st.warning(f"Gambar confusion matrix model tidak ditemukan di: {image_path}")

st.sidebar.title("Tutorial Penggunaan:")
st.sidebar.info(
    "1) Pilih Device Camera pada Dashboard dan Klik Done setelah memilih.\n"
    "2) Pastikan anda telah memberikan izin akses kamera pada browser saat muncul notifikasi.\n"
    "3) Klik Start untuk memulai kamera.\n"
    "4) Pastikan tangan Anda berada dalam bingkai kamera.\n"
    "5) Pastikan tangan Anda dalam posisi yang jelas dan stabil.\n"
    "6) Lihat prediksi gerakan tangan di layar."
)
