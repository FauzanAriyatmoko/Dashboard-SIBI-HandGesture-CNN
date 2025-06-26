import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import mediapipe as mp
import av
import os
import queue

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="SIBI Hand Gesture Recognition",
    page_icon="👋",
    layout="wide", # Menggunakan layout 'wide' untuk lebih banyak ruang
    initial_sidebar_state="expanded"
)

# --- Path dan Konstanta ---
MODEL_PATH = 'model/sibi_model.h5' # Pastikan Anda menggunakan model yang kompatibel
CLASS_LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# --- Fungsi Pemuatan Model (Menggunakan Cache) ---
@st.cache_resource
def load_sibi_model(model_path):
    if not os.path.exists(model_path):
        return None
    try:
        model = tf.keras.models.load_model(model_path, compile=False) # compile=False untuk mempercepat pemuatan inferensi
        return model
    except Exception:
        return None

# --- Kelas Transformer Video untuk WebRTC (100% Mandiri) ---
class SIBITransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_sibi_model(MODEL_PATH)
        self.hands_detector = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.class_labels = CLASS_LABELS

    def _preprocess_for_model(self, hand_roi):
        if self.model is None: return None
        input_shape = self.model.input_shape[1:3]
        resized_roi = cv2.resize(hand_roi, input_shape, interpolation=cv2.INTER_AREA)
        normalized_roi = resized_roi / 255.0
        return np.expand_dims(normalized_roi, axis=0)

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")

        if self.model is None:
            cv2.putText(img, "Model Gagal Dimuat!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return img

        img = cv2.flip(img, 1)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands_detector.process(rgb_img)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                h, w, _ = img.shape
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                
                x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
                
                padding = 30
                x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
                x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)
                
                hand_roi = rgb_img[y_min:y_max, x_min:x_max]
                
                if hand_roi.size > 0:
                    input_data = self._preprocess_for_model(hand_roi)
                    prediction = self.model.predict(input_data)
                    confidence = np.max(prediction)
                    predicted_class_index = np.argmax(prediction)
                    label_text = f"{self.class_labels[predicted_class_index]} ({confidence:.2f})"
                    
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(img, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return img

# --- UI Utama ---
def main():
    st.title("👋 SIBI Real-Time Hand Gesture Recognition")
    st.markdown("Aplikasi ini menggunakan CNN untuk mengenali Bahasa Isyarat Indonesia (SIBI) secara real-time. Klik **'Start'** untuk memulai.")

    # Cek apakah model berhasil dimuat sebelum menampilkan komponen WebRTC
    model = load_sibi_model(MODEL_PATH)
    if model is None:
        st.error(f"Model tidak dapat dimuat dari path: {MODEL_PATH}. Pastikan file model ada dan kompatibel.")
        return

    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}, {"urls": ["stun:stun1.l.google.com:19302"]}]})
    
    # PERBAIKAN TERAKHIR: Mengontrol status pemutar video secara eksplisit
    webrtc_ctx = webrtc_streamer(
        key="sibi-recognizer",
        video_processor_factory=SIBITransformer,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,
    )

    if webrtc_ctx.state.playing:
        st.sidebar.success("Kamera aktif dan sedang memproses.")
    else:
        st.sidebar.info("Klik 'Start' untuk memulai deteksi.")

    # Tampilkan gambar-gambar tambahan di bawah
    st.markdown("---")
    st.subheader("Informasi Model dan Aturan Tangan")
    col1, col2 = st.columns(2)
    with col1:
        st.image("gambar/aturan_tangan.jpg", caption="Aturan Tangan SIBI", use_container_width=True)
        st.image("gambar/model_accuracy.png", caption="Kurva Akurasi Model", use_container_width=True)
    with col2:
        st.image("gambar/confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
        st.image("gambar/model_loss.png", caption="Kurva Loss Model", use_container_width=True)
            
    st.sidebar.title("Cara Penggunaan")
    st.sidebar.info("1. Klik **'Start'** untuk memulai.\n2. Izinkan akses kamera.\n3. Posisikan tangan Anda di depan kamera.\n4. Lihat prediksi di layar.")
    st.sidebar.title("Developer")
    st.sidebar.markdown("**Muhammad Fauzan Ariyatmoko**\nFisika ITS 2021")

if __name__ == "__main__":
    main()