import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import mediapipe as mp
import av
import os

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="SIBI Hand Gesture Recognition",
    page_icon="👋",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Path dan Variabel Konstan ---
MODEL_PATH = 'model/sibi_model.h5'
CLASS_LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# --- Fungsi Pemuatan Model (Menggunakan Cache) ---
@st.cache_resource
def load_sibi_model(model_path):
    """
    Memuat model Keras dari path yang diberikan.
    Decorator @st.cache_resource memastikan model hanya dimuat sekali saja.
    """
    if not os.path.exists(model_path):
        st.error(f"File model tidak ditemukan di path: {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# --- Kelas Transformer Video untuk WebRTC (100% Mandiri) ---
class SIBITransformer(VideoTransformerBase):
    def __init__(self):
        """
        Inisialisasi semua sumber daya (model, detector) DI DALAM kelas.
        Ini memastikan kelas ini mandiri dan tidak bergantung pada variabel global,
        sehingga aman untuk dijalankan di thread terpisah oleh streamlit-webrtc.
        """
        # 1. Muat model dari dalam kelas. Cache akan mencegah pemuatan berulang.
        self.model = load_sibi_model(MODEL_PATH)
        
        # 2. Inisialisasi MediaPipe Hands
        self.hands_detector = mp.solutions.hands.Hands(
            max_num_hands=4,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        
        # 3. Definisikan utilitas drawing dan label kelas sebagai atribut kelas
        self.mp_drawing = mp.solutions.drawing_utils
        self.class_labels = CLASS_LABELS

    def _preprocess_for_model(self, hand_roi):
        """Helper function untuk preprocess gambar tangan."""
        # Dapatkan ukuran input yang diharapkan model secara dinamis
        input_shape = self.model.input_shape[1:3]  # (height, width)
        
        # Resize dan normalisasi gambar
        resized_roi = cv2.resize(hand_roi, input_shape)
        normalized_roi = resized_roi / 255.0
        
        # Tambahkan dimensi batch
        return np.expand_dims(normalized_roi, axis=0)

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        # Jika model gagal dimuat, tampilkan pesan error pada frame
        if self.model is None:
            img = frame.to_ndarray(format="bgr24")
            cv2.putText(img, "Model Gagal Dimuat!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return img

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Flip seperti cermin
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = self.hands_detector.process(rgb_img)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Gambar landmark tangan
                self.mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                )
                
                # Kalkulasi bounding box dari landmark
                h, w, _ = img.shape
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                
                x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
                
                padding = 30
                x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
                x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)
                
                # Ekstrak ROI (Region of Interest) tangan
                hand_roi = rgb_img[y_min:y_max, x_min:x_max]
                
                if hand_roi.size > 0:
                    # Preprocess gambar dan lakukan prediksi
                    input_data = self._preprocess_for_model(hand_roi)
                    prediction = self.model.predict(input_data)
                    
                    # Dapatkan hasil prediksi
                    confidence = np.max(prediction)
                    predicted_class_index = np.argmax(prediction)
                    label_text = f"{self.class_labels[predicted_class_index]} ({confidence:.2f})"
                    
                    # Tampilkan hasil pada frame video
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(img, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return img

# --- Antarmuka Pengguna (UI) Streamlit ---
def main():
    st.title("👋 SIBI Real-Time Hand Gesture Recognition")
    st.markdown(
        "Aplikasi ini menggunakan CNN untuk mengenali Bahasa Isyarat Indonesia (SIBI) secara real-time. "
        "Arahkan tangan Anda ke kamera untuk melihat prediksi."
    )

    # --- WebRTC Streamer ---
    # Konfigurasi server STUN untuk membantu membangun koneksi P2P
    rtc_config = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]}
        ]
    })

    webrtc_streamer(
        key="sibi-recognizer",
        video_processor_factory=SIBITransformer, # Panggil kelas mandiri kita
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False, # Set ke False untuk stabilitas maksimum
    )

    # --- Tampilkan Gambar-gambar Tambahan ---
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Aturan Tangan SIBI")
        if os.path.exists("gambar/aturan_tangan.jpg"):
            st.image("gambar/aturan_tangan.jpg")
        else:
            st.warning("File 'aturan_tangan.jpg' tidak ditemukan.")

    with col2:
        st.subheader("Confusion Matrix")
        if os.path.exists("gambar/confusion_matrix.png"):
            st.image("gambar/confusion_matrix.png")
        else:
            st.warning("File 'confusion_matrix.png' tidak ditemukan.")
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Kurva Akurasi")
        if os.path.exists("gambar/model_accuracy.png"):
            st.image("gambar/model_accuracy.png")
        else:
            st.warning("File 'model_accuracy.png' tidak ditemukan.")

    with col4:
        st.subheader("Kurva Loss")
        if os.path.exists("gambar/model_loss.png"):
            st.image("gambar/model_loss.png")
        else:
            st.warning("File 'model_loss.png' tidak ditemukan.")

    # --- Sidebar ---
    st.sidebar.title("Tutorial Penggunaan:")
    st.sidebar.info(
        "1. Klik **'Start'** untuk memulai kamera.\n"
        "2. Izinkan akses kamera pada browser Anda.\n"
        "3. Posisikan tangan Anda dengan jelas di depan kamera.\n"
        "4. Lihat hasil prediksi secara langsung pada video."
    )
    st.sidebar.title("Developer")
    st.sidebar.markdown("Muhammad Fauzan Ariyatmoko\nFisika ITS 2021")

if __name__ == "__main__":
    main()