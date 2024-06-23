import os
import datetime
import uuid
from shutil import copy2
from roboflow import Roboflow
import matplotlib.pyplot as plt
import cv2

# Konstanta
API_KEY = "O8csWqO4aBsCcIfx0dDf"
PROJECT_NAME = "skincare-detection"
VERSION = 1

def detect_and_save(image_path, confidence=10, overlap=30):
    # Inisialisasi Roboflow dengan API key
    rf = Roboflow(api_key=API_KEY)

    # Ambil proyek dan model berdasarkan endpoint dan versi
    project = rf.workspace().project(PROJECT_NAME)
    model = project.version(VERSION).model

    # Paths
    upload_folder = "./uploads"
    detection_result_folder = "./detection_result"

    # Buat folder jika belum ada
    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(detection_result_folder, exist_ok=True)

    # Buat timestamp dan unique ID untuk penamaan file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]  # Mengambil 8 karakter pertama dari UUID

    # Nama file baru berdasarkan timestamp dan unique ID
    new_filename = f"{timestamp}_{unique_id}.jpg"
    upload_image_path = os.path.join(upload_folder, new_filename)
    prediction_image_path = os.path.join(detection_result_folder, f"prediction_{new_filename}")

    # Melakukan inferensi pada gambar lokal
    result = model.predict(image_path, confidence=confidence, overlap=overlap).json()
    print(result)

    # Visualisasi prediksi dan simpan hasil ke folder detection_result
    model.predict(image_path, confidence=confidence, overlap=overlap).save(prediction_image_path)

    # Salin gambar asli ke folder uploads dengan nama baru
    copy2(image_path, upload_image_path)

    print(f"Input image copied to {upload_image_path}")
    print(f"Prediction saved in {prediction_image_path}")

    # Menampilkan gambar asli dan hasil deteksi
    original_image = cv2.imread(image_path)  # Tetap menggunakan path asli
    detected_image = cv2.imread(prediction_image_path)

    # Konversi warna dari BGR ke RGB untuk matplotlib
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    detected_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)

    # Plot perbandingan gambar asli dan hasil deteksi
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(original_image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(detected_image)
    axs[1].set_title('Detected Image')
    axs[1].axis('off')

    # Menampilkan plot dan memblokir eksekusi script
    plt.show(block=True)

    # Berhenti setelah plot ditampilkan
    return result, upload_image_path, prediction_image_path

# Contoh penggunaan fungsi
result, upload_path, prediction_path = detect_and_save(image_path="test.jpg")
