import os
import datetime
import uuid
from flask import Flask, request, jsonify, send_from_directory, render_template, url_for
import cv2

# Konstanta
API_KEY = "O8csWqO4aBsCcIfx0dDf"
PROJECT_NAME = "skincare-detection"
VERSION = 1
UPLOAD_FOLDER = './uploads'
DETECTION_RESULT_FOLDER = './detection_result'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTION_RESULT_FOLDER'] = DETECTION_RESULT_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_and_save(image_path, confidence=10, overlap=30):
    # Inisialisasi Roboflow hanya saat digunakan
    from roboflow import Roboflow
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace().project(PROJECT_NAME)
    model = project.version(VERSION).model

    # Buat folder jika belum ada
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DETECTION_RESULT_FOLDER'], exist_ok=True)

    # Buat timestamp dan unique ID untuk penamaan file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]  # Mengambil 8 karakter pertama dari UUID

    # Nama file baru berdasarkan timestamp dan unique ID
    new_filename = f"{timestamp}_{unique_id}.jpg"
    upload_image_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
    prediction_image_path = os.path.join(app.config['DETECTION_RESULT_FOLDER'], f"prediction_{new_filename}")

    # Melakukan inferensi pada gambar lokal
    result = model.predict(image_path, confidence=confidence, overlap=overlap).json()
    print(result)

    # Visualisasi prediksi dan simpan hasil ke folder detection_result
    model.predict(image_path, confidence=confidence, overlap=overlap).save(prediction_image_path)

    # Salin gambar asli ke folder uploads dengan nama baru
    os.rename(image_path, upload_image_path)

    print(f"Input image moved to {upload_image_path}")
    print(f"Prediction saved in {prediction_image_path}")

    return result, upload_image_path, prediction_image_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        result, upload_path, prediction_path = detect_and_save(file_path)
        
        return jsonify({
            'original_image': url_for('uploaded_file', filename=os.path.basename(upload_path)),
            'detected_image': url_for('detection_file', filename=os.path.basename(prediction_path))
        })
    return jsonify({'error': 'Invalid file type'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/detection_result/<filename>')
def detection_file(filename):
    return send_from_directory(app.config['DETECTION_RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
