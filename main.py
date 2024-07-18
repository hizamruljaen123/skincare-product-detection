import os
import datetime
import uuid
from flask import Flask, request, jsonify, send_from_directory, render_template, url_for
import cv2
from roboflow import Roboflow

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

def detect_and_save(image_path, confidence=3, overlap=30):
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace().project(PROJECT_NAME)
    model = project.version(VERSION).model

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DETECTION_RESULT_FOLDER'], exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]

    new_filename = f"{timestamp}_{unique_id}.jpg"
    upload_image_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
    prediction_image_path = os.path.join(app.config['DETECTION_RESULT_FOLDER'], f"prediction_{new_filename}")

    result = model.predict(image_path, confidence=confidence, overlap=overlap).json()
    print(result)

    if result['predictions']:
        best_prediction = max(result['predictions'], key=lambda x: x['confidence'])

        image = cv2.imread(image_path)
        x, y, w, h = int(best_prediction['x']), int(best_prediction['y']), int(best_prediction['width']), int(best_prediction['height'])
        cv2.rectangle(image, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
        label = f"{best_prediction['class']} ({best_prediction['confidence']*100:.2f}%)"
        cv2.putText(image, label, (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite(prediction_image_path, image)

        os.rename(image_path, upload_image_path)

        print(f"Input image moved to {upload_image_path}")
        print(f"Prediction saved in {prediction_image_path}")

        return result, upload_image_path, prediction_image_path

    else:
        os.rename(image_path, upload_image_path)
        return result, upload_image_path, None

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

        detections = [{
            'class': pred['class'],
            'confidence': f"{pred['confidence'] * 100:.2f}%",
            'x': pred['x'],
            'y': pred['y'],
            'width': pred['width'],
            'height': pred['height']
        } for pred in result['predictions']]
        
        response = {
            'original_image': url_for('uploaded_file', filename=os.path.basename(upload_path)),
            'detections': detections
        }
        if prediction_path:
            response['detected_image'] = url_for('detection_file', filename=os.path.basename(prediction_path))

        return jsonify(response)

    return jsonify({'error': 'Invalid file type'})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/detection_result/<filename>')
def detection_file(filename):
    return send_from_directory(app.config['DETECTION_RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
