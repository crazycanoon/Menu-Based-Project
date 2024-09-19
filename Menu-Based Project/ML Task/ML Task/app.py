from flask import Flask, request, render_template, send_from_directory, redirect, url_for, jsonify
import os
import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['MODEL_PATH'] = 'model.pkl'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Train and save the model
def train_model():
    data = load_iris()
    X, y = data.data, data.target
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, app.config['MODEL_PATH'])

# Load the trained model
model = joblib.load(app.config['MODEL_PATH'])

# Data processing function
def process_data(file_path):
    df = pd.read_csv(file_path)
    df.fillna(method='ffill', inplace=True)  # Forward fill missing values
    df.drop_duplicates(inplace=True)  # Drop duplicate rows
    processed_path = 'static/processed/processed_data.csv'
    df.to_csv(processed_path, index=False)
    return processed_path

# Function to detect and crop face
def detect_and_crop_face(image_path, output_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        raise ValueError("No face detected")

    (x, y, w, h) = faces[0]
    face = image[y:y+h, x:x+w]
    cv2.imwrite(output_path, face)
    return output_path

# Function to apply filters to images
def apply_filter(image_path, filter_type, output_path):
    image = cv2.imread(image_path)
    
    if filter_type == 'gray':
        filtered_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif filter_type == 'blur':
        filtered_image = cv2.GaussianBlur(image, (15, 15), 0)
    elif filter_type == 'edge':
        filtered_image = cv2.Canny(image, 100, 200)
    else:
        raise ValueError("Unsupported filter type")
    
    cv2.imwrite(output_path, filtered_image)
    return output_path

# Function to generate a custom image
def generate_custom_image(output_path):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[20:80, 20:80] = [255, 0, 0]  # Draw a blue rectangle
    cv2.imwrite(output_path, image)
    return output_path

# Function to apply cool filters
def apply_cool_filter(image_path, filter_type, output_path):
    image = cv2.imread(image_path)

    if filter_type == 'sunglasses':
        sunglasses = cv2.imread('sunglasses.png', -1)
        sx, sy, sw, sh = 0, 0, sunglasses.shape[1], sunglasses.shape[0]
        for (x, y, w, h) in [(50, 50, 150, 50)]:  # Example position
            x_offset, y_offset = x + 10, y + 10
            for i in range(sh):
                for j in range(sw):
                    if sunglasses[i, j, 3] > 0:  # Check alpha channel
                        image[y_offset + i, x_offset + j] = sunglasses[i, j, :3]
        cv2.imwrite(output_path, image)
        
    elif filter_type == 'stars':
        stars = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        image[:100, :100] = stars
        cv2.imwrite(output_path, image)

    else:
        raise ValueError("Unsupported filter type")

    return output_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_form')
def upload_form():
    return render_template('upload_form.html')

@app.route('/process_data_form')
def process_data_form():
    return render_template('process_data_form.html')

@app.route('/apply_filter_form')
def apply_filter_form():
    return render_template('apply_filter_form.html')

@app.route('/generate_custom_image')
def generate_custom_image_form():
    return render_template('generate_custom_image.html')

@app.route('/apply_cool_filter_form')
def apply_cool_filter_form():
    return render_template('apply_cool_filter_form.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        return redirect(url_for('results', filename=file.filename))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

@app.route('/results')
def results():
    filename = request.args.get('filename')
    return render_template('results.html', filename=filename)

@app.route('/process_data', methods=['POST'])
def process_data_route():
    file = request.files['dataset']
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        processed_file = process_data(file_path)
        return redirect(url_for('processed_data', filename='processed_data.csv'))

@app.route('/processed_data/<filename>')
def processed_data(filename):
    return send_from_directory('static/processed', filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/detect_and_crop', methods=['POST'])
def detect_and_crop():
    file = request.files['image']
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        face_path = os.path.join(app.config['PROCESSED_FOLDER'], 'face.jpg')
        detect_and_crop_face(file_path, face_path)
        return redirect(url_for('processed_file', filename='face.jpg'))

@app.route('/apply_filter', methods=['POST'])
def apply_filter_route():
    file = request.files['image']
    filter_type = request.form.get('filter')
    if file and filter_type:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        filter_path = os.path.join(app.config['PROCESSED_FOLDER'], f'{filter_type}_filter.jpg')
        apply_filter(file_path, filter_type, filter_path)
        return redirect(url_for('processed_file', filename=f'{filter_type}_filter.jpg'))

@app.route('/generate_custom_image', methods=['GET'])
def generate_custom_image_route():
    custom_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'custom_image.png')
    generate_custom_image(custom_image_path)
    return redirect(url_for('processed_file', filename='custom_image.png'))

@app.route('/apply_cool_filter', methods=['POST'])
def apply_cool_filter_route():
    file = request.files['image']
    filter_type = request.form.get('cool_filter')
    if file and filter_type:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        cool_filter_path = os.path.join(app.config['PROCESSED_FOLDER'], f'{filter_type}_filter.jpg')
        apply_cool_filter(file_path, filter_type, cool_filter_path)
        return redirect(url_for('processed_file', filename=f'{filter_type}_filter.jpg'))

if __name__ == "__main__":
    train_model()  # Ensure the model is trained before starting the app
    app.run(debug=True)
