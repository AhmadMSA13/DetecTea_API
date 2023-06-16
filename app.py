import os
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from google.cloud import storage
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import load_img, img_to_array
from waitress import serve
import urllib.request

app = Flask(__name__)
model = load_model('my_model_fix.h5')

class_names = [
    "Algal Leaf",
    "Anthracnose",
    "Bird Eye Spot",
    "Brown Blight",
    "Gray Light",
    "Healthy",
    "Red Leaf Spot",
    "White Spot"
]

def predict_disease(image_url):
    # Download gambar dari URL
    temp_dir = '/tmp'
    os.makedirs(temp_dir, exist_ok=True)
    image_path = os.path.join(temp_dir, 'image.jpg')
    urllib.request.urlretrieve(image_url, image_path)

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = tf.expand_dims(x, 0)

    predictions = model.predict(x)
    predicted_class = tf.argmax(predictions[0])

    predicted_class_name = class_names[predicted_class]
    description = f"This image belongs to class {predicted_class_name}"

    return description, predicted_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file found. Please select a file.'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected!'})
    if file:
        # Inisialisasi klien GCS
        storage_client = storage.Client()
        bucket_name = 'detectea-uploads'

        # Upload file ke GCS
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file.filename)
        blob.upload_from_file(file)

        # Dapatkan URL file yang diunggah
        uploaded_file_url = blob.public_url

        result, predicted_class = predict_disease(uploaded_file_url)
        return jsonify({'result': {'class': class_names[predicted_class], 'description': result}})

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)