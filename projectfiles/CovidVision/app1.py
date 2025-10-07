import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = r"C:\projectfiles\CovidVision\notebook\covid_model.h5"
UPLOAD_FOLDER = r"C:\projectfiles\CovidVision\Flask\static\uploads"
TEST_FOLDER = r"C:\projectfiles\CovidVision\Flask\test"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

CLASSES = ['Normal', 'COVID-19', 'Pneumonia']

# Create folders if missing
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

# Initialize Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -------------------------------
# LOAD MODEL
# -------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# -------------------------------
# HELPERS
# -------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]
    return CLASSES[class_idx], round(confidence * 100, 2)

def get_latest_test_image():
    images = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(tuple(ALLOWED_EXTENSIONS))]
    if not images:
        return None
    images = sorted(images, key=lambda x: os.path.getmtime(os.path.join(TEST_FOLDER, x)), reverse=True)
    return os.path.join(TEST_FOLDER, images[0])

# -------------------------------
# ROUTES
# -------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_class = None
    confidence = None
    filename = None

    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '' and allowed_file(file.filename):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                predicted_class, confidence = predict_image(filepath)
                filename = file.filename

    if predicted_class is None:
        test_image = get_latest_test_image()
        if test_image:
            predicted_class, confidence = predict_image(test_image)
            filename = os.path.basename(test_image)

    return render_template('result.html',
                           filename=filename,
                           predicted_class=predicted_class,
                           confidence=confidence)

# -------------------------------
# RUN APP
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
