import os
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# ✅ Folder to save uploaded files
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Load the model (update the path if needed)
MODEL_PATH = os.path.join(os.getcwd(), '..', 'notebook', 'covid_model.h5')

# Check model file
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Define class names
class_names = ['Covid', 'Normal', 'Pneumonia']

def predict_covid(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # adjust size to match training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = round(100 * np.max(predictions[0]), 2)
    return class_names[predicted_index], confidence


@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            predicted_class, confidence = predict_covid(filepath)

            return render_template('index.html',
                                   filename=file.filename,
                                   predicted_class=predicted_class,
                                   confidence=confidence)

    return render_template('index.html', filename=None)


if __name__ == '__main__':
    app.run(debug=True)
