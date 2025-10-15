import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# -----------------------------
# CONFIG
# -----------------------------
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Make sure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = os.path.join('..', 'notebook', 'covid_model.h5')
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# -----------------------------
# ROUTES
# -----------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Predict
            preds = model.predict(img_array)[0]
            predicted_class = "COVID Positive" if preds[0] > 0.5 else "Normal"
            confidence = round(float(preds[0] * 100 if preds[0] > 0.5 else (1 - preds[0]) * 100), 2)

            return render_template('index.html',
                                   filename=filename,
                                   predicted_class=predicted_class,
                                   confidence=confidence)
    return render_template('index.html', filename=None)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
