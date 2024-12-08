import os
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Define the directory for model files (relative to the Flask app)
model_dir = 'models'

# Load the trained models for each disease
disease_models = {
    'Pneumoniadetection': load_model(os.path.join(model_dir, 'my_model.h5')),
}

# Upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def classify_disease(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        return 'Positive'
    else:
        return 'Negative'

@app.route('/', methods=['GET', 'POST'])
def upload_and_display():
    results = {}
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.htm', message='No file part')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.htm', message='No selected file')

        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Perform disease predictions for each selected model
            for disease, model in disease_models.items():
                result = classify_disease(filename, model)
                results[disease] = result

            return render_template('result.html', results=results)

    return render_template('index.htm')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5002)
