import os
import sys

# Set encoding to UTF-8 for stdout
sys.stdout.reconfigure(encoding='utf-8')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO messages

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model 

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation



app = Flask(__name__)

# ******************************** classification  ********************************
# Load the classification model
model_path = "classification_model.keras"
model = load_model(model_path)

class_mappings = {'Glioma': 0, 'Meninigioma': 1, 'Notumor': 2, 'Pituitary': 3}

# Function to preprocess the image for classification
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('L')  # Open image in grayscale mode
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    return img_array

def classify_image(image_path):
    # Preprocess the uploaded image
    img = preprocess_image(image_path)
    # Perform classification
    prediction = model.predict(img)
    # Get the predicted class index
    predicted_class_index = np.argmax(prediction)
    # Get the predicted class
    predicted_class = list(class_mappings.keys())[predicted_class_index]
    # Get the probability of the predicted class
    predicted_probability = prediction[0][predicted_class_index]
    return predicted_class, predicted_probability


# ******************************** Application ********************************
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded file
        imagefile = request.files['imagefile']

        # Check if the file is selected
        if imagefile.filename == '':
            return render_template('index.html', error="No file selected. Please select an image file."), 400

        # Save the file
        image_path = "static/uploads/original.jpg"
        imagefile.save(image_path)

        predicted_class, predicted_probability = classify_image(image_path)


        if predicted_class != "Notumor":
            highlight_tumor(image_path) 

        return render_template('result.html', predicted_class=predicted_class,predicted_probability=predicted_probability)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)