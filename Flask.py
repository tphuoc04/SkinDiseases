from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import cv2

model = load_model('skin23class.h5')

def predict():
    image = cv2.imread(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Preprocess the image
    image = cv2.resize(image, (180, 180))  # Resize to the input size expected by MobileNetV2
    image = 

    # Make a prediction
    predictions = model.predict(np.expand_dims(image, axis=0))
    decoded_predictions = decode_predictions(predictions, top=1)[0][0]

    # Return the prediction
    return jsonify({'class_name': decoded_predictions[1], 'confidence': float(decoded_predictions[2])})


app = Flask(__name__)

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No image file provided', 400

    image = request.files['image']

    if image.filename == '':
        return 'No selected file', 400
    
    return 'Image uploaded successfully'



if __name__ == '__main__':
    app.run(debug=True)