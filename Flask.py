from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import sys

# Constants
IMAGE_SIZE = (180, 180)  # Model input size
NORMALIZATION_FACTOR = 255.0  # For normalizing pixel values to [0, 1]

# Load the trained model with error handling
try:
    model = load_model('skin23class.h5')
except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    print("Please ensure 'skin23class.h5' exists in the current directory.", file=sys.stderr)
    sys.exit(1)

# Define class names for the 23 skin disease classes
CLASS_NAMES = [
    'Scabies Lyme Disease and other Infestations and Bites',
    'Eczema Photos',
    'Warts Molluscum and other Viral Infections',
    'Nail Fungus and other Nail Disease',
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
    'Hair Loss Photos Alopecia and other Hair Diseases',
    'Bullous Disease Photos',
    'Vasculitis Photos',
    'Exanthems and Drug Eruptions',
    'Lupus and other Connective Tissue diseases',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Urticaria Hives',
    'Poison Ivy Photos and other Contact Dermatitis',
    'Vascular Tumors',
    'Systemic Disease',
    'Melanoma Skin Cancer Nevi and Moles',
    'Herpes HPV and other STDs Photos',
    'Acne and Rosacea Photos',
    'Light Diseases and Disorders of Pigmentation',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Cellulitis Impetigo and other Bacterial Infections',
    'Seborrheic Keratoses and other Benign Tumors',
    'Atopic Dermatitis Photos'
]

def predict_image(image_file):
    """
    Predict skin disease from an image file
    """
    # Read image from file
    image_bytes = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    
    # Validate image was decoded successfully
    if image is None:
        raise ValueError('Invalid image format or corrupted image file')
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess the image - resize to match model input size
    image = cv2.resize(image, IMAGE_SIZE)
    
    # Convert to tensor and normalize to [0, 1] range
    image = np.array(image, dtype=np.float32) / NORMALIZATION_FACTOR
    
    # Make a prediction
    predictions = model.predict(np.expand_dims(image, axis=0))
    
    # Get the predicted class index and confidence
    predicted_class_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_index])
    
    # Get the class name
    class_name = CLASS_NAMES[predicted_class_index]
    
    return class_name, confidence


app = Flask(__name__)

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']

    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Predict the skin disease
        class_name, confidence = predict_image(image)
        
        return jsonify({
            'success': True,
            'class_name': class_name,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
