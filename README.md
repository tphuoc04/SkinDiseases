# Skin Disease Classification Web Application

A Flask-based web application for classifying skin diseases using a deep learning model trained on 23 different skin disease categories.

## Features

- Web-based interface for easy image upload
- Real-time skin disease classification
- Supports 23 different skin disease categories
- Displays prediction confidence score
- User-friendly interface with image preview

## Skin Disease Categories

The model can classify the following 23 skin disease categories:

1. Scabies Lyme Disease and other Infestations and Bites
2. Eczema Photos
3. Warts Molluscum and other Viral Infections
4. Nail Fungus and other Nail Disease
5. Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions
6. Hair Loss Photos Alopecia and other Hair Diseases
7. Bullous Disease Photos
8. Vasculitis Photos
9. Exanthems and Drug Eruptions
10. Lupus and other Connective Tissue diseases
11. Psoriasis pictures Lichen Planus and related diseases
12. Urticaria Hives
13. Poison Ivy Photos and other Contact Dermatitis
14. Vascular Tumors
15. Systemic Disease
16. Melanoma Skin Cancer Nevi and Moles
17. Herpes HPV and other STDs Photos
18. Acne and Rosacea Photos
19. Light Diseases and Disorders of Pigmentation
20. Tinea Ringworm Candidiasis and other Fungal Infections
21. Cellulitis Impetigo and other Bacterial Infections
22. Seborrheic Keratoses and other Benign Tumors
23. Atopic Dermatitis Photos

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tphuoc04/SkinDiseases.git
cd SkinDiseases
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure the trained model file `skin23class.h5` is present in the root directory.

2. Run the Flask application:
```bash
python Flask.py
```

3. Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

4. Upload an image of a skin condition and click "Classify Image" to get the prediction.

## API Endpoints

### GET /
- Returns the main HTML interface for image upload

### POST /upload
- Accepts an image file for classification
- **Request**: Form-data with 'image' field containing the image file
- **Response**: JSON object with prediction results
  ```json
  {
    "success": true,
    "class_name": "Disease name",
    "confidence": 0.95
  }
  ```

## Requirements

- Python 3.8+
- Flask 2.3.0
- TensorFlow 2.13.0
- NumPy 1.24.3
- OpenCV-Python 4.8.1.78

## Model Information

The model is a deep learning model trained on the DermNet dataset with 23 skin disease categories. The model expects input images to be resized to 180x180 pixels.

## Notes

- This is a demonstration application and should not be used as a substitute for professional medical diagnosis.
- Always consult with a healthcare professional for accurate diagnosis and treatment.

## License

This project is provided as-is for educational purposes.
