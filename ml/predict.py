import argparse
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from .data_loader import get_spectrogram

def predict(file_path, model_path='models/best_model.keras'):
    # Load model
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

    # Process audio
    spec = get_spectrogram(file_path)
    if spec is None:
        print("Could not process audio file.")
        return None
    
    # Add batch dimension: (1, 1025, 87, 1)
    spec = np.expand_dims(spec, axis=0)
    
    # Predict
    prediction = model.predict(spec)
    predicted_index = np.argmax(prediction, axis=1)[0]
    confidence = float(prediction[0][predicted_index])
    
    # Map index to label
    classes = sorted(['cello', 'contrabassoon', 'flute', 'mandolin', 'oboe', 'sax', 'trumpet', 'viola'])
    predicted_label = classes[predicted_index]
    
    return {
        "label": predicted_label,
        "confidence": confidence
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict instrument from audio file')
    parser.add_argument('--file', type=str, required=True, help='Path to audio file')
    parser.add_argument('--model', type=str, default='models/best_model.keras', help='Path to trained model')
    
    args = parser.parse_args()
    
    result = predict(args.file, args.model)
    if result:
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.2f}")
