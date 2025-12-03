import argparse
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from data_loader import get_spectrogram

def predict(file_path, model_path='models/best_model.keras'):
    # Load model
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    # Process audio
    spec = get_spectrogram(file_path)
    if spec is None:
        print("Could not process audio file.")
        return
    
    # Add batch dimension: (1, 1025, 87, 1)
    spec = np.expand_dims(spec, axis=0)
    
    # Predict
    prediction = model.predict(spec)
    predicted_index = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_index]
    
    # Map index to label
    # We need the label encoder used during training.
    # Ideally, we should save the label encoder during training.
    # For now, let's assume a fixed list based on the dataset folders if we don't save it.
    # But better to save it. I'll update train.py to save the encoder.
    # For now, I'll hardcode the classes based on the folders I saw earlier.
    # cello, contrabassoon, flute, mandolin, oboe, sax, trumpet, viola
    
    classes = sorted(['cello', 'contrabassoon', 'flute', 'mandolin', 'oboe', 'sax', 'trumpet', 'viola'])
    predicted_label = classes[predicted_index]
    
    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict instrument from audio file')
    parser.add_argument('--file', type=str, required=True, help='Path to audio file')
    parser.add_argument('--model', type=str, default='models/best_model.keras', help='Path to trained model')
    
    args = parser.parse_args()
    
    predict(args.file, args.model)
