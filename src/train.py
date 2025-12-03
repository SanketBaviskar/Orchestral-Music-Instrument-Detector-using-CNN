import os
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from data_loader import load_data, DataGenerator
from model import build_model
from utils import plot_history

def train(audio_path, epochs=100, batch_size=32):
    # Load file paths and labels
    files, labels = load_data(path=audio_path)
    print(f"Total files: {len(files)}")
    
    # Split data
    X_train_files, X_val_files, y_train_labels, y_val_labels = train_test_split(
        files, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    print(f"Training files: {len(X_train_files)}")
    print(f"Validation files: {len(X_val_files)}")
    
    # Create generators
    train_gen = DataGenerator(X_train_files, y_train_labels, batch_size=batch_size)
    val_gen = DataGenerator(X_val_files, y_val_labels, batch_size=batch_size, shuffle=False)
    
    # Build model
    # Input shape is (1025, 87, 1) based on 1 sec duration, n_fft=2048, hop=512
    input_shape = (1025, 87, 1) 
    num_classes = len(np.unique(labels))
    model = build_model(input_shape, num_classes=num_classes)
    model.summary()
    
    # Callbacks
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    checkpoint = ModelCheckpoint('models/best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, verbose=1, restore_best_weights=True)
    tensorboard = TensorBoard(log_dir='logs')
    
    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, tensorboard]
    )
    
    # Plot history
    plot_history(history)
    
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the Orchestral Instrument Detector')
    parser.add_argument('--path', type=str, default='audio/', help='Path to audio files')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    train(args.path, args.epochs, args.batch_size)
