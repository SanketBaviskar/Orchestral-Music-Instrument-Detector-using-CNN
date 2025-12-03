import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.utils import to_categorical

def load_data(path='data/raw/', sr=44100, duration=None):
    """
    Loads audio files from the specified path and extracts labels.
    """
    files = []
    labels = []
    
    print("Loading audio files from:", path)
    
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.mp3'):
                filepath = os.path.join(root, filename)
                files.append(filepath)
                # Assuming label is the directory name or part of filename
                # Based on notebook: cello_A2_025_forte_arco-normal.mp3 -> cello
                label = filename.split('_')[0]
                labels.append(label)
                
    return np.array(files), np.array(labels)

def get_spectrogram(file, sr=44100, n_fft=2048, hop_length=512, duration=None):
    """
    Computes the spectrogram for an audio file.
    """
    try:
        y, _ = librosa.load(file, sr=sr)
        
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=60)
        
        # Normalize
        y = librosa.util.normalize(y)
        
        # Fix length (pad or truncate)
        # The notebook uses a fixed duration of 1 second (implied by 87 frames for n_fft=2048, hop=512 at 44.1k)
        # 44100 / 512 = ~86.13 frames per second. 
        # Let's use a fixed length of 1 second (44100 samples)
        target_length = int(sr * 1.0)
        y = librosa.util.fix_length(data=y, size=target_length)
        
        # Compute STFT
        y_stft = librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length)
        
        # Convert to dB
        y_spec = librosa.amplitude_to_db(np.abs(y_stft), ref=np.max)
        
        # Expand dims for CNN input (height, width, channels)
        # y_spec shape is (1025, 87)
        y_spec = np.expand_dims(y_spec, axis=-1)
        
        return y_spec
        
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, files, labels, batch_size=32, sr=44100, n_fft=2048, hop_length=512, n_classes=8, shuffle=True):
        self.files = files
        self.labels = labels
        self.batch_size = batch_size
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.y_int = self.label_encoder.fit_transform(self.labels)
        self.y_cat = to_categorical(self.y_int, num_classes=self.n_classes)

    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        files_temp = [self.files[k] for k in indexes]
        y_temp = [self.y_cat[k] for k in indexes]
        
        X, y = self.__data_generation(files_temp, y_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, files_temp, y_temp):
        X = []
        y = []
        
        for i, file in enumerate(files_temp):
            spec = get_spectrogram(file, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
            if spec is not None:
                X.append(spec)
                y.append(y_temp[i])
                
        return np.array(X), np.array(y)


def prepare_datasets(X, y, test_size=0.25, random_state=0):
    """
    Encodes labels and splits data into training and testing sets.
    """
    # Encode labels
    labelencoder = LabelEncoder()
    y_int = labelencoder.fit_transform(y)
    y_cat = to_categorical(y_int)
    
    # Split data
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in sss.split(X, y_int):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_cat[train_index], y_cat[test_index]
        
    return X_train, X_test, y_train, y_test, labelencoder

if __name__ == "__main__":
    # Test the data loader
    files, labels = load_data(path='f:/Projects/Orchestral-Music-Instrument-Detector-using-CNN/audio/')
    print(f"Found {len(files)} files.")
    if len(files) > 0:
        X = extract_features(files[:10]) # Test with first 10
        print(f"Extracted features shape: {X.shape}")
