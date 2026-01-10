import sounddevice as sd
import numpy as np
import joblib
import librosa
import time
from feature_extractor import get_syllable_features
import warnings

# Suppress sklearn/librosa warnings
warnings.filterwarnings("ignore")

import os

# Configuration
SR = 16000
DURATION = 0.5 # 500ms recording window
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "svm_unified_M.pkl")

def record_audio():
    print(f"\n🎤 RECORDING (Speak 'me', 'sa', 'ma', etc) in 3... 2... 1...")
    time.sleep(1)
    print("🔴 GO!")
    recording = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("⏹️ Done.")
    return recording.flatten()

def predict_syllable(audio):
    # Load model (Pipeline containing Scaler + SVM)
    try:
        clf = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        return

    # Extract Features (Same logic as training)
    print("⚙️ Extracting features in Python (Librosa)...")
    features_dict = get_syllable_features(audio, sr=SR)
    
    # Flatten dictionary to list in correct order (Onset -> Trans -> Nucleus)
    feature_vector = []
    # Onset
    for i in range(13): feature_vector.append(features_dict[f'mfcc_onset_{i}'])
    # Transition
    for i in range(13): feature_vector.append(features_dict[f'mfcc_trans_{i}'])
    # Nucleus
    for i in range(13): feature_vector.append(features_dict[f'mfcc_nucleus_{i}'])

    # Pipeline handles scaling automatically
    feature_vector = [feature_vector] 
    
    # Predict
    pred = clf.predict(feature_vector)[0]
    probs = clf.predict_proba(feature_vector)[0]
    
    # Top 3
    classes = clf.classes_
    top_3 = sorted(zip(classes, probs), key=lambda x: -x[1])[:3]
    
    print("\n" + "="*30)
    print(f"🎯 PREDICTION: {pred.upper()}")
    print("="*30)
    print("Top 3 probabilities:")
    for cls, prob in top_3:
        print(f"  {cls}: {prob*100:.2f}%")
    print("="*30)

if __name__ == "__main__":
    while True:
        audio = record_audio()
        predict_syllable(audio)
        
        cont = input("\nPress ENTER to try again (or 'q' to quit): ")
        if cont.lower() == 'q':
            break
