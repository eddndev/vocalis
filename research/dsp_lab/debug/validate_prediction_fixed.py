import librosa
import joblib
import numpy as np
import sys
import os

# Ensure we can import from parent directory
sys.path.append(os.getcwd())
from research.dsp_lab.feature_extractor import get_syllable_features

def test_prediction():
    print("="*60)
    print("TEST: Python Feature Extraction & Prediction")
    print("="*60)

    # 1. Load Model
    model_path = 'research/dsp_lab/models/svm_unified_M.pkl'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model: {model_path}")
    model = joblib.load(model_path)

    # 2. Load Audio
    # audio_path = 'research/dsp_lab/syllable_dataset/audio/s001_M_me_00031.wav'
    # Use the file the user was trying to test
    audio_path = 'research/dsp_lab/syllable_dataset/audio/s001_M_me_00031.wav' 
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        # Try finding ANY wav file to test
        import glob
        wavs = glob.glob('research/dsp_lab/syllable_dataset/audio/*.wav')
        if wavs:
            audio_path = wavs[0]
            print(f"Falling back to: {audio_path}")
        else:
            return

    print(f"Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=16000)

    # 3. Extract Features (Returns a DICT)
    print("Extracting features...")
    features_dict = get_syllable_features(y, sr)
    
    # 4. Convert Dict to List (CRITICAL STEP)
    # The model expects a list of 39 values in specific order:
    # Onset (0-12) -> Transition (0-12) -> Nucleus (0-12)
    feature_vector = []
    
    # Onset
    for i in range(13):
        feature_vector.append(features_dict[f'mfcc_onset_{i}'])
    
    # Transition
    for i in range(13):
        feature_vector.append(features_dict[f'mfcc_trans_{i}'])
        
    # Nucleus
    for i in range(13):
        feature_vector.append(features_dict[f'mfcc_nucleus_{i}'])

    print(f"Feature vector shape: {len(feature_vector)} (Expected: 39)")
    
    print("GOLD STANDARD FEATURES (First 3 per segment + C0s)")
    print(f"Onset C0 (idx 0): {feature_vector[0]:.4f}")
    print(f"Trans C0 (idx 13): {feature_vector[13]:.4f}")
    print(f"Nucl C0  (idx 26): {feature_vector[26]:.4f}")
    print("Full Vector:")
    print(feature_vector)
    
    with open("features_gold.txt", "w") as f:
        f.write(str(list(feature_vector)))
        
    print("="*40)
    
    # 5. Predict
    print("Predicting...")
    # Reshape for sklearn (1 sample, 39 features)
    X = np.array([feature_vector])
    
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    classes = model.classes_

    print("-" * 30)
    print(f"PREDICTION: {pred}")
    print("-" * 30)
    
    # Show probabilities
    top_3_indices = np.argsort(proba)[::-1][:3]
    print("Top 3 probabilities:")
    for idx in top_3_indices:
        print(f"  {classes[idx]}: {proba[idx]*100:.2f}%")

if __name__ == "__main__":
    test_prediction()
