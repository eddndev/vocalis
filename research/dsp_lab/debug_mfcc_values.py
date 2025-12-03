import numpy as np
import librosa
import sys

def debug_mfcc():
    sr = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration))
    
    # Generate 150Hz Sine Wave (Amplitude 0.5)
    y = 0.5 * np.sin(2 * np.pi * 150 * t)
    
    # Ensure float32 to match Rust
    y = y.astype(np.float32)

    print(f"Audio Signal (First 5): {y[:5]}")
    print(f"Audio Signal (Max): {np.max(y)}")
    
    # 1. MFCC Extraction (Exact logic from feature_extractor.py)
    # n_mfcc=13, n_fft=1024, hop_length=256
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1024, hop_length=256)
    
    print("\n--- Raw MFCCs (First Frame) ---")
    print(mfccs[:, 0])

    # 2. CMN (Cepstral Mean Normalization)
    mfccs_cmn = mfccs - np.mean(mfccs, axis=1, keepdims=True)
    
    print("\n--- MFCCs after CMN (First Frame) ---")
    print(mfccs_cmn[:, 0])

    # 3. Bag-of-Frames (Mean)
    mfccs_mean = np.mean(mfccs_cmn, axis=1)
    
    print("\n--- Final Averaged MFCCs (Target for Rust) ---")
    print(list(mfccs_mean))

if __name__ == "__main__":
    debug_mfcc()
