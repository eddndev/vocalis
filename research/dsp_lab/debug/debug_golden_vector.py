"""
Debug Script: Golden Vector Extraction
Generates reference MFCC features from a test audio file for comparison with Rust output.
Usage: python debug_golden_vector.py path/to/test_audio.wav
"""

import numpy as np
import librosa
import sys
import joblib

def debug_extraction(wav_path):
    """Extract features with debug output at each step"""
    
    print(f"\n{'='*60}")
    print(f"GOLDEN VECTOR EXTRACTION - {wav_path}")
    print(f"{'='*60}\n")
    
    # 1. Audio Loading
    y, sr = librosa.load(wav_path, sr=16000)
    print(f"[1] Audio Loaded")
    print(f"    Sample Rate: {sr} Hz")
    print(f"    Length: {len(y)} samples ({len(y)/sr:.3f} sec)")
    print(f"    Range: [{y.min():.6f}, {y.max():.6f}]")
    print(f"    First sample: {y[0]:.6f}")
    
    # 2. STFT and Power Spectrum
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=128, window='hann'))**2
    print(f"\n[2] Power Spectrum")
    print(f"    Shape: {S.shape} (freq_bins x frames)")
    print(f"    Bin[0, Frame 0]: {S[0, 0]:.6f}")
    print(f"    Bin[256, Frame 0]: {S[256, 0]:.6f}")
    
    # 3. Mel Filterbank (Slaney normalization)
    mel_basis = librosa.filters.mel(
        sr=16000, 
        n_fft=512, 
        n_mels=40, 
        fmin=0.0, 
        fmax=8000.0, 
        norm='slaney'
    )
    mels = np.dot(mel_basis, S)
    print(f"\n[3] Mel Energies (Slaney norm)")
    print(f"    Shape: {mels.shape} (mels x frames)")
    print(f"    Mel[0, Frame 0]: {mels[0, 0]:.6f}")
    print(f"    Mel[39, Frame 0]: {mels[39, 0]:.6f}")
    
    # 4. Log Mel Spectrogram
    log_mels = 10 * np.log10(mels + 1e-6)
    print(f"\n[4] Log Mel Spectrogram")
    print(f"    Log Mel[0, Frame 0]: {log_mels[0, 0]:.3f} dB")
    print(f"    Log Mel[39, Frame 0]: {log_mels[39, 0]:.3f} dB")
    
    # 5. MFCC with Orthonormal DCT
    mfcc = librosa.feature.mfcc(
        S=log_mels, 
        n_mfcc=13, 
        dct_type=2, 
        norm='ortho'  # CRITICAL
    )
    print(f"\n[5] MFCCs (DCT with norm='ortho')")
    print(f"    Shape: {mfcc.shape} (mfcc_coeffs x frames)")
    print(f"    MFCC[0] (Energy - Frame 0): {mfcc[0, 0]:.3f}")
    print(f"    MFCC[1] (Frame 0): {mfcc[1, 0]:.3f}")
    print(f"    MFCC[12] (Frame 0): {mfcc[12, 0]:.3f}")
    
    # 6. Temporal Averaging (Bag-of-Frames for single segment)
    mfcc_mean = np.mean(mfcc, axis=1)
    print(f"\n[6] Temporal Average (Single Segment)")
    print(f"    Feature Vector (13-dim):")
    for i, val in enumerate(mfcc_mean):
        print(f"    MFCC[{i}]: {val:.6f}")
    
    # 7. Load Scaler (if available)
    try:
        # Adjust path to your model location
        import json
        with open('../dsp_lab/models/vocalis_model.json', 'r') as f:
            model_data = json.load(f)
        
        # Extract male model scaler (for demo)
        scaler_mean = np.array(model_data['unified_male']['scaler']['mean'][:13])
        scaler_scale = np.array(model_data['unified_male']['scaler']['scale'][:13])
        
        # Apply StandardScaler
        scaled = (mfcc_mean - scaler_mean) / scaler_scale
        
        # Neutralize C0
        scaled[0] = 0.0
        
        print(f"\n[7] After StandardScaler + C0 Neutralization")
        print(f"    Scaled Vector (13-dim):")
        for i, val in enumerate(scaled):
            print(f"    Scaled[{i}]: {val:.6f}")
            
    except Exception as e:
        print(f"\n[7] Scaler not loaded: {e}")
        print("    (Run with model to see final scaled values)")
    
    print(f"\n{'='*60}")
    print("Compare these values with Rust console.log() output")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_golden_vector.py <audio_file.wav>")
        sys.exit(1)
    
    wav_file = sys.argv[1]
    debug_extraction(wav_file)
