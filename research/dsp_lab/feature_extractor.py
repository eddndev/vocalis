import numpy as np
import librosa
import scipy.signal

def get_features(y, sr=16000):
    """
    Extrae F0 (Pitch), Formantes (F1, F2) y MFCCs de una señal de audio.
    Retorna: dictionary con features DSP
    """
    features = {'f0': 0, 'f1': 0, 'f2': 0}
    
    # FIX: Padding para audios muy cortos (< 1024 muestras / 64ms)
    # Esto evita warnings de n_fft y errores de cálculo
    if len(y) < 1024:
        y = librosa.util.fix_length(y, size=1024)
    
    # 1. Extraer Pitch (F0) - Mantenemos esto para la detección de género
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=400, sr=sr, frame_length=1024)
    valid_f0 = f0[voiced_flag]
    if len(valid_f0) > 0:
        features['f0'] = np.nanmedian(valid_f0)
    else:
        features['f0'] = 0

    # 2. Extraer Formantes (F1, F2) - Mantenemos por referencia histórica/comparación
    y_pre = librosa.effects.preemphasis(y)
    if len(y_pre) > 512:
        center = len(y_pre) // 2
        window = y_pre[center-256 : center+256] * scipy.signal.get_window('hamming', 512)
    else:
        window = y_pre
    n_coeff = int(2 + (sr / 1000))
    A = librosa.lpc(window, order=n_coeff)
    roots = np.roots(A)
    roots = [r for r in roots if np.imag(r) >= 0]
    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs = sorted(angles * (sr / (2 * np.pi)))
    formants = [f for f in freqs if f > 200 and f < 4000]
    if len(formants) >= 1: features['f1'] = formants[0]
    if len(formants) >= 2: features['f2'] = formants[1]

    # 3. Extraer MFCCs (La nueva propuesta)
    # n_mfcc=13 es el estándar para voz
    # n_fft=1024, hop_length=256 (16ms)
    # IMPORTANTE: n_mels=40 y htk=True para coincidir EXACTAMENTE con Rust (dsp.rs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_mels=40, htk=True, n_fft=1024, hop_length=256)
    
    # A. Normalización del Canal (CMN) ESTRATEGIA ACTUALIZADA:
    # Para vocales aisladas, CMN local anula la señal. 
    # Usamos RAW MFCCs aquí y confiamos en el StandardScaler global durante el entrenamiento.
    
    # mfccs_cmn = mfccs - np.mean(mfccs, axis=1, keepdims=True) # <-- ELIMINADO
    
    # B. Promedio temporal ("Bag-of-frames") de RAW MFCCs
    mfccs_mean = np.mean(mfccs, axis=1)
    
    # Guardar en el diccionario
    for i in range(13):
        features[f'mfcc_{i}'] = mfccs_mean[i]
        
    return features

def get_syllable_features(y, sr=16000, onset_ratio=0.35, transition_ratio=0.15):
    """
    Extract 39-dimensional feature vector for CV syllables.
    
    Segments the audio into 3 temporal regions:
    - Onset (35%): Captures consonant characteristics
    - Transition (15%): Captures coarticulation between consonant and vowel
    - Nucleus (50%): Captures the vowel
    
    Each region gets 13 MFCCs (averaged over time), resulting in 39 total features.
    
    Args:
        y: Audio signal (numpy array)
        sr: Sample rate (default 16000)
        onset_ratio: Proportion of audio for onset region (default 0.35)
        transition_ratio: Proportion for transition region (default 0.15)
    
    Returns:
        dict with keys: mfcc_onset_0..12, mfcc_trans_0..12, mfcc_nucleus_0..12
    """
    features = {}
    
    # Ensure minimum length for MFCC extraction
    min_samples = 512  # Minimum for n_fft=512
    if len(y) < min_samples * 3:
        y = librosa.util.fix_length(y, size=min_samples * 3)
    
    total_samples = len(y)
    
    # Calculate split points
    onset_end = int(total_samples * onset_ratio)
    trans_end = int(total_samples * (onset_ratio + transition_ratio))
    
    # Ensure each segment has minimum size
    onset_end = max(onset_end, min_samples)
    trans_end = max(trans_end, onset_end + min_samples)
    
    # Split audio into 3 regions
    y_onset = y[:onset_end]
    y_transition = y[onset_end:trans_end]
    y_nucleus = y[trans_end:]
    
    # Pad segments if too short
    if len(y_onset) < min_samples:
        y_onset = librosa.util.fix_length(y_onset, size=min_samples)
    if len(y_transition) < min_samples:
        y_transition = librosa.util.fix_length(y_transition, size=min_samples)
    if len(y_nucleus) < min_samples:
        y_nucleus = librosa.util.fix_length(y_nucleus, size=min_samples)
    
    # MFCC parameters (same as vowel extraction for consistency with Rust)
    mfcc_params = {
        'sr': sr,
        'n_mfcc': 13,
        'n_mels': 40,
        'htk': True,
        'n_fft': 512,  # Smaller for short segments
        'hop_length': 128
    }
    
    # Extract MFCCs for each region
    try:
        mfcc_onset = librosa.feature.mfcc(y=y_onset, **mfcc_params)
        mfcc_trans = librosa.feature.mfcc(y=y_transition, **mfcc_params)
        mfcc_nucleus = librosa.feature.mfcc(y=y_nucleus, **mfcc_params)
        
        # Bag-of-frames (temporal average) for each region
        onset_mean = np.mean(mfcc_onset, axis=1)
        trans_mean = np.mean(mfcc_trans, axis=1)
        nucleus_mean = np.mean(mfcc_nucleus, axis=1)
        
        # Store in feature dict
        for i in range(13):
            features[f'mfcc_onset_{i}'] = onset_mean[i]
            features[f'mfcc_trans_{i}'] = trans_mean[i]
            features[f'mfcc_nucleus_{i}'] = nucleus_mean[i]
            
    except Exception as e:
        # Fallback: return zeros if extraction fails
        for i in range(13):
            features[f'mfcc_onset_{i}'] = 0.0
            features[f'mfcc_trans_{i}'] = 0.0
            features[f'mfcc_nucleus_{i}'] = 0.0
    
    return features


def get_pitch_for_gender(y, sr=16000):
    """
    Extract F0 (pitch) for gender detection.
    Returns median F0 in Hz, or 0 if unvoiced.
    """
    if len(y) < 1024:
        y = librosa.util.fix_length(y, size=1024)
    
    f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr, frame_length=1024)
    valid_f0 = f0[voiced_flag]
    
    if len(valid_f0) > 0:
        return np.nanmedian(valid_f0)
    return 0


# Bloque de prueba rápida
if __name__ == "__main__":
    import sys
    # Generar una onda simple para probar
    sr = 16000
    t = np.linspace(0, 0.5, int(0.5*sr))
    # Simular un tono de 150Hz (Hombre) + armónicos
    y = 0.5 * np.sin(2 * np.pi * 150 * t)
    
    # Test vowel features (original)
    feats = get_features(y, sr)
    print(f"Vowel features (13 dims): {len([k for k in feats if k.startswith('mfcc_')])} MFCCs")
    
    # Test syllable features (new)
    syl_feats = get_syllable_features(y, sr)
    print(f"Syllable features (39 dims): {len(syl_feats)} features")
    print(f"  Onset: {[f'mfcc_onset_{i}' for i in range(3)]}...")
    print(f"  Trans: {[f'mfcc_trans_{i}' for i in range(3)]}...")
    print(f"  Nucleus: {[f'mfcc_nucleus_{i}' for i in range(3)]}...")
