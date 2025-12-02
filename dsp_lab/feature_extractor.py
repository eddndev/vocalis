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
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1024, hop_length=256)
    
    # A. Normalización del Canal (CMN - Cepstral Mean Normalization)
    # Restamos la media de cada coeficiente a lo largo del tiempo
    # Esto elimina el ruido estacionario (efecto del micrófono)
    mfccs_cmn = mfccs - np.mean(mfccs, axis=1, keepdims=True)
    
    # B. Promedio temporal ("Bag-of-frames")
    # Colapsamos la matriz de tiempo a un vector de 13 valores
    mfccs_mean = np.mean(mfccs_cmn, axis=1)
    
    # Guardar en el diccionario
    for i in range(13):
        features[f'mfcc_{i}'] = mfccs_mean[i]
        
    return features

# Bloque de prueba rápida
if __name__ == "__main__":
    import sys
    # Generar una onda simple para probar
    sr = 16000
    t = np.linspace(0, 0.5, int(0.5*sr))
    # Simular un tono de 150Hz (Hombre) + armónicos
    y = 0.5 * np.sin(2 * np.pi * 150 * t)
    
    feats = get_features(y, sr)
    print(f"Prueba con tono sintético 150Hz: {feats}")
