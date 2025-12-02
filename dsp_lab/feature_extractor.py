import numpy as np
import librosa
import scipy.signal

def get_features(y, sr=16000):
    """
    Extrae F0 (Pitch) y Formantes (F1, F2) de una señal de audio.
    Retorna: dictionary {'f0': float, 'f1': float, 'f2': float}
    """
    features = {'f0': 0, 'f1': 0, 'f2': 0}
    
    # 1. Extraer Pitch (F0) usando autocorrelación (Yin o Pyin)
    # Limitamos el rango a voz humana (50Hz - 400Hz)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=400, sr=sr, frame_length=1024)
    
    # Tomamos la mediana del pitch donde hay voz
    valid_f0 = f0[voiced_flag]
    if len(valid_f0) > 0:
        features['f0'] = np.nanmedian(valid_f0)
    else:
        features['f0'] = 0

    # 2. Extraer Formantes (F1, F2) usando LPC (Linear Predictive Coding)
    # Pre-énfasis para aplanar el espectro
    y_pre = librosa.effects.preemphasis(y)
    
    # Ventana de Hamming sobre la parte central
    if len(y_pre) > 512:
        center = len(y_pre) // 2
        window = y_pre[center-256 : center+256] * scipy.signal.get_window('hamming', 512)
    else:
        window = y_pre
        
    # LPC Order: Regla general = 2 + (sr / 1000)
    # Para 16kHz -> 2 + 16 = 18 coeficientes
    n_coeff = int(2 + (sr / 1000))
    A = librosa.lpc(window, order=n_coeff)
    
    # Raíces del polinomio LPC
    roots = np.roots(A)
    
    # Filtrar raíces: mantener solo las que tienen parte imaginaria positiva
    roots = [r for r in roots if np.imag(r) >= 0]
    
    # Calcular ángulos y frecuencias
    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs = sorted(angles * (sr / (2 * np.pi)))
    
    # Filtrar formantes válidos (generalmente F1 > 200Hz)
    formants = [f for f in freqs if f > 200 and f < 4000]
    
    if len(formants) >= 1:
        features['f1'] = formants[0]
    if len(formants) >= 2:
        features['f2'] = formants[1]
        
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
