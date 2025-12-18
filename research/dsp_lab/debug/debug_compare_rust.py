import librosa
import numpy as np
import os

# Archivo a probar (el mismo que falló en Rust)
TEST_FILE = "research/train_lab/dataset/audio/s001_M_a_0001.wav"

def debug_features():
    if not os.path.exists(TEST_FILE):
        print(f"Error: No existe {TEST_FILE}")
        return

    # Cargar audio
    y, sr = librosa.load(TEST_FILE, sr=16000)
    
    # Extraer MFCCs EXACTAMENTE como en feature_extractor.py
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_mels=40, htk=True, n_fft=1024, hop_length=256)
    
    # Promedio
    mfccs_mean = np.mean(mfccs, axis=1)
    
    print(f"Archivo: {TEST_FILE}")
    print(f"Duración: {len(y)/sr:.2f}s")
    print("-" * 40)
    print("PYTHON MFCCs (Promedio):")
    print(mfccs_mean)
    print("-" * 40)
    print("Primeros 5 valores para comparar con Rust:")
    print(list(mfccs_mean[:5]))

    # Debug adicional: Verificar rango de valores antes de promedio
    print(f"Rango MFCC[0]: {mfccs[0].min():.2f} a {mfccs[0].max():.2f}")

if __name__ == "__main__":
    debug_features()
