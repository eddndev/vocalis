import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd
from tqdm import tqdm

DATASET_DIR = "train_lab/dataset/audio"
METADATA_FILE = "train_lab/dataset/metadata.csv"
OUTPUT_DIR = "dsp_lab/results"

def compute_average_spectrum(df, label_vowel, label_gender, n_fft=512):
    # Filtrar archivos
    subset = df[(df['label_vowel'] == label_vowel) & (df['label_gender'] == label_gender)]
    files = subset['filename'].tolist()
    
    if not files:
        return None, None

    # Acumulador de espectros
    avg_spec = None
    count = 0
    
    print(f"Procesando {label_vowel} ({label_gender})... {len(files)} muestras.")
    
    # Limitamos a 200 muestras aleatorias para velocidad, si hay muchas
    if len(files) > 200:
        files = np.random.choice(files, 200, replace=False)

    for fname in files:
        path = os.path.join(DATASET_DIR, fname)
        try:
            y, sr = librosa.load(path, sr=16000)
            
            # Pad si es muy corto
            if len(y) < n_fft:
                y = librosa.util.fix_length(y, size=n_fft)

            # Normalizar energía
            y = y / (np.max(np.abs(y)) + 1e-6)
            
            # Calcular FFT
            D = librosa.stft(y, n_fft=n_fft)
            magnitude = np.abs(D)
            
            # Promediar en el tiempo para obtener espectro estático del clip
            mean_mag = np.mean(magnitude, axis=1)
            
            if avg_spec is None:
                avg_spec = mean_mag
            else:
                # Asegurar que tengan la misma longitud (por si acaso n_fft cambió implícitamente)
                if len(mean_mag) == len(avg_spec):
                    avg_spec += mean_mag
            count += 1
        except Exception as e:
            continue

    if count == 0 or avg_spec is None:
        return None, None

    avg_spec /= count
    
    # Frecuencias correspondientes
    freqs = librosa.fft_frequencies(sr=16000, n_fft=n_fft)
    
    return freqs, avg_spec

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Cargando metadatos...")
    if not os.path.exists(METADATA_FILE):
        print("No se encontró el dataset procesado. Por favor ejecuta train_lab/process_dataset.py primero.")
        return
        
    df = pd.read_csv(METADATA_FILE)
    
    vowels = ['a', 'e', 'i', 'o', 'u']
    genders = ['M', 'F']
    
    # 1. Gráfica comparativa de VOCALES (Promedio general)
    plt.figure(figsize=(12, 6))
    for v in vowels:
        # Promedio de M y F combinados para ver la esencia de la vocal
        f_m, s_m = compute_average_spectrum(df, v, 'M')
        f_f, s_f = compute_average_spectrum(df, v, 'F')
        
        if f_m is not None and f_f is not None:
            # Promedio simple de ambos géneros
            avg_s = (s_m + s_f) / 2
            # Convertir a dB
            avg_db = librosa.amplitude_to_db(avg_s, ref=np.max)
            plt.plot(f_m, avg_db, label=f"Vocal '{v}'")

    plt.xlim(0, 4000) # Nos interesan los formantes (0-4kHz)
    plt.title("Espectro Promedio por Vocal (Formantes)")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Amplitud (dB)")
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "vocales_spectrum.png"))
    print("Guardado vocales_spectrum.png")

    # 2. Gráfica comparativa de GÉNERO (Promedio de todas las vocales)
    plt.figure(figsize=(12, 6))
    
    # Acumuladores globales
    total_spec_m = None
    total_spec_f = None
    freqs = None
    
    for v in vowels:
        f, s_m = compute_average_spectrum(df, v, 'M')
        _, s_f = compute_average_spectrum(df, v, 'F')
        
        if s_m is not None:
            total_spec_m = s_m if total_spec_m is None else total_spec_m + s_m
        if s_f is not None:
            total_spec_f = s_f if total_spec_f is None else total_spec_f + s_f
        freqs = f

    # Convertir a dB
    spec_m_db = librosa.amplitude_to_db(total_spec_m, ref=np.max)
    spec_f_db = librosa.amplitude_to_db(total_spec_f, ref=np.max)

    plt.plot(freqs, spec_m_db, label="Masculino (Promedio)", color='blue', alpha=0.8)
    plt.plot(freqs, spec_f_db, label="Femenino (Promedio)", color='red', alpha=0.8)
    
    plt.xlim(0, 1000) # Zoom en frecuencias bajas para ver Pitch fundamental
    plt.title("Comparación de Género (Frecuencias Bajas / Pitch)")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Amplitud (dB)")
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "genero_spectrum.png"))
    print("Guardado genero_spectrum.png")

if __name__ == "__main__":
    main()
