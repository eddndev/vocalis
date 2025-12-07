# Vocalis: Sistema de Análisis de Voz (Estado Actual)

Este documento describe el estado actual del sistema **Vocalis**, una herramienta para la clasificación de vocales y género a partir de segmentos cortos de audio. Se detallan las etapas de preprocesamiento, la arquitectura de los clasificadores implementados y las observaciones de rendimiento.

## 1. Resumen del Flujo General

El sistema opera completamente en el cliente (navegador web), utilizando JavaScript y WebAssembly. El audio capturado por el micrófono pasa por un pipeline de procesamiento y es analizado por dos módulos de clasificación independientes.

**Flujo General:**
`Micrófono` → `Preprocesamiento JavaScript` → `Visualizador de Espectro (JS)`
                                     ↓
          `Clasificador de Género (JS - DSP)`   →   `Interfaz de Usuario`
                                     ↓
          `Clasificador de Vocal (Solución Actual - ONNX)`

## 2. Preprocesamiento de Audio (Frontend - JavaScript)

El audio capturado por el micrófono del usuario es sometido a un pipeline de procesamiento en tiempo real para estandarizar y limpiar la señal antes de cualquier análisis:

*   **Captura:** `MediaRecorder` de 0.5 segundos de duración.
*   **Downmixing:** Conversión de audio estéreo a mono.
*   **Resampling:** Muestreo a 16,000 Hz.
*   **Filtrado Pasa-Altos (High-Pass):** Filtro IIR (~80 Hz) para eliminar DC offset y ruido grave.
*   **Centrado Inteligente (Smart Energy Centering):** Identificación y centrado del segmento de 50ms con mayor energía para evitar silencios.
*   **Normalización de Pico:** Escalado al 95% de la amplitud máxima.

## 3. Componentes de Clasificación

El sistema Vocalis utiliza actualmente un enfoque híbrido:

### A. Clasificación de Género (Implementación DSP en JavaScript)

Este módulo es el más robusto del sistema. Se basa en el análisis de la **Frecuencia Fundamental (F0)**.

*   **Algoritmo:** Estimación de F0 mediante autocorrelación simplificada en el dominio del tiempo.
*   **Clasificador:** Regla de decisión determinista derivada de un Árbol de Decisión.

```javascript
// Function to estimate F0 (Pitch) from audio data in JavaScript
function getPitch(audioData, sr) {
    // Implementación simplificada de detección de pitch (autocorrelación)
    // Esto es una aproximación, no un pyin completo como librosa,
    // pero debería ser suficiente para la distinción M/F.

    // Parámetros básicos de autocorrelación para pitch
    const FMIN = 50; // Hz
    const FMAX = 400; // Hz
    const HOP_LENGTH = 512;
    const WIN_LENGTH = 2048; // Ventana de análisis
    
    let pitches = [];
    
    // Dividir audio en ventanas
    for (let i = 0; i < audioData.length - WIN_LENGTH; i += HOP_LENGTH) {
        const window = audioData.slice(i, i + WIN_LENGTH);
        
        // Autocorrelación (simplified)
        // Podríamos usar FFT -> Inverse FFT para una mejor autocorrelación,
        // pero una directa es más simple para JS puro.
        const r = new Float32Array(WIN_LENGTH);
        for (let lag = 0; lag < WIN_LENGTH; lag++) {
            for (let j = 0; j < WIN_LENGTH - lag; j++) {
                r[lag] += window[j] * window[j + lag];
            }
        }
        
        // Buscar el pico en el rango de frecuencia deseado
        let maxCorr = -1;
        let bestLag = -1;
        
        // Convertir FMIN/FMAX a lags
        const minLag = Math.floor(sr / FMAX);
        const maxLag = Math.floor(sr / FMIN);

        for (let lag = minLag; lag <= maxLag; lag++) {
            if (r[lag] > maxCorr) {
                maxCorr = r[lag];
                bestLag = lag;
            }
        }
        
        if (bestLag > 0) {
            pitches.push(sr / bestLag);
        }
    }
    
    // Mediana de los pitches encontrados
    if (pitches.length > 0) {
        pitches.sort((a, b) => a - b);
        return pitches[Math.floor(pitches.length / 2)];
    }
    return 0; // No pitch found
}

// Function to classify gender using DSP features (F0)
function classifyGenderDSP(audioData, sr) {
    const f0 = getPitch(audioData, sr);
    console.log('Detected F0 (Pitch):', f0);

    // Decision rule from our trained classifier: f0 <= 178.70
    if (f0 > 0) { // Only classify if a pitch was detected
        if (f0 <= 178.70) {
            return "M (DSP)";
        } else {
            return "F (DSP)";
        }
    } else {
        return "Unknown (DSP - No Pitch)";
    }
}

```

*   **Rendimiento:** Alta precisión (>93%) y estabilidad en pruebas reales.

### B. Clasificación de Vocal (Solución Actual - ONNX/WASM)

Actualmente se utiliza una **Red Neuronal Convolucional (CNN)** pre-entrenada exportada a ONNX.

*   **Entrada:** Mel-Spectrograma (64 Mels).
*   **Estado:** Aunque obtiene >92% de precisión en validación (datos limpios), su rendimiento en el navegador cae drásticamente (~50%).
*   **Diagnóstico:** Falta de robustez ante variaciones del micrófono y ruido ambiental, además de ser una "caja negra" difícil de ajustar paramétricamente.
*   **Acción:** **Se planea descartar esta solución en favor de un enfoque DSP.**

## 4. Visualización de Espectro en Tiempo Real

La interfaz incluye un analizador de espectro en tiempo real (FFT) renderizado en un `canvas` HTML5, lo que permite validar visualmente la presencia de señal y ruido antes de la grabación.

## 5. Análisis y Fundamentación DSP

Para sustentar la migración a un clasificador de vocales puramente DSP, se ha realizado un análisis exhaustivo del corpus **DIMEx100**.

### Extracción de Características (Python)
Hemos desarrollado scripts para extraer F0, F1 y F2 usando `librosa` (Pyin, LPC).

```python

#feature_extractor: 
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

```

### Generación de Dataset DSP
Se procesaron >130,000 clips de audio para generar un dataset tabular `(F0, F1, F2, Vocal, Género)`.

```python
import os
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm
from feature_extractor import get_features

DATASET_DIR = "train_lab/dataset/audio"
METADATA_FILE = "train_lab/dataset/metadata.csv"
OUTPUT_CSV = "dsp_lab/dsp_features.csv"

def main():
    if not os.path.exists(METADATA_FILE):
        print("Metadata no encontrada.")
        return

    df = pd.read_csv(METADATA_FILE)
    print(f"Procesando {len(df)} archivos para extraer F0, F1, F2...")

    results = []
    
    # Procesar cada archivo
    # Usamos tqdm para barra de progreso
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        fname = row['filename']
        path = os.path.join(DATASET_DIR, fname)
        
        try:
            y, sr = librosa.load(path, sr=16000)
            
            # Extraer features DSP
            feats = get_features(y, sr)
            
            results.append({
                'filename': fname,
                'f0': feats['f0'],
                'f1': feats['f1'],
                'f2': feats['f2'],
                'label_vowel': row['label_vowel'],
                'label_gender': row['label_gender']
            })
        except Exception as e:
            # Si falla un archivo (muy corto o corrupto), lo saltamos
            continue
            
    # Guardar DataFrame
    dsp_df = pd.DataFrame(results)
    dsp_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Dataset DSP guardado en {OUTPUT_CSV}")
    print(dsp_df.head())

if __name__ == "__main__":
    main()


```

### Evidencia Visual (Espectros Promedio)
Se generaron gráficas promediando miles de muestras, confirmando que los patrones acústicos (formantes) son distinguibles, aunque con solapamiento.

```python
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

```

## 6. Desafío Actual y Necesidad de Propuesta (Enfoque 100% DSP)

El objetivo actual es **descartar el uso de Redes Neuronales ("cajas negras")** para la clasificación de vocales y consolidar un sistema basado puramente en **Procesamiento Digital de Señales (DSP)**.

El clasificador de vocales basado en F1/F2 (Árbol de Decisión simple) alcanza solo un ~60% de precisión debido a la variabilidad de los hablantes.

**Se busca una propuesta técnica experta para:**

1.  **Mejorar la Extracción de Características:** Ir más allá de F1/F2. ¿Deberíamos incluir F3/F4? ¿Ancho de banda de formantes? ¿Coeficientes Cepstrales (MFCC)?
2.  **Seleccionar un Clasificador Robusto y Explicable:** Reemplazar el árbol simple por algoritmos como **SVM (Support Vector Machines)**, **k-NN** o **GMM (Gaussian Mixture Models)** que modelen mejor el espacio acústico no lineal.
3.  **Estrategia de Normalización:** Técnicas para hacer el sistema invariante al canal (micrófono).

El entregable deseado es una arquitectura DSP que garantice **determinismo, explicabilidad y alta precisión (>95%)** en la clasificación de las 5 vocales.

---
**Recursos Adjuntos:**
*   `dsp_lab/results/vocales_spectrum.png`
*   `dsp_lab/results/genero_spectrum.png`

Prueba de 5 vocales a 10 intentos:

a: aM aM aM aM aM aM uM aM aM aM
 
e: eM eM eM eM oM eM eM iM eM eM

i: iM iM iM iM iM iM iM iF iM iM

o: uM aM oM oM oM uM uM uM uM uM

u: uM uM uM uM uM uM uM uM uM uM