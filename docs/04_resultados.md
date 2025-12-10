# 4. Resultados y Validación Experimental

Este documento presenta los resultados obtenidos tras el entrenamiento de la arquitectura DSP propuesta (MFCC + SVM Dependiente del Género).

> **Estado:** Completado.

## 4.1. Modelo Masculino (SVM-M)

El clasificador optimizado para voces masculinas ha superado todas las expectativas.

### Métricas Globales
*   **Hiperparámetros Óptimos:** `C=100`, `gamma=0.1`.
*   **Precisión Global (Accuracy):** **97.21%**

### Reporte de Clasificación por Clase

| Vocal | Precision | Recall | F1-Score | Muestras (Test) |
| :--- | :---: | :---: | :---: | :---: |
| **a** | 0.96 | 0.98 | **0.97** | 3320 |
| **e** | 0.98 | 0.98 | **0.98** | 3599 |
| **i** | 0.98 | 0.97 | **0.98** | 2229 |
| **o** | 0.97 | 0.97 | **0.97** | 2658 |
| **u** | 0.96 | 0.86 | **0.91** | 791 |

## 4.2. Modelo Femenino (SVM-F)

El modelo femenino mostró un desempeño incluso superior, rozando el 98% de precisión, lo que demuestra la robustez de los MFCCs para capturar los formantes más altos característicos de las voces femeninas.

### Métricas Globales
*   **Hiperparámetros Óptimos:** `C=100`, `gamma=0.1`.
*   **Muestras de Entrenamiento:** ~57,000 clips.
*   **Muestras de Test:** 14,246 clips.
*   **Precisión Global (Accuracy):** **97.87%**

### Reporte de Clasificación por Clase

| Vocal | Precision | Recall | F1-Score | Muestras (Test) |
| :--- | :---: | :---: | :---: | :---: |
| **a** | 0.98 | 0.99 | **0.98** | 3661 |
| **e** | 0.98 | 0.99 | **0.99** | 4208 |
| **i** | 0.98 | 0.98 | **0.98** | 2536 |
| **o** | 0.98 | 0.98 | **0.98** | 2952 |
| **u** | 0.97 | 0.88 | **0.92** | 889 |

## 4.3. Comparativa de Evolución

| Arquitectura | Features | Clasificador | Precisión Aprox. |
| :--- | :--- | :--- | :---: |
| **V1 (Deep Learning)** | Mel-Spectrogram | CNN (ONNX) | ~50% (Web real) |
| **V2 (DSP Básico)** | Formantes (F1, F2) | Árbol de Decisión | 60.17% |
| **V3 (Propuesta Final)** | **MFCCs (13) + CMN** | **SVM (RBF)** | **97.54% (Promedio)** |

## 4.4. Conclusión

La migración a una arquitectura puramente DSP ha resultado en una mejora dramática del rendimiento (+37% respecto a V2). La combinación de **MFCCs con Estandarización Global** y **SVMs dependientes del género** proporciona una solución que es a la vez extremadamente precisa, computacionalmente eficiente y teóricamente sólida.

## 4.5. Desafíos de Implementación y Soluciones Críticas

Durante la fase de integración entre el entorno de investigación (Python) y el motor de producción (Rust/WASM), se identificaron y superaron dos obstáculos técnicos mayores:

### 1. La Paradoja de la Vocal Estacionaria (Fallo de CMN)
*   **Problema:** Inicialmente se utilizó *Cepstral Mean Normalization* (CMN) para normalizar el audio. Se descubrió que para vocales sostenidas (señales estacionarias), la operación $\text{Señal} - \text{Media}(\text{Señal})$ tiende matemáticamente a cero, eliminando la información fonética y generando vectores nulos ("Vectores Zero").
*   **Solución:** Se reemplazó la CMN local por una **Estandarización Global**, utilizando las estadísticas ($\mu, \sigma$) de todo el corpus de entrenamiento para normalizar cada input, preservando así la identidad de la vocal.

### 2. Discrepancia en el Banco de Filtros Mel
*   **Problema:** Se observaron predicciones caóticas en el cliente web a pesar de una alta precisión teórica. La causa raíz fue una divergencia en la configuración del extractor de características:
    *   *Python (Librosa):* Default `n_mels=128`, Escala Slaney.
    *   *Rust (DSP Core):* Configurado `n_mels=40`, Escala HTK.
*   **Solución:** Se alineó el stack configurando explícitamente Python para usar **40 filtros Mel** y la fórmula **HTK**, regenerando el dataset completo para garantizar una correspondencia matemática 1:1 entre entrenamiento e inferencia.

### 3. Alineación Espectral (El "Bug del Factor 2" en FFT)
*   **Problema:** Tras corregir n_mels, la precisión seguía siendo 0% en validación cruzada. Se descubrió mediante un script de depuración comparativo (`validate_model.rs`) que la fórmula usada para mapear frecuencias a bins de FFT en Rust era `bin = freq * N_FFT / (SR/2)`. Esto mapeaba el banco de filtros al doble de su tamaño real (Nyquist en el bin 1025 en vez de 512).
*   **Solución:** Corrección de la fórmula a `bin = freq * N_FFT / SR`. Adicionalmente, se ajustó el ventaneo a **Hann** (para igualar Librosa) y se implementó un "Energy Neutralizer" en inferencia para mitigar diferencias de ganancia de micrófono.
*   **Resultado:** Precisión validada de **80-100%** en muestras de test reales.

