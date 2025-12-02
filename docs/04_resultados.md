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

La migración a una arquitectura puramente DSP ha resultado en una mejora dramática del rendimiento (+37% respecto a V2). La combinación de **MFCCs normalizados por canal (CMN)** y **SVMs dependientes del género** proporciona una solución que es a la vez extremadamente precisa, computacionalmente eficiente y teóricamente sólida.

