# 1. Introducción y Planteamiento del Problema

## 1.1. Contexto

La clasificación de fonemas, específicamente vocales, es un problema fundamental en el procesamiento de voz. Tradicionalmente resuelto mediante modelos ocultos de Markov (HMM) y más recientemente mediante Redes Neuronales Profundas (DNN/CNN), el despliegue de estos sistemas en entornos web ("Edge AI") presenta desafíos únicos.

En la primera iteración de **Vocalis**, se implementó una Red Neuronal Convolucional (CNN) entrenada "End-to-End" sobre espectrogramas de Mel. Si bien este modelo alcanzó una alta precisión (>92%) en datasets controlados (DIMEx100), su rendimiento se degradó significativamente (~50-60%) al ser desplegado en navegadores web reales.

## 1.2. El Problema: La "Caja Negra" y el Desajuste de Dominio

El fallo del enfoque basado en Deep Learning se atribuye a dos factores principales:

1.  **Falta de Robustez (Domain Mismatch):** Las CNNs tienden a aprender características espurias del dataset de entrenamiento (ruido de fondo específico, respuesta de frecuencia del micrófono de estudio) en lugar de las invariantes físicas de la voz. Al cambiar al micrófono de una laptop o celular en una habitación con eco, la distribución de datos cambia drásticamente.
2.  **Indeterminismo:** Pequeñas perturbaciones en la señal de entrada pueden causar cambios impredecibles en la clasificación, lo cual es inaceptable para una aplicación educativa o científica que requiere explicabilidad.

## 1.3. La Solución: Retorno a los Fundamentos (DSP)

Para superar estas limitaciones, este proyecto propone un cambio de paradigma: **volver a la física del sonido**.

La voz humana se produce mediante un mecanismo fuente-filtro bien entendido. Las cuerdas vocales producen una frecuencia fundamental ($F_0$), y el tracto vocal actúa como un filtro resonante que amplifica ciertas frecuencias (Formantes $F_1, F_2, \dots$).

Al modelar explícitamente estos componentes mediante algoritmos de **Procesamiento Digital de Señales (DSP)**, podemos:
*   **Garantizar Explicabilidad:** Sabemos exactamente por qué el sistema clasifica una vocal (basado en la energía espectral y resonancias).
*   **Mejorar la Robustez:** Aplicar técnicas matemáticas específicas (como CMN) para cancelar matemáticamente el efecto del micrófono.
*   **Optimizar el Rendimiento:** Ejecutar algoritmos deterministas y ligeros en lugar de tensores masivos.

Este documento detalla la implementación de esta arquitectura basada en **MFCCs (Mel-Frequency Cepstral Coefficients)** y **SVMs (Support Vector Machines)**.
