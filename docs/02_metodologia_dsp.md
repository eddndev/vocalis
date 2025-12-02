# 2. Metodología DSP y Algoritmos

La arquitectura propuesta se basa en la extracción de características robustas que modelan la percepción auditiva humana y el tracto vocal, seguido de una clasificación jerárquica.

## 2.1. Pipeline de Procesamiento de Señal

El flujo de datos sigue la siguiente secuencia:

$$ \text{Audio} \rightarrow \text{Pre-énfasis} \rightarrow \text{Ventaneo} \rightarrow \text{FFT} \rightarrow \text{Banco de Filtros Mel} \rightarrow \text{Log} \rightarrow \text{DCT} \rightarrow \text{MFCCs} $$

### 2.1.1. Extracción de Características: MFCCs
A diferencia de los formantes crudos ($F_1, F_2$), que son propensos a errores de estimación, utilizamos **Coeficientes Cepstrales en Frecuencia Mel (MFCCs)**.
*   **Configuración:** Se extraen 13 coeficientes estáticos.
*   **Justificación:** Los MFCCs capturan la envolvente espectral completa (el "timbre" de la vocal) de forma compacta y decorrelacionada.

### 2.1.2. Invarianza al Canal: Cepstral Mean Normalization (CMN)
Uno de los mayores desafíos es que cada micrófono imprime su propia "huella" (respuesta de frecuencia) en el audio, lo que se modela como una convolución en el tiempo o una suma en el dominio log-espectral (cepstral).

$$ Y[n] = X[n] * H[n] \implies C_y = C_x + C_h $$

Donde $C_y$ es el MFCC observado, $C_x$ es la voz real y $C_h$ es el micrófono.
Para eliminar $C_h$, aplicamos **CMN**: restamos la media de los vectores MFCC a lo largo del tiempo del clip de audio. Dado que el micrófono es constante ($C_h$ es constante) y la voz varía, la media captura el sesgo del micrófono.

$$ \hat{C}_x = C_y - \mu(C_y) $$

## 2.2. Clasificación Jerárquica Dependiente del Género

Las diferencias fisiológicas entre hombres y mujeres (longitud del tracto vocal) desplazan significativamente las frecuencias de los formantes, creando confusión en un clasificador único.

Para resolver esto, implementamos una estrategia "Divide y Vencerás":

1.  **Nivel 1: Clasificación de Género (Basada en Física)**
    *   Utilizamos la **Frecuencia Fundamental ($F_0$)** extraída mediante el algoritmo de autocorrelación (YIN/Pyin).
    *   **Regla de Decisión:** Si $F_0 \le 178.7 \text{ Hz} \rightarrow \text{Masculino}$, de lo contrario $\rightarrow \text{Femenino}$.
    *   *Precisión:* >93%.

2.  **Nivel 2: Clasificación de Vocal (SVM Especializado)**
    *   Seleccionamos uno de dos modelos expertos: `SVM_Masculino` o `SVM_Femenino`.
    *   Cada modelo es una **Máquina de Vectores de Soporte (SVM)** con Kernel RBF (Radial Basis Function).
    *   Los SVM buscan el hiperplano óptimo que separa las clases de vocales en el espacio de 13 dimensiones de los MFCCs.

## 2.3. Ventajas del Enfoque

*   **Robustez:** La CMN hace que el sistema sea agnóstico al micrófono.
*   **Precisión:** Los modelos especializados por género tienen fronteras de decisión más simples y precisas.
*   **Eficiencia:** El clasificador SVM final es simplemente un cálculo de distancias contra vectores de soporte, extremadamente rápido en tiempo de inferencia.
