# 2. Metodología DSP y Algoritmos

La arquitectura propuesta se basa en la extracción de características robustas que modelan la percepción auditiva humana y el tracto vocal, seguido de una clasificación jerárquica.

## 2.1. Pipeline de Procesamiento de Señal

El flujo de datos sigue la siguiente secuencia:

$$ \text{Audio} \rightarrow \text{Pre-énfasis} \rightarrow \text{Ventaneo} \rightarrow \text{FFT} \rightarrow \text{Banco de Filtros Mel} \rightarrow \text{Log} \rightarrow \text{DCT} \rightarrow \text{MFCCs Crudos} \rightarrow \text{Estandarización Global} $$

### 2.1.1. Extracción de Características: MFCCs
A diferencia de los formantes crudos ($F_1, F_2$), que son propensos a errores de estimación, utilizamos **Coeficientes Cepstrales en Frecuencia Mel (MFCCs)**.
*   **Configuración:** Se extraen 13 coeficientes estáticos.
*   **Justificación:** Los MFCCs capturan la envolvente espectral completa (el "timbre" de la vocal) de forma compacta y decorrelacionada.

### 2.1.2. Normalización: Estandarización Global vs. CMN
Originalmente se consideró el uso de **CMN (Cepstral Mean Normalization)** para eliminar el sesgo del micrófono. Sin embargo, se identificó un problema crítico específico para la clasificación de vocales sostenidas: la **"Paradoja de la Vocal Estacionaria"**.

*   **El Problema:** En una señal estacionaria (como una "aaaa" larga), la media temporal del vector MFCC es casi idéntica a la señal misma. Al restar la media local ($x_t - \mu_{local}$), la señal resultante tiende a cero, eliminando la información fonética relevante.
*   **La Solución:** Implementamos una **Estandarización Global (Global Standardization)**.
    *   Durante el entrenamiento, se calculan la media ($\mu_{global}$) y desviación estándar ($\sigma_{global}$) de **todo el corpus de voces**.
    *   Cada nuevo vector de entrada $x$ se normaliza contra estas estadísticas fijas, preservando así las diferencias locales de la vocal mientras se centra el espacio de características para el SVM.

$$ z = \frac{x - \mu_{global}}{\sigma_{global}} $$

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
    *   Los SVM buscan el hiperplano óptimo que separa las clases de vocales en el espacio de 13 dimensiones de los MFCCs estandarizados.

## 2.3. Ventajas del Enfoque

*   **Robustez:** La Estandarización Global asegura que la señal nunca se anule, incluso en grabaciones perfectamente estacionarias, solucionando el problema de "vectores cero".
*   **Precisión:** Los modelos especializados por género tienen fronteras de decisión más simples y precisas.
*   **Eficiencia:** El clasificador SVM final es simplemente un cálculo de distancias contra vectores de soporte, extremadamente rápido en tiempo de inferencia.

## 2.4. Interpretación Visual y Justificación Determinística

Para trascender la "caja negra" de la probabilidad y ofrecer una **justificación determinística** al usuario, se introducen las Gráficas de Trayectorias MFCC.

### 2.4.1. Estructura de la Gráfica
Visualizamos el vector de características de 39 dimensiones no como un espectro de frecuencias (Hz), sino como una **"Firma Acústica"** dividida en 3 zonas temporales:

| Zona | Índices | Descripción | Interpretación |
|:---:|:---:|:---|:---|
| **Ataque (Onset)** | 0-12 | Inicio del sonido (35%) | Identidad de la Consonante. <br>• **Ruido/Caos:** Fricativas ('S'). <br>• **Suave/Resonante:** Nasales ('M') o Vocales puras. |
| **Transición** | 13-25 | Cuerpo medio (15%) | Coarticulación (Movimiento de la boca hacia la vocal). |
| **Núcleo (Nucleus)** | 26-38 | Final estable (50%) | Identidad de la Vocal ('A' vs 'I'). |

### 2.4.2. ¿Por qué MFCC y no Espectrograma?
*   **Espectrograma:** Muestra el *tono* absoluto. Varía si el usuario tiene voz aguda o grave, lo cual es indeseable para reconocer fonemas.
*   **MFCC (Cepstrum):** Muestra la **forma del tracto vocal**. Separa la fuente (cuerdas vocales) del filtro (boca), permitiendo evaluar la *pronunciación* independientemente del *tono*.

### 2.4.3. Criterio de Aceptación Visual
Se establece una "Zona de Confianza" basada en la estadística de miles de muestras:
*   **Línea Azul:** Promedio ideal del fonema.
*   **Área Sombreada:** $\pm 1$ Desviación Estándar.

**Regla de Justificación:** Si la trayectoria del usuario sale del área sombreada en una zona crítica (ej. zona de Ataque para 'S'), se justifica el rechazo por "Falla en articulación inicial", independientemente de la probabilidad del modelo.

