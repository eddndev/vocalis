## Estado de la Sesión - Vocalis

**Fecha:** 2 de Diciembre de 2025
**Estrategia Actual:** Clasificación Determinista basada en DSP (MFCC + SVM)
**Estado del Proyecto:** Migración a Arquitectura Híbrida (Rust Core + Astro Client)

## Decisiones Arquitectónicas (Validación Experta)
*   **DSP:** Se confirma estrategia "Bag-of-Frames" con CMN (Cepstral Mean Normalization). No se requieren Deltas ni RASTA-PLP.
*   **Inferencia Web:** Se utilizará **Float32 (f32)** en todo el pipeline de Rust/WASM para maximizar rendimiento y aprovechar SIMD, dado que la precisión es suficiente.
*   **Mejora Continua:** Para la vocal 'u', se priorizará "Cost-Sensitive Learning" (pesos de clase) antes que clasificadores jerárquicos.
*   **Visualización:** Se usará **t-SNE** para documentar la separación de clusters.

## Resumen Ejecutivo de la Sesión
Se realizó un giro estratégico fundamental, abandonando el enfoque de Deep Learning (CNN) debido a su baja robustez en entornos reales (~50% precisión). Se implementó y validó una arquitectura basada en **Procesamiento Digital de Señales (DSP)** clásica, logrando una precisión de **97.8%**. El repositorio fue reestructurado para separar la investigación (`research/`) de la ingeniería de producto (`vocalis_core/`).

---

## Logros Técnicos (Hitos Alcanzados)

### 1. Validación Científica (Fase DSP)
*   **Extracción de Características:** Se implementó un pipeline en Python (`research/dsp_lab`) que extrae **13 coeficientes MFCC** y aplica **Normalización Cepstral (CMN)** para eliminar el sesgo del micrófono.
*   **Modelado:** Se entrenaron dos Máquinas de Vectores de Soporte (SVM) con kernel RBF, especializadas por género:
    *   **Modelo Masculino:** 97.21% Precisión.
    *   **Modelo Femenino:** 97.87% Precisión.
*   **Evidencia:** Se generó documentación técnica tipo "Paper" en el directorio `docs/`, detallando la metodología y resultados.

### 2. Ingeniería de Software (Refactorización)
*   **Limpieza del Repo:** Se movieron todos los scripts experimentales, datasets y prototipos antiguos a la carpeta `research/`. La raíz quedó limpia para el nuevo desarrollo.
*   **Exportación de Modelos:** Se creó `research/dsp_lab/export_to_json.py` para extraer los parámetros matemáticos crudos (vectores de soporte, coeficientes duales, interceptos) de los modelos `.pkl` a un archivo `vocalis_model.json`.

### 3. Inicialización del Núcleo (`vocalis_core`)
*   Se inicializó un nuevo crate de **Rust** configurado para compilar a **WebAssembly** (`wasm32-unknown-unknown`).
*   **Estructura Implementada:**
    *   `model.rs`: Estructuras de datos (`serde`) para cargar el JSON del modelo.
    *   `lib.rs`: Punto de entrada WASM que carga el modelo estáticamente (`include_str!`).
    *   `inference.rs`: Esqueleto de la lógica de predicción SVM.

---

## Estado Actual de los Componentes

| Componente | Estado | Descripción |
| :--- | :--- | :--- |
| **Modelo Matemático** | ✅ Completado | Archivo JSON con pesos SVM optimizados y escaladores. |
| **Research Labs** | 🔒 Archivado | Scripts de Python movidos a `research/`. |
| **Vocalis Core (Rust)** | ✅ Completado | Lógica completa: DSP (Pitch/MFCC) + SVM (OvO). Compilado a WASM. |
| **Web Client** | 🚧 En Pruebas | Integrado con WASM. Se debe validar precisión de predicción en entorno real. |
| **Documentación** | ✅ Completada | Documentos técnicos en `docs/`. |

---

## Plan de Acción para la Próxima Sesión

El objetivo es convertir los números del modelo en una aplicación funcional en tiempo real.

### 1. Implementación DSP en Rust (`vocalis_core`)
*   **Módulo `dsp.rs`:** Implementar la cadena de procesamiento de señal usando crates como `rustfft`.
    *   Input: Buffer de audio (f32).
    *   Proceso: Pre-énfasis -> Ventana Hamming -> FFT -> Mel Filterbank -> Log -> DCT -> CMN.
    *   Output: Vector de 13 MFCCs.
*   **Módulo `inference.rs`:** Completar la función `predict`.
    *   Implementar la función de decisión del SVM: $f(x) = \sum (\alpha_i \cdot K(x, x_i)) + b$.
    *   Implementar el Kernel RBF en Rust.

### 2. Compilación WASM
*   Utilizar `wasm-pack` (desde WSL) para compilar el crate a un módulo `.wasm` + glue code JS.

### 3. Desarrollo Frontend (Astro)
*   Inicializar proyecto Astro en `web_client/`.
*   Crear componentes de interfaz (Grabadora, Visualizador de Espectro).
*   Integrar el módulo WASM generado.
*   Desplegar lógica de grabación y visualización (Canvas) migrada del prototipo anterior.

### Notas Técnicas para el Desarrollador
*   **Atención:** El archivo `vocalis_model.json` es grande. Rust lo carga en tiempo de compilación (`include_str!`), por lo que el binario WASM será pesado (~megabytes). Esto es aceptable para la web, pero hay que monitorear el tiempo de carga.
*   **Entorno:** Recordar que la compilación WASM se debe ejecutar en WSL (`wsl wasm-pack build --target web`).