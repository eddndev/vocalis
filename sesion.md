# Estado de la Sesión - Vocalis

**Fecha:** 1 de Diciembre de 2025
**Objetivo:** Entrenar modelo de reconocimiento de vocales y género (VocalisNet) y crear un cliente web.

## Progreso Actual

1.  **Preprocesamiento de Datos (Completado):**
    *   Script `train_lab/process_dataset.py` creado y ejecutado.
    *   Dataset de audio generado en `train_lab/dataset/audio/`.
    *   Metadata en `train_lab/dataset/metadata.csv`.

2.  **Infraestructura de Entrenamiento (Completado):**
    *   `train_lab/dataset.py`: Clase `VocalisDataset` implementada (usa `torchaudio`).
    *   `train_lab/model.py`: Arquitectura CNN Multi-Head implementada.
    *   `train_lab/train.py`: Script de entrenamiento listo.
    *   `train_lab/export_onnx.py`: Script de exportación modificado para exportar `vocalis_model.onnx`.

3.  **Core en Rust (`vocalis_core`) (Completado):**
    *   Se adaptó el código para ser compatible con la versión 2.0 de la librería `ort` (ONNX Runtime).
    *   Se corrigieron las rutas de carga del modelo ONNX y los archivos de audio para pruebas.
    *   Los tests unitarios de `vocalis_core` ahora pasan.

4.  **Cliente Web (`web_client`) (En Progreso - Nueva Estrategia):**
    *   Se decidió cambiar la estrategia para el cliente web. En lugar de intentar compilar `vocalis_core` (con `ort`) a WASM (lo cual presenta desafíos de compilación y empaquetado para `ort`), se optó por un enfoque más nativo de navegador.
    *   **Arquitectura:** La inferencia del modelo ONNX se realizará directamente en el navegador usando la librería `onnxruntime-web` (JavaScript). Rust se reservará para el backend/servidor si es necesario, o para lógica compleja no relacionada con la inferencia de ONNX en el frontend.
    *   **Progreso:**
        *   Creación de la rama `feat/js-inference`.
        *   Inicialización de un proyecto web con Vite (Vanilla JS) en `web_client/`.
        *   Instalación de `onnxruntime-web`.
        *   Copia del modelo `vocalis_model.onnx` a `web_client/public/`.
        *   Configuración inicial de `web_client/index.html` y `web_client/src/main.js` para cargar el modelo.

## Próximos Pasos (Cliente Web)

1.  **Implementar captura de audio en `web_client/src/main.js`:**
    *   Acceso a micrófono (`navigator.mediaDevices.getUserMedia`).
    *   Grabación y procesamiento del stream de audio a un formato compatible con el modelo (Float32Array a 16kHz).
2.  **Implementar lógica de inferencia en `web_client/src/main.js`:**
    *   Preprocesar el audio capturado según los requisitos del modelo.
    *   Ejecutar la sesión de inferencia de `onnxruntime-web` con el tensor de audio.
    *   Parsear los resultados y mostrarlos en la UI.
3.  **Estilizar la interfaz de usuario.**
4.  **Añadir controles de error y usabilidad.**
