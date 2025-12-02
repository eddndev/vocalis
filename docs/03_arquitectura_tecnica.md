# 3. Arquitectura de Software e Implementación

Para llevar los algoritmos DSP descritos a un entorno de producción web de alto rendimiento, el sistema **Vocalis** adopta una arquitectura moderna basada en WebAssembly.

## 3.1. Vocalis Core (Motor DSP en Rust)

El núcleo del procesamiento numérico se implementará en **Rust**, un lenguaje de sistemas que garantiza seguridad de memoria y rendimiento nativo. Este núcleo (`vocalis_core`) se compilará a **WebAssembly (WASM)** para ejecutarse en el navegador.

### Responsabilidades del Core (Rust):
1.  **Ingesta de Audio:** Recibir buffers de audio PCM (Float32) desde JavaScript.
2.  **Procesamiento DSP:**
    *   Implementación eficiente de FFT (usando crates como `rustfft`).
    *   Cálculo de MFCCs y CMN.
    *   Estimación de Pitch ($F_0$) mediante autocorrelación.
3.  **Inferencia (Máquina de Estados):**
    *   Carga de los vectores de soporte y parámetros entrenados en Python (exportados como JSON/Binary).
    *   Ejecución de la lógica de decisión SVM.

### Stack Tecnológico del Core:
*   **Lenguaje:** Rust (Edition 2021).
*   **Target:** `wasm32-unknown-unknown`.
*   **Herramienta de Build:** `wasm-pack`.

## 3.2. Aplicación Web (Frontend en Astro)

La interfaz de usuario y la capa de presentación se construirán utilizando **Astro**, un framework web moderno optimizado para contenido estático y rendimiento.

### Características del Frontend:
*   **Componentes UI:** Interactividad mediante "Islands Architecture" (probablemente React o Preact para los componentes de grabación).
*   **Visualización:** Canvas HTML5 para renderizar el espectrograma en tiempo real (usando `requestAnimationFrame`).
*   **Integración WASM:** Carga asíncrona del módulo `vocalis_core.wasm`.

## 3.3. Flujo de CI/CD y Despliegue

El ciclo de vida del software se automatizará para garantizar calidad y despliegue continuo.

*   **Repositorio:** GitHub.
*   **Pipeline (GitHub Actions):**
    1.  **Test:** Ejecución de unit tests en Rust (`cargo test`).
    2.  **Build WASM:** Compilación del crate Rust.
    3.  **Build Web:** Construcción del sitio estático Astro (`npm run build`).
    4.  **Deploy:** Despliegue automático a la infraestructura de hosting.
*   **Infraestructura:**
    *   **Dominio:** `vocalis.eddndev.com`
    *   **Hosting:** Hosting Compartido estándar (cPanel/Apache).
    *   **Despliegue:** Transferencia automatizada de los artefactos estáticos (HTML/JS/WASM) vía FTP/SFTP desde GitHub Actions.

Esta arquitectura asegura que la pesada carga matemática del DSP se maneje con eficiencia nativa (Rust/WASM), mientras que la experiencia de usuario se mantiene fluida y moderna (Astro).
