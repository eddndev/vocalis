<div align="center">
  <br />
  <a href="https://vocalis.eddn.dev" target="_blank">
    <!-- Placeholder for Logo if available, using text for now -->
    <h1 align="center" style="font-size: 3rem; font-weight: 900;">VOCALIS</h1>
  </a>
  
  <p align="center">
    <strong>Real-time DSP Vowel Analysis Engine powered by Rust & WebAssembly.</strong>
  </p>

  <p align="center">
    <a href="https://vocalis.eddn.dev">Live Demo</a>
    ·
    <a href="https://github.com/eddndev/vocalis/issues">Report Bug</a>
    ·
    <a href="https://github.com/eddndev/vocalis/pulls">Request Feature</a>
  </p>
</div>

<div align="center">

![License](https://img.shields.io/github/license/eddndev/vocalis?style=for-the-badge&color=black)
![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)
![WebAssembly](https://img.shields.io/badge/wasm-%235C4EE5.svg?style=for-the-badge&logo=webassembly&logoColor=white)
![Astro](https://img.shields.io/badge/astro-%232C205F.svg?style=for-the-badge&logo=astro&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white)

</div>

<br />

## ⚡ Introduction

**Vocalis** is a high-performance spectral analysis engine designed to classify vowel sounds in real-time directly within the browser. 

Unlike traditional ML approaches that rely on heavy server-side processing or opaque neural networks, Vocalis leverages **Digital Signal Processing (DSP)** fundamentals—FFT, MFCCs, and Formant Analysis—implemented in **Rust** and compiled to **WebAssembly (WASM)** for near-native performance.

The project features a stunning, minimalist frontend built with **Astro** and **Konpo Design principles**, ensuring that the visualization of physics is as beautiful as the math behind it.

## ✨ Features

- **🦀 Rust Core**: DSP algorithms (FFT, Mel Filterbanks, DCT) written in Rust for safety and speed.
- **🕸️ WebAssembly**: Zero-latency client-side execution using `wasm-bindgen`.
- **📊 Real-time Visualization**: Canvas-based spectrograms and waveform rendering.
- **🧠 SVM Inference**: Support Vector Machine logic for precise vowel classification ($F_1$ vs $F_2$ formants).
- **🎨 Konpo Design**: A "Blueprint" aesthetic with dark/light modes, interactive grids, and fluid animations.

## 🏗️ Architecture

The project is structured as a monorepo:

- **`vocalis_core/`**: The Rust crate containing the DSP pipeline and inference logic.
- **`vocalis/`**: The Astro web application (Frontend).

## 🚀 Getting Started

### Prerequisites

- **Rust & Cargo**: Latest stable version.
- **Node.js**: v18+
- **wasm-pack**: For building the Rust crate.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/eddndev/vocalis.git
   cd vocalis
   ```

2. **Build the WASM Core**
   ```bash
   cd vocalis_core
   wasm-pack build --target web
   ```

3. **Install Frontend Dependencies**
   ```bash
   cd ../vocalis
   npm install
   ```

4. **Run Development Server**
   ```bash
   npm run dev
   ```

## 🛠️ Tech Stack

- **Core**: Rust, Wasm-bindgen, Nalgebra (Linear Algebra).
- **Frontend**: Astro, Vanilla JS (Canvas API), Tailwind CSS v4.
- **Animation**: GSAP (GreenSock), Lenis (Smooth Scroll).
- **Design System**: Konpo / Blueprint.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feat/AmazingFeature`)
3. Commit your Changes (`git commit -m 'feat: Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feat/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">
  <p>Built with precision by <a href="https://github.com/eddndev">@eddndev</a> and <a href="https://github.com/achronyme">@achronyme</a>.</p>
</div>
