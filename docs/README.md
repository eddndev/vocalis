# Vocalis: Clasificación Acústica Determinista mediante DSP y Machine Learning Clásico

**Autores:** Eduardo Alonso  
**Fecha:** Diciembre 2025  
**Versión:** 2.0 (Enfoque DSP/WASM)

## Resumen (Abstract)

Este documento técnico detalla el diseño, implementación y validación de **Vocalis**, un sistema de clasificación de vocales y género en tiempo real ejecutado totalmente en el navegador (Client-side). 

El proyecto marca una evolución desde enfoques basados en Deep Learning ("Cajas Negras") hacia una arquitectura determinista y explicable basada en **Procesamiento Digital de Señales (DSP)**. Utilizando características espectrales robustas (MFCCs), normalización de canal (CMN) y Máquinas de Vectores de Soporte (SVM), el sistema logra alta precisión y robustez ante la variabilidad del entorno.

La implementación final se proyecta sobre un núcleo de alto rendimiento escrito en **Rust** y compilado a **WebAssembly**, consumido por una interfaz moderna en **Astro**.

## Tabla de Contenidos

El contenido de esta investigación técnica se divide en los siguientes módulos:

1.  [**Introducción y Planteamiento del Problema**](./01_introduccion.md)  
    *Análisis de las limitaciones de las redes neuronales en entornos web no controlados y la necesidad de un enfoque físico.*

2.  [**Metodología DSP y Algoritmos**](./02_metodologia_dsp.md)  
    *Explicación profunda de la ingeniería de características: MFCCs, Normalización Cepstral (CMN) y la estrategia de clasificación jerárquica por género.*

3.  [**Arquitectura de Software (Rust + Astro)**](./03_arquitectura_tecnica.md)  
    *Diseño del sistema final: Vocalis Core (Rust/WASM) y Frontend (Astro).*

4.  [**Resultados y Validación**](./04_resultados.md) *(Pendiente de ejecución final)*  
    *Métricas de precisión, matrices de confusión y comparación contra modelos anteriores.*
