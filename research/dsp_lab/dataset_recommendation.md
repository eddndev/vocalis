# Recomendación de Selección de Dataset (Sila-banco)

Basado en el análisis de `116,346` sílabas del corpus DIMEX100 y considerando la robustez para características MFCC.

## 1. Análisis de Disponibilidad ("Data Richness")

Para evitar overfitting, buscamos clases con **>800 muestras** y balance entre vocales.

| Consonante | a | e | i | o | u | Total Set | Calidad Datos |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **s** (sibilante) | 2642 | 4452 | **7715** | 1909 | 1300 | **18,018** | 🟢 Excelente |
| **t** (plosiva) | 2836 | 3287 | 2193 | 3018 | 868 | **12,202** | 🟢 Balanceado |
| **k** (velar) | 2606 | 1591 | 🔴 316 | **3936** | 1076 | 9,525 | 🟡 Desbalanceado (ki) |
| **l** (líquida) | **5076** | 2442 | 1803 | 2643 | 🔴 416 | 12,380 | 🟡 Escaso en 'u' |
| **m** (nasal) | 2220 | 1689 | 1151 | 1126 | 🔴 **484** | 6,670 | 🟠 Riesgo Overfit (mu) |
| **p** (plosiva) | 1708 | 1003 | 🔴 353 | 1604 | 751 | 5,419 | 🟠 Escaso en 'pi' |

## 2. Idoneidad para MFCC y SVM

### Serie Recomendada #1: `/s/` (sa, se, si, so, su)
*   **Ventaja Acústica:** La `/s/` tiene energía en alta frecuencia (ruido blanco/fricativo) que es **extremadamente distinta** de la estructura armónica de las vocales. Esto facilita enormemente la segmentación (Onset Detection) y la clasificación.
*   **Disponibilidad:** Es la serie más abundante y robusta. Ninguna vocal baja de 1000 muestras.
*   **Riesgo:** La `si` está sobre-representada (7715), se recomienda hacer *undersampling* para igualar a ~2000 muestras y evitar sesgo hacia 'i'.

### Serie Recomendada #2: `/t/` (ta, te, ti, to, tu)
*   **Ventaja Acústica:** La `/t/` es una plosiva sorda con un "burst" limpio y rápido. Transición clara a vocales.
*   **Balance:** Es la serie más equilibrada "naturalmente" (todas >800).
*   **Uso:** Excelente si se prefiere una consonante de golpe (plosiva) en lugar de fricativa.

### Sobre la serie `/m/` (ma, me, mi, mo, mu)
*   **Problema:** Las nasales (`m`, `n`) tienen "anti-resonancias" y murmullos de baja frecuencia que se confunden espectralmente (en MFCC) con vocales cerradas o entre sí.
*   **Riesgo:** Con solo **484 muestras para `mu`**, un SVM RBF (Radial Basis Function) memorizará estos pocos ejemplos (overfitting), fallando con nuevos locutores.
*   **Solución si se fuerza su uso:** Se requiere **Data Augmentation** agresivo (Pitch Shift, Time Stretch, Noise Injection) específicamente para `mu` (x3 o x4) para alcanzar ~1500 muestras.

## Conclusión

Para el modelo más robusto y generalizable "en chinga":

1.  **Opción A (Mejor Rendimiento):** Cambiar el target a **`sa, se, si, so, su`**.
    *   *Acción:* Filtrar dataset para `s` + vocal.
    *   *Nota:* Normalizar `si` (downsample).

2.  **Opción B (Consistencia Clásica):** Usar **`ta, te, ti, to, tu`**.
    *   *Acción:* Filtrar dataset para `t` + vocal.

3.  **Opción C (Arreglar `/m/`):** Mantener `ma, me...` pero **obligatorio** aumentar `mu`.
    *   *Acción:* Generar sintéticos para `mu` antes de entrenar.
