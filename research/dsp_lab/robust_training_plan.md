# Vocalis Robustness Architecture (Implemented)

**Date:** Jan 05, 2026
**Status:** ✅ Implemented & Deployed
**Strategy:** "Synthesis + Augmentation"

## 1. Core Problem Solved
The initial Vocalis model suffered from **overfitting to clean studio audio** (Dimex100). It failed in real-world browser environments due to:
- Microphone noise floor differences.
- Distance/gain variations.
- Lack of pure vowel samples (deleted dataset).

## 2. The "Synthesis + Augmentation" Pipeline

We implemented a 4-stage processing pipeline to generate a robust 67k+ sample dataset from scratch:

### Stage 1: Raw Data Recovery
- **Source:** Dimex100 Raw Audio (`research/data`)
- **Action:** Recovered missing `speakers.json` from Git History (Commit `1f0ee2e`) to restore accurate **Gender Labels** (Critical for F0 pitch detection).
- **Result:** ~8,398 raw syllable files with verified metadata.

### Stage 2: Syllable Feature Extraction
- **Script:** `extract_syllables.py`
- **Logic:** Scans phoneme files (`.phn`) to extract CV pairs (e.g., `sa`, `me`, `ti`).
- **Output:** `dsp_lab/syllable_dataset` (Clean, Raw Audio).

### Stage 3: Vowel Synthesis (The "Recovery" Hack)
- **Problem:** We lost the original "Pure Vowel" dataset.
- **Solution:** We synthesized pure vowels by extracting the **Nucleus** (steady-state) from every syllable.
  - *Example:* `sa` -> Extract `a` segment -> Label as class `a`.
- **Impact:** Recovered all 5 vowel classes (`a,e,i,o,u`) without needing new recordings.

### Stage 4: 4x Data Augmentation
We used `audiomentations` to generate 3 augmented variants for EVERY sample (both syllables and synthesized vowels):

| Variant | Transformation | Purpose |
| :--- | :--- | :--- |
| **Clean** | Original Audio | Baseline accuracy |
| **Aug 0** | Gaussian Noise (SNR 20-30dB) | Resistance to mic hiss |
| **Aug 1** | Gain Transition (-6dB to +6dB) | Resistance to distance moving |
| **Aug 2** | SevenBandParametricEQ | Resistance to diff mic frequency responses |

## 3. Final Dataset Stats
- **Total Training Samples:** **67,184**
- **Composition:** 25% Clean / 75% Augmented
- **Classes:** 25 (5 Vowels + 20 Syllables) x 2 Genders

## 4. Model Performance
- **Unified SVM (Male/Female):**
- **Validation Accuracy:** ~92.7% (Female)
- **Real-World Impact:** Significantly improved robust detection of `e` vs `i` and `o` vs `u` in noisy browser environments.

## 5. Deployment
- **Model Format:** `vocalis_model.json` (Includes all SVM support vectors).
- **Integration:** Compiled directly into `vocalis_core.wasm` via `include_str!`.
- **Update Procedure:**
  1. python `build_unified_dataset.py` (Generates .csv)
  2. python `train_unified_classifier.py` (Generates .pkl)
  3. python `export_to_json.py` (Generates .json)
  4. `wasm-pack build` (Compiles WASM)
