# Vocalis Model Robustness Plan: Data Augmentation Strategy

## 1. Problem Diagnosis
Current model performs well on validation data (92% accuracy) but fails in real-world browser usage and even in direct Python tests with a different microphone/environment (accuracy ~30-50%).
- **Symptoms:** Confusion between Front Vowels (E/I) and Back Vowels (O/U).
- **Cause:** Overfitting to the spectral characteristics (EQ, SN) of the clean training dataset. The model relies on absolute frequency shaping which varies by microphone.
- **Goal:** Train a model that is "mic-agnostic" and Robust to spectral tilt and mild noise.

## 2. Solution: Offline Data Augmentation
Instead of collecting more data immediately, we will multiply the existing dataset by applying random audio transformations that simulate real-world imperfections.

### A. Tools
We will use the library `audiomentations` which is industry standard for audio ML.

### B. Augmentation Pipeline
For each original sample in the dataset, we will generate **3-5 augmented versions** with a random mix of:

1.  **AddGaussianNoise**: Simulates electrical noise / thermal noise from cheap preamps.
    *   *Range:* 0.001 to 0.015 amplitude.
2.  **AirAbsorption / HighLowPass**: Simulates different microphone frequency responses (spectral tilt).
    *   *Critical for determining E vs I confusion.*
    *   *Action:* Randomly rolloff highs (simulating "dull" mics) or cut lows (simulating "tinny" phone mics).
3.  **Gain**: Random volume changes.
    *   *Range:* -6dB to +6dB.
    *   *Purpose:* Ensure model doesn't rely on absolute volume even after normalization.
4.  **PitchShift (Optional/Light):** +/- 1 semitone.
    *   *Purpose:* Account for slight intonation differences, though we want to preserve the F1/F2 formant positions mostly.

## 3. Implementation Steps

### Step 1: Install Dependencies
```bash
pip install audiomentations
```

### Step 2: Modify `build_unified_dataset.py`
We will inject the augmentation logic into the feature extraction loop.

**Pseudocode Logic:**
```python
import audiomentations as A

augmenter = A.Compose([
    A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    A.Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.5),
    A.AirAbsorption(p=0.3), # Simulate distance/freq loss
    # A.HighPassFilter(min_cutoff_freq=200, max_cutoff_freq=800, p=0.3)
])

# Inside the processing loop:
# 1. Extract features from Original
features.append(extract_features(y))

# 2. Generate Augmented Versions
for _ in range(3):
    y_aug = augmenter(samples=y, sample_rate=sr)
    features.append(extract_features(y_aug))
```

### Step 3: Retrain and Validate
1.  Run `build_unified_dataset.py` (Dataset size will increase 4x).
2.  Run `train_unified_classifier.py`.
3.  **Crucial:** Check if validation accuracy *drops* slightly (expected, as the task is harder) but stays acceptable (>85%).
4.  Export model.

### Step 4: Verification
Run `live_test.py` again. The prediction stability should be significantly higher.

## 4. Future Considerations (If Augmentation isn't enough)
- **Mozilla Common Voice:** Incorporate a subset of this massive dataset labeled for vowels (requires alignment logic).
- **Spectral Normalization layers:** Move normalization deeper into the feature extraction (e.g., RASTA-PLP).
