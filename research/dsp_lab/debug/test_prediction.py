"""
Test: Prediction comparison Python vs Rust
Feed the EXACT SAME feature vector to both and compare results.
"""

import json
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

# Load JSON model
with open('research/dsp_lab/models/vocalis_model.json', 'r') as f:
    model_data = json.load(f)

male_model = model_data['unified_male']

# Test vector from browser: "me" input that predicted "mi"
# These are the EXACT RAW MFCCs from Rust console.log (RMS normalized)
test_features_raw = np.array([
    -151.270264, 96.990906, 22.734461, 28.461168, 11.610863, -15.455238, -38.976437, 
    -18.937061, -13.018241, -12.450939, -9.070406, 8.114470, -19.133718, # Onset
    -110.832085, 101.257141, -4.523974, 32.937489, 6.967783, -32.281391, -24.163864,
    -16.529745, -12.950900, -14.184695, -9.829621, 13.803642, -9.136177, # Transition
    -131.132492, 58.563065, 14.903972, 51.071350, -18.162764, -28.046846, -8.164118,
    -8.478069, -12.512608, -12.459381, 0.610715, 4.090120, -10.829538 # Nucleus
])

print("\n" + "="*60)
print("PREDICTION COMPARISON: Python vs Rust")
print("="*60)

# Apply StandardScaler
scaler_mean = np.array(male_model['scaler']['mean'])
scaler_scale = np.array(male_model['scaler']['scale'])

scaled = (test_features_raw - scaler_mean) / scaler_scale

print(f"\nAfter StandardScaler:")
print(f"  Scaled[0]: {scaled[0]:.6f}")
print(f"  Scaled[1]: {scaled[1]:.6f}")
print(f"  Scaled[13]: {scaled[13]:.6f}")
print(f"  Scaled[26]: {scaled[26]:.6f}")

# Neutralize C0
scaled[0] = 0.0
scaled[13] = 0.0
scaled[26] = 0.0

print(f"\nAfter C0 neutralization:")
print(f"  Scaled[0]: {scaled[0]:.6f}")
print(f"  Scaled[13]: {scaled[13]:.6f}")
print(f"  Scaled[26]: {scaled[26]:.6f}")

# Manual RBF kernel calculation (first support vector only, for debugging)
sv_0 = np.array(male_model['svm']['support_vectors'][0])
gamma = male_model['svm']['gamma']

diff = scaled - sv_0
dist_sq = np.sum(diff ** 2)
kernel_val = np.exp(-gamma * dist_sq)

print(f"\nRBF Kernel Debug (SV[0]):")
print(f"  Gamma: {gamma:.6f}")
print(f"  ||x - sv||²: {dist_sq:.6f}")
print(f"  K(x, sv[0]): {kernel_val:.6f}")

# Load scikit-learn model to get actual prediction
try:
    sklearn_model = joblib.load('research/dsp_lab/models/svm_unified_M.pkl')
    pred = sklearn_model.predict([test_features_raw])[0]
    proba = sklearn_model.predict_proba([test_features_raw])[0]
    
    classes = sklearn_model.classes_
    top_3 = sorted(zip(classes, proba), key=lambda x: -x[1])[:3]
    
    with open("verification_result.txt", "w") as f:
        f.write(f"PREDICTION: {pred}\n")
        f.write("Top 3 probabilities:\n")
        for cls, prob in top_3:
            f.write(f"  {cls}: {prob*100:.2f}%\n")

    print(f"\n✅ PYTHON SKLEARN PREDICTION:")
    print(f"  Predicted class: {pred}")
    print(f"  Top 3 probabilities:")
    for cls, prob in top_3:
        print(f"    {cls}: {prob*100:.2f}%")
    
except FileNotFoundError:
    print("\n⚠️  sklearn model file not found")
    print("Run: python research/dsp_lab/train_unified_classifier.py")

print("\n" + "="*60)
print("Compare these values with Rust console logs")
print("="*60 + "\n")
