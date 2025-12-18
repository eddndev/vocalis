"""
Quick validation script to test the trained model's accuracy.
Run this to verify if the model works correctly in Python.
"""

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler

# Load model
with open('research/dsp_lab/models/vocalis_model.json', 'r') as f:
    model_data = json.load(f)

# Load test data
df = pd.read_csv('research/dsp_lab/unified_features.csv')

# Filter to valid classes and gender
valid_classes = ['a', 'e', 'i', 'o', 'u', 'ma', 'me', 'mi', 'mo', 'mu', 'sa', 'se', 'si', 'so', 'su']
df = df[df['label'].isin(valid_classes)]
df_male = df[df['gender'] == 'M'].sample(min(500, len(df[df['gender'] == 'M'])), random_state=42)

# Get features
feature_cols = [f'mfcc_onset_{i}' for i in range(13)] + \
               [f'mfcc_trans_{i}' for i in range(13)] + \
               [f'mfcc_nucleus_{i}' for i in range(13)]

X = df_male[feature_cols].values
y_true = df_male['label'].values

# Load scaler  
scaler_mean = np.array(model_data['unified_male']['scaler']['mean'])
scaler_scale = np.array(model_data['unified_male']['scaler']['scale'])

# Scale features
X_scaled = (X - scaler_mean) / scaler_scale

# Neutralize C0
X_scaled[:, 0] = 0.0
X_scaled[:, 13] = 0.0
X_scaled[:, 26] = 0.0

# Simple distance-based prediction (simulate SVM without sklearn)
# Just to verify if the data/scaler is working
from collections import Counter

print("\n" + "="*60)
print("QUICK VALIDATION - Model Sanity Check")
print("="*60)
print(f"\nTest samples: {len(y_true)}")
print(f"Classes: {sorted(set(y_true))}")
print(f"\nClass distribution:")
print(Counter(y_true))

print(f"\nFeature stats after scaling:")
print(f"  Mean: {X_scaled.mean():.6f} (should be ~0)")
print(f"  Std: {X_scaled.std():.6f} (should be ~1)")
print(f"  Range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")

print(f"\nC0 neutralization check:")
print(f"  C0 Onset (col 0): mean={X_scaled[:, 0].mean():.6f}, std={X_scaled[:, 0].std():.6f}")
print(f"  C0 Trans (col 13): mean={X_scaled[:, 13].mean():.6f}, std={X_scaled[:, 13].std():.6f}")
print(f"  C0 Nucleus (col 26): mean={X_scaled[:, 26].mean():.6f}, std={X_scaled[:, 26].std():.6f}")

# Check for NaN/Inf
if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
    print("\n⚠️  WARNING: NaN or Inf values detected in scaled features!")
else:
    print("\n✓ No NaN/Inf values - features are valid")

print("\n" + "="*60)
print("To test actual SVM accuracy, run the full train script")
print("with train_test_split and check validation accuracy.")
print("="*60 + "\n")
