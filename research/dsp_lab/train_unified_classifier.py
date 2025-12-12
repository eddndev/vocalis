"""
Train UNIFIED classifier for vowels + syllables (25 classes).

This trains a single SVM model that can classify:
- 5 pure vowels: a, e, i, o, u
- 20 CV syllables: pa, pe, pi, po, pu, ta, te, ti, to, tu, ma, me, mi, mo, mu, sa, se, si, so, su

Models are trained per-gender for better accuracy.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

INPUT_CSV = "research/dsp_lab/unified_features.csv"
MODEL_DIR = "research/dsp_lab/models"


def get_feature_columns():
    """Get all 39 MFCC feature column names."""
    onset_cols = [f'mfcc_onset_{i}' for i in range(13)]
    trans_cols = [f'mfcc_trans_{i}' for i in range(13)]
    nucleus_cols = [f'mfcc_nucleus_{i}' for i in range(13)]
    return onset_cols + trans_cols + nucleus_cols


def train_unified_model(df, gender):
    """
    Train unified SVM classifier for 25 classes (vowels + syllables).
    """
    print(f"\n{'='*60}")
    print(f"Training UNIFIED model for gender: {gender}")
    print(f"{'='*60}")
    
    # Filter by gender
    data = df[df['gender'] == gender].copy()
    
    if len(data) < 100:
        print(f"  Skipping: insufficient data ({len(data)} samples)")
        return None
    
    # Get all 39 features
    feature_cols = get_feature_columns()
    X = data[feature_cols].values
    y = data['label'].values
    
    # Class statistics
    classes = np.unique(y)
    
    # --- PROMPT: Filter to only Vowels + M/P Syllables to improve accuracy ---
    # Requested by user: Reduce cardinality to 2 consonant families (M & P)
    valid_syllables = [
        'a', 'e', 'i', 'o', 'u',  # Vowels
        'ma', 'me', 'mi', 'mo', 'mu', # M family
        'pa', 'pe', 'pi', 'po', 'pu'  # P family
    ]
    
    print(f"\nFiltering for {len(valid_syllables)} target classes: {valid_syllables}")
    mask = data['label'].isin(valid_syllables)
    data = data[mask]
    
    # Reload filtered data
    X = data[feature_cols].values
    y = data['label'].values
    classes = np.unique(y)
    
    print(f"Total samples (filtered): {len(data)}")
    print(f"Total classes: {len(classes)}")
    print(f"Classes: {sorted(classes)}")
    
    # Show class distribution
    class_counts = data['label'].value_counts()
    print(f"\nClass distribution (top 10):")
    print(class_counts.head(10).to_string())
    
    # Check for minimum samples per class
    min_samples = class_counts.min()
    if min_samples < 10:
        print(f"\nWARNING: Some classes have very few samples (min={min_samples})")
        # Filter out classes with too few samples
        valid_classes_by_count = class_counts[class_counts >= 10].index.tolist()
        mask = data['label'].isin(valid_classes_by_count)
        data = data[mask]
        X = data[feature_cols].values
        y = data['label'].values
        print(f"Filtered to {len(valid_classes_by_count)} classes with >= 10 samples")
    
    # --- PROMPT: Subsampling to speed up training ---
    # Determine max samples per class (e.g., 1000 or min of top classes, but here fixed for speed)
    MAX_SAMPLES_PER_CLASS = 5000
    
    print(f"\nSubsampling: Limiting to max {MAX_SAMPLES_PER_CLASS} samples per class...")
    subsampled_dfs = []
    for cls in classes:
        cls_data = data[data['label'] == cls]
        if len(cls_data) > MAX_SAMPLES_PER_CLASS:
            cls_data = cls_data.sample(n=MAX_SAMPLES_PER_CLASS, random_state=42)
        subsampled_dfs.append(cls_data)
    
    data = pd.concat(subsampled_dfs)
    # Shuffle
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X = data[feature_cols].values
    y = data['label'].values
    
    print(f"Total samples (subsampled): {len(data)}")
    print(f"Class distribution (subsampled):\n{data['label'].value_counts().head(10).to_string()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)}")
    print(f"Test set: {len(X_test)}")
    
    # Create pipeline with SVM
    # Using higher C for 25-class problem
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(C=10, gamma='scale', kernel='rbf', probability=True, cache_size=1000))
    ])
    
    print("\nTraining SVM (this may take a few minutes)...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"\n{'='*40}")
    print(f">> ACCURACY ({gender}): {acc*100:.2f}%")
    print(f"{'='*40}")
    
    # Detailed report
    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0))
    
    # Show confusion between similar classes (vowels vs syllables ending in same vowel)
    print("\nMost confused pairs:")
    cm = confusion_matrix(y_test, preds, labels=sorted(np.unique(y_test)))
    class_labels = sorted(np.unique(y_test))
    
    # Find top confusions
    confusions = []
    for i, true_label in enumerate(class_labels):
        for j, pred_label in enumerate(class_labels):
            if i != j and cm[i, j] > 0:
                confusions.append((true_label, pred_label, cm[i, j]))
    
    confusions.sort(key=lambda x: x[2], reverse=True)
    for true_l, pred_l, count in confusions[:10]:
        print(f"  {true_l} → {pred_l}: {count} errors")
    
    return pipeline


def main():
    print("=" * 60)
    print("UNIFIED Model Training (Vowels + Syllables)")
    print("=" * 60)
    
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        print("Please run build_unified_dataset.py first.")
        return
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # Load dataset
    print("Loading unified features...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Total samples: {len(df)}")
    print(f"Unique labels: {df['label'].nunique()}")
    print(f"Genders: {df['gender'].value_counts().to_dict()}")
    
    # Train models for each gender
    for gender in ['M', 'F']:
        model = train_unified_model(df, gender)
        
        if model is not None:
            model_path = os.path.join(MODEL_DIR, f"svm_unified_{gender}.pkl")
            joblib.dump(model, model_path)
            print(f"Saved: {model_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("Models saved to:", MODEL_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
