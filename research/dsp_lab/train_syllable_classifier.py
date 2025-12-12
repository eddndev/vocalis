"""
Train hierarchical SVM classifiers for syllable recognition.

Architecture:
- Consonant Classifier: Uses onset MFCCs (13 dims) -> Classes: p, t, m, s
- Vowel Classifier: Uses nucleus MFCCs (13 dims) -> Classes: a, e, i, o, u
- Final syllable = consonant + vowel

Models are trained per-gender (M/F) for better accuracy.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

INPUT_CSV = "research/dsp_lab/syllable_features.csv"
MODEL_DIR = "research/dsp_lab/models"


def get_feature_columns():
    """Get column names for each feature group."""
    onset_cols = [f'mfcc_onset_{i}' for i in range(13)]
    trans_cols = [f'mfcc_trans_{i}' for i in range(13)]
    nucleus_cols = [f'mfcc_nucleus_{i}' for i in range(13)]
    return onset_cols, trans_cols, nucleus_cols


def train_consonant_classifier(df, gender):
    """
    Train SVM to classify consonants using onset MFCCs.
    """
    print(f"\n{'='*60}")
    print(f"Training CONSONANT classifier for gender: {gender}")
    print(f"{'='*60}")
    
    # Filter by gender
    data = df[df['gender'] == gender].copy()
    
    # Get onset features only
    onset_cols, _, _ = get_feature_columns()
    X = data[onset_cols].values
    y = data['consonant'].values
    
    print(f"Samples: {len(data)}")
    print(f"Classes: {np.unique(y)}")
    print(f"Features: {len(onset_cols)} (onset MFCCs)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(C=10, gamma='scale', kernel='rbf', probability=True, cache_size=500))
    ])
    
    print("Training...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n>> Consonant Accuracy ({gender}): {acc*100:.2f}%")
    print(classification_report(y_test, preds))
    
    return pipeline


def train_vowel_classifier(df, gender):
    """
    Train SVM to classify vowels using nucleus MFCCs.
    """
    print(f"\n{'='*60}")
    print(f"Training VOWEL classifier for gender: {gender}")
    print(f"{'='*60}")
    
    # Filter by gender
    data = df[df['gender'] == gender].copy()
    
    # Get nucleus features only
    _, _, nucleus_cols = get_feature_columns()
    X = data[nucleus_cols].values
    y = data['vowel'].values
    
    print(f"Samples: {len(data)}")
    print(f"Classes: {np.unique(y)}")
    print(f"Features: {len(nucleus_cols)} (nucleus MFCCs)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(C=10, gamma='scale', kernel='rbf', probability=True, cache_size=500))
    ])
    
    print("Training...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n>> Vowel Accuracy ({gender}): {acc*100:.2f}%")
    print(classification_report(y_test, preds))
    
    return pipeline


def train_full_syllable_classifier(df, gender):
    """
    Train a single SVM to classify full syllables using all 39 features.
    This is an alternative to the hierarchical approach.
    """
    print(f"\n{'='*60}")
    print(f"Training FULL SYLLABLE classifier for gender: {gender}")
    print(f"{'='*60}")
    
    # Filter by gender
    data = df[df['gender'] == gender].copy()
    
    # Get all features
    onset_cols, trans_cols, nucleus_cols = get_feature_columns()
    all_cols = onset_cols + trans_cols + nucleus_cols
    X = data[all_cols].values
    y = data['syllable'].values
    
    print(f"Samples: {len(data)}")
    print(f"Classes: {np.unique(y)} ({len(np.unique(y))} total)")
    print(f"Features: {len(all_cols)} (all MFCCs)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(C=10, gamma='scale', kernel='rbf', probability=True, cache_size=1000))
    ])
    
    print("Training (this may take longer)...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n>> Full Syllable Accuracy ({gender}): {acc*100:.2f}%")
    print(classification_report(y_test, preds, zero_division=0))
    
    return pipeline


def evaluate_hierarchical_vs_full(df, gender):
    """
    Compare hierarchical approach vs full syllable classification.
    """
    print(f"\n{'='*60}")
    print(f"COMPARISON: Hierarchical vs Full Syllable ({gender})")
    print(f"{'='*60}")
    
    data = df[df['gender'] == gender].copy()
    onset_cols, trans_cols, nucleus_cols = get_feature_columns()
    all_cols = onset_cols + trans_cols + nucleus_cols
    
    # Split data (same split for fair comparison)
    X_all = data[all_cols].values
    y_syllable = data['syllable'].values
    y_consonant = data['consonant'].values
    y_vowel = data['vowel'].values
    
    X_train, X_test, y_syl_train, y_syl_test = train_test_split(
        X_all, y_syllable, test_size=0.2, random_state=42, stratify=y_syllable
    )
    
    _, _, y_con_train, y_con_test = train_test_split(
        X_all, y_consonant, test_size=0.2, random_state=42, stratify=y_syllable
    )
    
    _, _, y_vow_train, y_vow_test = train_test_split(
        X_all, y_vowel, test_size=0.2, random_state=42, stratify=y_syllable
    )
    
    # Train hierarchical models
    cons_model = Pipeline([('scaler', StandardScaler()), ('svm', SVC(C=10, gamma='scale', kernel='rbf'))])
    vowel_model = Pipeline([('scaler', StandardScaler()), ('svm', SVC(C=10, gamma='scale', kernel='rbf'))])
    
    # Consonant uses onset features only
    cons_model.fit(X_train[:, :13], y_con_train)
    # Vowel uses nucleus features only (last 13)
    vowel_model.fit(X_train[:, 26:], y_vow_train)
    
    # Hierarchical prediction
    cons_pred = cons_model.predict(X_test[:, :13])
    vowel_pred = vowel_model.predict(X_test[:, 26:])
    hier_pred = [f"{c}{v}" for c, v in zip(cons_pred, vowel_pred)]
    hier_acc = accuracy_score(y_syl_test, hier_pred)
    
    # Full syllable model
    full_model = Pipeline([('scaler', StandardScaler()), ('svm', SVC(C=10, gamma='scale', kernel='rbf'))])
    full_model.fit(X_train, y_syl_train)
    full_pred = full_model.predict(X_test)
    full_acc = accuracy_score(y_syl_test, full_pred)
    
    print(f"Hierarchical Accuracy: {hier_acc*100:.2f}%")
    print(f"Full Syllable Accuracy: {full_acc*100:.2f}%")
    
    return hier_acc, full_acc


def main():
    print("=" * 60)
    print("Syllable Classifier Training")
    print("=" * 60)
    
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        print("Please run build_syllable_dataset.py first.")
        return
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # Load dataset
    print("Loading syllable features...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Total samples: {len(df)}")
    print(f"Genders: {df['gender'].value_counts().to_dict()}")
    print(f"Syllables: {df['syllable'].value_counts().to_dict()}")
    
    # Compare approaches first
    print("\n>>> Running comparison test...")
    for gender in ['M', 'F']:
        if len(df[df['gender'] == gender]) > 100:
            evaluate_hierarchical_vs_full(df, gender)
    
    # Train hierarchical models (recommended)
    print("\n>>> Training final hierarchical models...")
    
    for gender in ['M', 'F']:
        gender_data = df[df['gender'] == gender]
        if len(gender_data) < 50:
            print(f"Skipping gender {gender}: insufficient data ({len(gender_data)} samples)")
            continue
        
        # Train consonant classifier
        cons_model = train_consonant_classifier(df, gender)
        joblib.dump(cons_model, os.path.join(MODEL_DIR, f"svm_consonant_{gender}.pkl"))
        
        # Train vowel classifier
        vowel_model = train_vowel_classifier(df, gender)
        joblib.dump(vowel_model, os.path.join(MODEL_DIR, f"svm_vowel_{gender}.pkl"))
        
        # Also train full syllable model (for comparison/alternative)
        full_model = train_full_syllable_classifier(df, gender)
        joblib.dump(full_model, os.path.join(MODEL_DIR, f"svm_syllable_{gender}.pkl"))
    
    print("\n" + "=" * 60)
    print("Training complete! Models saved to:", MODEL_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
