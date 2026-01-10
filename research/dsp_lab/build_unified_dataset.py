"""
Build UNIFIED feature dataset combining pure vowels + CV syllables.

This creates a single training dataset with 25 classes:
- 5 pure vowels: a, e, i, o, u
- 20 syllables: pa, pe, pi, po, pu, ta, te, ti, to, tu, ma, me, mi, mo, mu, sa, se, si, so, su

All samples use the same 39-dimensional feature vector (onset + transition + nucleus MFCCs).
"""

import os
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from feature_extractor import get_syllable_features, get_pitch_for_gender
import audiomentations as A

# Configuration - Paths Dynamic
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.dirname(SCRIPT_DIR)

VOWEL_AUDIO_DIR = os.path.join(RESEARCH_DIR, "train_lab", "dataset", "audio")
VOWEL_METADATA = os.path.join(RESEARCH_DIR, "train_lab", "dataset", "metadata.csv")
SYLLABLE_AUDIO_DIR = os.path.join(SCRIPT_DIR, "syllable_dataset", "audio")
SYLLABLE_METADATA = os.path.join(SCRIPT_DIR, "syllable_dataset", "metadata.csv")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "unified_features.csv")
SAMPLE_RATE = 16000

# Batch size for parallel processing
BATCH_SIZE = 20

# AUGMENTATION PIPELINE (Robustness Strategy)
augmenter = A.Compose([
    A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    A.Gain(min_gain_db=-6.0, max_gain_db=6.0, p=0.5),
    A.AirAbsorption(p=0.3), # Simulate spectral tilt / distance
    # PitchShift optionally if needed, but kept conservative for now
])

def process_vowel_batch(batch_data):
    """
    Process a batch of pure vowel audio files.
    For vowels, the 'label' is just the vowel itself (a, e, i, o, u).
    """
    batch_results = []
    audio_dir, batch_subset = batch_data
    
    for _, row in batch_subset.iterrows():
        fname = row['filename']
        path = os.path.join(audio_dir, fname)
        
        try:
            y, sr = librosa.load(path, sr=SAMPLE_RATE, res_type='kaiser_fast')
            
            # 1. Original
            feats = get_syllable_features(y, sr)
            f0 = get_pitch_for_gender(y, sr)
            
            entry = {
                'filename': fname,
                'label': row['label_vowel'],
                'gender': row['label_gender'],
                'speaker_id': row['speaker_id'],
                'source': 'vowel_clean',
                'f0': f0
            }
            for key, value in feats.items(): entry[key] = value
            batch_results.append(entry)

            # 2. Augmentations (3 variants)
            for i in range(3):
                y_aug = augmenter(samples=y, sample_rate=sr)
                feats_aug = get_syllable_features(y_aug, sr)
                
                entry_aug = entry.copy()
                entry_aug['source'] = f'vowel_aug_{i}'
                # F0 might change slightly with noise but we keep original reference or re-calculate
                # Re-calculating F0 for augmented might be unstable with noise, but let's try
                # entry_aug['f0'] = get_pitch_for_gender(y_aug, sr) 
                # Actually, stick to original F0/Gender for stability unless PitchShifted
                
                for key, value in feats_aug.items(): entry_aug[key] = value
                batch_results.append(entry_aug)
            
        except Exception as e:
            continue
    
    return batch_results


def process_syllable_batch(batch_data):
    """
    Process a batch of syllable audio files.
    For syllables, the 'label' is the full syllable (pa, te, mi, etc.).
    """
    batch_results = []
    audio_dir, batch_subset = batch_data
    
    for _, row in batch_subset.iterrows():
        fname = row['filename']
        path = os.path.join(audio_dir, fname)
        
        try:
            y, sr = librosa.load(path, sr=SAMPLE_RATE, res_type='kaiser_fast')
            
            # 1. Original Syllable
            feats = get_syllable_features(y, sr)
            f0 = get_pitch_for_gender(y, sr)
            
            # --- SYLLABLE ENTRY ---
            entry_syll = {
                'filename': fname,
                'label': row['syllable'],
                'gender': row['gender'],
                'speaker_id': row['speaker_id'],
                'source': 'syllable_clean',
                'f0': f0
            }
            for key, value in feats.items(): entry_syll[key] = value
            batch_results.append(entry_syll)
            
            # --- SYNTHETIC VOWEL ENTRY (Recovery Strategy) ---
            # Extract vowel from syllable (last char: 'sa' -> 'a')
            vowel_label = row['syllable'][-1]
            if vowel_label in ['a', 'e', 'i', 'o', 'u']:
                entry_vowel = {
                    'filename': fname,
                    'label': vowel_label, # Synthetic Pure Vowel
                    'gender': row['gender'],
                    'speaker_id': row['speaker_id'],
                    'source': 'vowel_synth_from_syllable',
                    'f0': f0
                }
                # Construct steady-state vowel features from Nucleus
                for i in range(13):
                    n_val = feats[f'mfcc_nucleus_{i}']
                    entry_vowel[f'mfcc_onset_{i}'] = n_val
                    entry_vowel[f'mfcc_trans_{i}'] = n_val
                    entry_vowel[f'mfcc_nucleus_{i}'] = n_val
                
                batch_results.append(entry_vowel)

            # 2. Augmentations (3 variants) - Generates BOTH Syllable and Synthetic Vowel
            for i in range(3):
                y_aug = augmenter(samples=y, sample_rate=sr)
                feats_aug = get_syllable_features(y_aug, sr)
                
                # Augmented Syllable
                entry_syll_aug = entry_syll.copy()
                entry_syll_aug['source'] = f'syllable_aug_{i}'
                for key, value in feats_aug.items(): entry_syll_aug[key] = value
                batch_results.append(entry_syll_aug)
                
                # Augmented Synthetic Vowel
                if vowel_label in ['a', 'e', 'i', 'o', 'u']:
                    entry_vowel_aug = entry_vowel.copy()
                    entry_vowel_aug['source'] = f'vowel_synth_aug_{i}'
                    for k in range(13):
                        n_val = feats_aug[f'mfcc_nucleus_{k}']
                        entry_vowel_aug[f'mfcc_onset_{k}'] = n_val
                        entry_vowel_aug[f'mfcc_trans_{k}'] = n_val
                        entry_vowel_aug[f'mfcc_nucleus_{k}'] = n_val
                    batch_results.append(entry_vowel_aug)
            
        except Exception as e:
            continue
    
    return batch_results


def process_dataset(audio_dir, metadata_path, process_func, desc):
    """
    Process a dataset (vowels or syllables) using parallel workers.
    """
    if not os.path.exists(metadata_path):
        print(f"  Skipping: {metadata_path} not found")
        return []
    
    df = pd.read_csv(metadata_path)
    
    # SUBSAMPLING for Speed (20% of data = ~8400 files -> ~67k augmented samples)
    df = df.sample(frac=0.2, random_state=42)
    
    total = len(df)
    print(f"  Found {total} samples in {metadata_path} (Subsampled 20%)")
    
    # Create batches with audio_dir included
    chunks = [(audio_dir, df.iloc[i:i + BATCH_SIZE]) for i in range(0, total, BATCH_SIZE)]
    
    all_results = []
    # Parallel processing
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_func, chunk) for chunk in chunks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="batch"):
            try:
                result = future.result()
                all_results.extend(result)
            except Exception as e:
                print(f"Batch error: {e}")
    
    return all_results


def main():
    print("=" * 60)
    print("UNIFIED Feature Extraction (Vowels + Syllables)")
    print("=" * 60)
    
    all_results = []
    
    # 1. Process pure vowels
    print("\n[1/2] Processing PURE VOWELS...")
    vowel_results = process_dataset(
        VOWEL_AUDIO_DIR, 
        VOWEL_METADATA, 
        process_vowel_batch, 
        "Vowels"
    )
    all_results.extend(vowel_results)
    print(f"  Extracted: {len(vowel_results)} vowel samples")
    
    # 2. Process syllables
    print("\n[2/2] Processing SYLLABLES...")
    syllable_results = process_dataset(
        SYLLABLE_AUDIO_DIR, 
        SYLLABLE_METADATA, 
        process_syllable_batch, 
        "Syllables"
    )
    all_results.extend(syllable_results)
    print(f"  Extracted: {len(syllable_results)} syllable samples")
    
    # Save combined results
    print("\nSaving unified dataset...")
    result_df = pd.DataFrame(all_results)
    result_df.to_csv(OUTPUT_CSV, index=False)
    
    # Statistics
    print(f"\n{'=' * 60}")
    print("UNIFIED DATASET COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(all_results)}")
    print(f"Output file: {OUTPUT_CSV}")
    print(f"\nClass distribution:")
    print(result_df['label'].value_counts().to_string())
    print(f"\nGender distribution:")
    print(result_df['gender'].value_counts().to_string())


if __name__ == "__main__":
    main()
