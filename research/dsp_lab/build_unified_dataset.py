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

# Configuration - Paths relative to research/ directory
VOWEL_AUDIO_DIR = "train_lab/dataset/audio"
VOWEL_METADATA = "train_lab/dataset/metadata.csv"
SYLLABLE_AUDIO_DIR = "dsp_lab/syllable_dataset/audio"
SYLLABLE_METADATA = "dsp_lab/syllable_dataset/metadata.csv"
OUTPUT_CSV = "dsp_lab/unified_features.csv"
SAMPLE_RATE = 16000

# Batch size for parallel processing
BATCH_SIZE = 100


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
            
            # Extract 39-dim features (same for vowels and syllables)
            feats = get_syllable_features(y, sr)
            f0 = get_pitch_for_gender(y, sr)
            
            entry = {
                'filename': fname,
                'label': row['label_vowel'],  # Pure vowel: a, e, i, o, u
                'gender': row['label_gender'],
                'speaker_id': row['speaker_id'],
                'source': 'vowel',
                'f0': f0
            }
            
            for key, value in feats.items():
                entry[key] = value
            
            batch_results.append(entry)
            
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
            
            # Extract 39-dim features
            feats = get_syllable_features(y, sr)
            f0 = get_pitch_for_gender(y, sr)
            
            entry = {
                'filename': fname,
                'label': row['syllable'],  # Full syllable: pa, te, mi, etc.
                'gender': row['gender'],
                'speaker_id': row['speaker_id'],
                'source': 'syllable',
                'f0': f0
            }
            
            for key, value in feats.items():
                entry[key] = value
            
            batch_results.append(entry)
            
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
    total = len(df)
    print(f"  Found {total} samples in {metadata_path}")
    
    # Create batches with audio_dir included
    chunks = [(audio_dir, df.iloc[i:i + BATCH_SIZE]) for i in range(0, total, BATCH_SIZE)]
    
    all_results = []
    max_workers = os.cpu_count()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
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
