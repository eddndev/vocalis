"""
Build syllable feature dataset from extracted syllable audio clips.

This script processes the syllable audio files extracted by extract_syllables.py
and generates a CSV with 39-dimensional feature vectors for each syllable.
"""

import os
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from feature_extractor import get_syllable_features, get_pitch_for_gender

# Configuration - Paths relative to research/ directory
SYLLABLE_DATASET_DIR = "dsp_lab/syllable_dataset/audio"
METADATA_FILE = "dsp_lab/syllable_dataset/metadata.csv"
OUTPUT_CSV = "dsp_lab/syllable_features.csv"
SAMPLE_RATE = 16000

# Batch size for parallel processing
BATCH_SIZE = 50


def process_batch(batch_subset):
    """
    Process a batch of syllable audio files.
    Runs in a separate process for parallelization.
    """
    batch_results = []
    
    for _, row in batch_subset.iterrows():
        fname = row['filename']
        path = os.path.join(SYLLABLE_DATASET_DIR, fname)
        
        try:
            # Load audio
            y, sr = librosa.load(path, sr=SAMPLE_RATE, res_type='kaiser_fast')
            
            # Extract 39-dim syllable features
            feats = get_syllable_features(y, sr)
            
            # Extract pitch for gender verification (optional)
            f0 = get_pitch_for_gender(y, sr)
            
            entry = {
                'filename': fname,
                'consonant': row['consonant'],
                'vowel': row['vowel'],
                'syllable': row['syllable'],
                'gender': row['gender'],
                'speaker_id': row['speaker_id'],
                'f0': f0
            }
            
            # Add all MFCC features
            for key, value in feats.items():
                entry[key] = value
            
            batch_results.append(entry)
            
        except Exception as e:
            # Skip failed files
            continue
    
    return batch_results


def main():
    print("=" * 60)
    print("Syllable Feature Extraction")
    print("=" * 60)
    
    if not os.path.exists(METADATA_FILE):
        print(f"Error: {METADATA_FILE} not found.")
        print("Please run extract_syllables.py first.")
        return
    
    # Load metadata
    df = pd.read_csv(METADATA_FILE)
    total_files = len(df)
    
    print(f"Total syllables to process: {total_files}")
    print(f"Feature dimensions: 39 (13 onset + 13 trans + 13 nucleus)")
    
    # Show syllable distribution
    print("\nSyllable distribution:")
    print(df['syllable'].value_counts())
    
    # Divide into batches
    chunks = [df.iloc[i:i + BATCH_SIZE] for i in range(0, total_files, BATCH_SIZE)]
    print(f"\nTotal batches: {len(chunks)} (batch size: {BATCH_SIZE})")
    
    all_results = []
    
    # Use parallel processing
    max_workers = os.cpu_count()
    print(f"Using {max_workers} CPU cores")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, chunk) for chunk in chunks]
        
        for future in tqdm(as_completed(futures), total=len(futures), unit="batch"):
            try:
                result = future.result()
                all_results.extend(result)
            except Exception as e:
                print(f"Batch error: {e}")
    
    # Save results
    print("\nSaving feature CSV...")
    result_df = pd.DataFrame(all_results)
    result_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n{'=' * 60}")
    print(f"Feature extraction complete!")
    print(f"Processed: {len(all_results)} / {total_files} syllables")
    print(f"Output: {OUTPUT_CSV}")
    print(f"{'=' * 60}")
    
    # Show sample
    print("\nSample output:")
    print(result_df.head(3))


if __name__ == "__main__":
    main()
