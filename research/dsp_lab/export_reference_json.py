import os
import pandas as pd
import numpy as np
import json

# Configuration
INPUT_CSV = "research/dsp_lab/unified_features.csv"
OUTPUT_JSON = "research/dsp_lab/results/reference_data.json"

# Target classes
TARGET_CLASSES = [
    'a', 'e', 'i', 'o', 'u',                # Vowels
    'ma', 'me', 'mi', 'mo', 'mu',           # M-family
    'sa', 'se', 'si', 'so', 'su'            # S-family
]

def get_feature_columns():
    """Get all 39 MFCC feature column names in order."""
    onset_cols = [f'mfcc_onset_{i}' for i in range(13)]
    trans_cols = [f'mfcc_trans_{i}' for i in range(13)]
    nucleus_cols = [f'mfcc_nucleus_{i}' for i in range(13)]
    return onset_cols + trans_cols + nucleus_cols

def main():
    print("="*60)
    print("Exporting Reference Data to JSON")
    print("="*60)
    
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    # Load data
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} samples.")
    
    # Filter
    df = df[df['label'].isin(TARGET_CLASSES)]
    
    feature_cols = get_feature_columns()
    data_export = {}
    
    count = 0
    for label in TARGET_CLASSES:
        for gender in ['M', 'F']:
            subset = df[(df['label'] == label) & (df['gender'] == gender)]
            
            if len(subset) < 5:
                continue
                
            # Compute stats
            matrix = subset[feature_cols].values
            mean_vec = np.mean(matrix, axis=0)
            std_vec = np.std(matrix, axis=0)
            
            key = f"{label}_{gender}"
            data_export[key] = {
                "mean": mean_vec.tolist(),
                "std": std_vec.tolist()
            }
            count += 1
            
    # Save to JSON
    output_dir = os.path.dirname(OUTPUT_JSON)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data_export, f, indent=2)
        
    print(f"\nSuccess! Exported {count} profiles to:")
    print(OUTPUT_JSON)

if __name__ == "__main__":
    main()
