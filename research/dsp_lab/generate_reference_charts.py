import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Configuration
INPUT_CSV = "research/dsp_lab/unified_features.csv"
OUTPUT_DIR = "research/dsp_lab/results/reference_charts"

# Define the target classes (15 classes)
TARGET_CLASSES = [
    'a', 'e', 'i', 'o', 'u',                # Vowels
    'ma', 'me', 'mi', 'mo', 'mu',           # M-family
    'sa', 'se', 'si', 'so', 'su'            # S-family
]

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_feature_columns():
    """Get all 39 MFCC feature column names in order."""
    onset_cols = [f'mfcc_onset_{i}' for i in range(13)]
    trans_cols = [f'mfcc_trans_{i}' for i in range(13)]
    nucleus_cols = [f'mfcc_nucleus_{i}' for i in range(13)]
    return onset_cols + trans_cols + nucleus_cols

def plot_class_trajectory(df, label, gender, output_path):
    """
    Generates a plot showing the mean MFCC trajectory across all samples
    for a specific (label, gender) pair.
    """
    # Filter data
    subset = df[(df['label'] == label) & (df['gender'] == gender)]
    
    if len(subset) < 5:
        print(f"Skipping {label}-{gender}: Insufficient data ({len(subset)} samples)")
        return

    # Get features
    cols = get_feature_columns()
    data_matrix = subset[cols].values
    
    # Calculate stats
    mean_trajectory = np.mean(data_matrix, axis=0)
    std_trajectory = np.std(data_matrix, axis=0)
    
    # Create X-axis (0 to 38)
    x = np.arange(len(cols))
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Define zones
    plt.axvspan(0, 12.5, color='#e3f2fd', alpha=0.5, label='Onset (Consonant)')
    plt.axvspan(12.5, 25.5, color='#f3e5f5', alpha=0.5, label='Transition')
    plt.axvspan(25.5, 38, color='#e8f5e9', alpha=0.5, label='Nucleus (Vowel)')
    
    # Plot Mean
    plt.plot(x, mean_trajectory, color='#1565c0', linewidth=2, marker='o', markersize=4, label='Mean MFCC Profile')
    
    # Plot StdDev (Variability)
    plt.fill_between(x, 
                     mean_trajectory - std_trajectory, 
                     mean_trajectory + std_trajectory, 
                     color='#1565c0', alpha=0.2, label='±1 Std Dev')
    
    # Styling
    gender_full = "Masculino" if gender == 'M' else "Femenino"
    plt.title(f"Perfil Acústico Promedio: '{label.upper()}' ({gender_full})\n(Muestras: {len(subset)})", fontsize=14, fontweight='bold')
    plt.xlabel("Feature Index (0-38)", fontsize=12)
    plt.ylabel("MFCC Value (Normalized)", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add separator lines
    plt.axvline(x=12.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=25.5, color='gray', linestyle='--', alpha=0.5)
    
    # Custom ticks
    plt.xticks([6, 19, 32], ['ONSET\n(Consonant)', 'TRANSITION', 'NUCLEUS\n(Vowel)'])
    
    # Save
    ensure_dir(output_path)
    filename = f"{label}_{gender}.png"
    plt.savefig(os.path.join(output_path, filename), dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Generated: {filename}")

def main():
    print("="*60)
    print("Generating Reference Charts (MFCC Trajectories)")
    print("="*60)
    
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    # Load data
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} samples.")
    
    # Filter for target classes only
    df = df[df['label'].isin(TARGET_CLASSES)]
    print(f"Filtered to {len(df)} samples (Target Classes Only).")
    
    # Ensure output directory exists
    ensure_dir(OUTPUT_DIR)
    
    # Generate charts
    count = 0
    for label in TARGET_CLASSES:
        for gender in ['M', 'F']:
            plot_class_trajectory(df, label, gender, OUTPUT_DIR)
            count += 1
            
    print(f"\nDone! {count} charts generated in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
