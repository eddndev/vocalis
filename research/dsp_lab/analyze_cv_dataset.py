import os
import glob
from collections import defaultdict, Counter

def analyze_cv_dataset(base_path):
    print(f"Scanning dataset at: {base_path}")
    
    # Define vowels and silence/noise tokens
    # Spanish vowels, including potential stressed versions typically found in phonetic transcriptions
    vowels = {'a', 'e', 'i', 'o', 'u', 'a+', 'e+', 'i+', 'o+', 'u+'} 
    silence_noise = {'.sil', 'sil', 'pau', 'vib', 'ruit'} # Common silence/noise markers
    
    cv_counts = defaultdict(Counter)
    total_files = 0
    total_cvs = 0

    # Pattern to match: data/s*/T22/{comunes,individuales}/*.phn
    # recursive glob might be easier if we just walk and filter
    
    for root, dirs, files in os.walk(base_path):
        # We only care about T22/comunes and T22/individuales
        path_parts = root.split(os.sep)
        
        # Check if we are in a valid directory
        # Expected structure: .../sXXX/T22/comunes or .../sXXX/T22/individuales
        if 'T22' not in path_parts:
            continue
        if not (root.endswith('comunes') or root.endswith('individuales')):
            continue
            
        for file in files:
            if file.endswith(".phn"):
                total_files += 1
                file_path = os.path.join(root, file)
                process_file(file_path, vowels, silence_noise, cv_counts)

    print(f"\nAnalysis Complete.")
    print(f"Total files processed: {total_files}")
    
    print("\n--- CV Syllable Counts ---")
    
    # Collect all unique consonants and vowels found
    all_consonants = sorted(cv_counts.keys())
    base_vowels = ['a', 'e', 'i', 'o', 'u']
    
    # Print header
    header = f"{'Consonant':<10} |"
    for v in base_vowels:
        header += f" {v:<6} |"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    global_cv_counts = 0
    
    for cons in all_consonants:
        row = f"{cons:<10} |"
        for v in base_vowels:
            # Sum counts for base vowel and its stressed variant (e.g., 'a' and 'a+')
            count = cv_counts[cons][v] + cv_counts[cons][v + '+']
            row += f" {count:<6} |"
            global_cv_counts += count
        print(row)

    print("-" * len(header))
    print(f"Total CV syllables found: {global_cv_counts}")

def process_file(filepath, vowels, silence_noise, cv_counts):
    try:
        with open(filepath, 'r', encoding='latin-1') as f: # Latin-1 common for older phonetic datasets
            lines = f.readlines()
            
        phonemes = []
        is_header = True
        
        for line in lines:
            line = line.strip()
            if is_header:
                if line == "END OF HEADER":
                    is_header = False
                continue
            
            if not line:
                continue
                
            parts = line.split()
            if len(parts) >= 3:
                phoneme = parts[2]
                phonemes.append(phoneme)
        
        # Sliding window to find CV pairs
        for i in range(len(phonemes) - 1):
            curr = phonemes[i]
            next_p = phonemes[i+1]
            
            # Check if current is Consonant and next is Vowel
            is_curr_vowel = curr in vowels
            is_curr_noise = curr in silence_noise
            
            is_next_vowel = next_p in vowels
            
            if not is_curr_vowel and not is_curr_noise and is_next_vowel:
                # We have a candidate CV pair
                cv_counts[curr][next_p] += 1
                
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    # Assuming script is run from project root, point to research/data
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    analyze_cv_dataset(dataset_path)
