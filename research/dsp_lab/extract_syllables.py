"""
Extract CV (Consonant-Vowel) syllable pairs from DIMEx-100 corpus.

This script parses .phn annotation files to find consonant+vowel sequences,
extracts the corresponding audio segments, and creates a syllable dataset.
"""

import os
import glob
import json
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# Configuration
DATA_ROOT = "research/data"
OUTPUT_DIR = "research/dsp_lab/syllable_dataset"
OUTPUT_AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")
SPEAKERS_JSON = "research/train_lab/speakers.json"
SAMPLE_RATE = 16000

# Target consonants for syllable families (p, t, m, s)
TARGET_CONSONANTS = {'p', 't', 'm', 's'}

# All valid consonants in Spanish (for future expansion)
ALL_CONSONANTS = {
    'p', 't', 'k', 'b', 'd', 'g',  # Oclusivas
    'f', 's', 'x',                  # Fricativas
    'm', 'n', 'J',                  # Nasales (J = ñ in SAMPA)
    'l', 'r', 'r(',                 # Líquidas
    'tS', 'jj'                      # Africadas
}

# Target vowels
VOWELS = {'a', 'e', 'i', 'o', 'u'}

# Minimum syllable duration in ms (filter out very short segments)
MIN_SYLLABLE_DURATION_MS = 80


def parse_phn_file(phn_path):
    """
    Parse a .phn file from DIMEx-100 (T22 level).
    
    Returns:
        List of tuples: (start_ms, end_ms, label)
    """
    intervals = []
    
    with open(phn_path, 'r', encoding='latin-1') as f:
        lines = f.readlines()
    
    header_passed = False
    for line in lines:
        line = line.strip()
        if line == "END OF HEADER":
            header_passed = True
            continue
        
        if not header_passed or not line:
            continue
        
        parts = line.split()
        if len(parts) >= 3:
            try:
                start_ms = float(parts[0])
                end_ms = float(parts[1])
                label = parts[2].lower()  # Normalize to lowercase
                intervals.append((start_ms, end_ms, label))
            except ValueError:
                continue
    
    return intervals


def find_cv_pairs(intervals, target_consonants=None):
    """
    Find Consonant-Vowel pairs in a sequence of phoneme intervals.
    
    Args:
        intervals: List of (start_ms, end_ms, label)
        target_consonants: Set of consonants to extract (or None for all)
    
    Returns:
        List of tuples: (consonant, vowel, start_ms, end_ms)
    """
    if target_consonants is None:
        target_consonants = ALL_CONSONANTS
    
    cv_pairs = []
    
    for i in range(len(intervals) - 1):
        start_c, end_c, label_c = intervals[i]
        start_v, end_v, label_v = intervals[i + 1]
        
        # Clean labels (remove special markers like ':' or "'")
        clean_c = label_c.replace("'", "").replace(":", "").replace("(", "")
        clean_v = label_v.replace("'", "").replace(":", "")
        
        # Check if this is a CV pair
        if clean_c in target_consonants and clean_v in VOWELS:
            duration = end_v - start_c
            if duration >= MIN_SYLLABLE_DURATION_MS:
                cv_pairs.append({
                    'consonant': clean_c,
                    'vowel': clean_v,
                    'syllable': f"{clean_c}{clean_v}",
                    'start_ms': start_c,
                    'end_ms': end_v,
                    'duration_ms': duration
                })
    
    return cv_pairs


def extract_audio_segment(wav_path, start_ms, end_ms, sr=SAMPLE_RATE):
    """
    Extract a segment of audio from a WAV file.
    
    Args:
        wav_path: Path to the WAV file
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        sr: Target sample rate
    
    Returns:
        numpy array of audio samples, or None if failed
    """
    try:
        y, orig_sr = librosa.load(wav_path, sr=sr)
        
        start_sample = int((start_ms / 1000.0) * sr)
        end_sample = int((end_ms / 1000.0) * sr)
        
        # Bounds check
        if end_sample > len(y):
            end_sample = len(y)
        if start_sample >= end_sample:
            return None
        
        return y[start_sample:end_sample]
    except Exception as e:
        return None


def process_speaker(speaker_id, gender, phn_dir, wav_dir, writer, target_consonants):
    """
    Process all files for a single speaker.
    
    Returns:
        Number of syllables extracted
    """
    count = 0
    
    # Find all .phn files
    phn_files = glob.glob(os.path.join(phn_dir, "*.phn"))
    
    for phn_path in phn_files:
        # Find corresponding WAV file
        base_name = os.path.basename(phn_path).replace(".phn", "")
        wav_path = os.path.join(wav_dir, base_name + ".wav")
        
        if not os.path.exists(wav_path):
            continue
        
        # Parse phoneme annotations
        intervals = parse_phn_file(phn_path)
        
        # Find CV pairs
        cv_pairs = find_cv_pairs(intervals, target_consonants)
        
        # Extract each syllable
        for pair in cv_pairs:
            audio = extract_audio_segment(wav_path, pair['start_ms'], pair['end_ms'])
            
            if audio is None or len(audio) < 100:
                continue
            
            # Generate filename
            filename = f"{speaker_id}_{gender}_{pair['syllable']}_{count:05d}.wav"
            out_path = os.path.join(OUTPUT_AUDIO_DIR, filename)
            
            # Save audio
            sf.write(out_path, audio, SAMPLE_RATE)
            
            # Write metadata
            writer.write(f"{filename},{pair['consonant']},{pair['vowel']},{pair['syllable']},{gender},{speaker_id},{pair['duration_ms']:.1f}\n")
            count += 1
    
    return count


def main():
    print("=" * 60)
    print("DIMEx-100 Syllable Extractor")
    print("=" * 60)
    
    # Create output directories
    if not os.path.exists(OUTPUT_AUDIO_DIR):
        os.makedirs(OUTPUT_AUDIO_DIR)
    
    # Load speaker gender map
    if not os.path.exists(SPEAKERS_JSON):
        print(f"Error: {SPEAKERS_JSON} not found.")
        print("Please ensure speakers.json exists with gender mapping.")
        return
    
    with open(SPEAKERS_JSON, 'r', encoding='utf-8') as f:
        speakers_map = json.load(f)
    
    # Open metadata file
    f_meta = open(METADATA_FILE, 'w', encoding='utf-8')
    f_meta.write("filename,consonant,vowel,syllable,gender,speaker_id,duration_ms\n")
    
    # Get list of speakers
    speakers = sorted([d for d in os.listdir(DATA_ROOT) 
                       if d.startswith('s') and os.path.isdir(os.path.join(DATA_ROOT, d))])
    
    print(f"Found {len(speakers)} speakers")
    print(f"Target consonants: {TARGET_CONSONANTS}")
    print(f"Target vowels: {VOWELS}")
    print(f"Expected syllables: {[c+v for c in sorted(TARGET_CONSONANTS) for v in sorted(VOWELS)]}")
    print()
    
    total_syllables = 0
    
    for speaker in tqdm(speakers, desc="Processing speakers"):
        speaker_path = os.path.join(DATA_ROOT, speaker)
        gender = speakers_map.get(speaker, "U")
        
        if gender == "U":
            continue  # Skip unknown gender
        
        # Process both "comunes" and "individuales" subdirectories
        for sub in ["comunes", "individuales"]:
            phn_dir = os.path.join(speaker_path, "T22", sub)
            wav_dir = os.path.join(speaker_path, "audio_editado", sub)
            
            if not os.path.exists(phn_dir) or not os.path.exists(wav_dir):
                continue
            
            count = process_speaker(speaker, gender, phn_dir, wav_dir, f_meta, TARGET_CONSONANTS)
            total_syllables += count
    
    f_meta.close()
    
    print()
    print("=" * 60)
    print(f"Extraction complete!")
    print(f"Total syllables extracted: {total_syllables}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Metadata file: {METADATA_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
