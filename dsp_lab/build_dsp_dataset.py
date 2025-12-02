import os
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm
from feature_extractor import get_features

DATASET_DIR = "train_lab/dataset/audio"
METADATA_FILE = "train_lab/dataset/metadata.csv"
OUTPUT_CSV = "dsp_lab/dsp_features.csv"

def main():
    if not os.path.exists(METADATA_FILE):
        print("Metadata no encontrada.")
        return

    df = pd.read_csv(METADATA_FILE)
    print(f"Procesando {len(df)} archivos para extraer F0, F1, F2...")

    results = []
    
    # Procesar cada archivo
    # Usamos tqdm para barra de progreso
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        fname = row['filename']
        path = os.path.join(DATASET_DIR, fname)
        
        try:
            y, sr = librosa.load(path, sr=16000)
            
            # Extraer features DSP
            feats = get_features(y, sr)
            
            results.append({
                'filename': fname,
                'f0': feats['f0'],
                'f1': feats['f1'],
                'f2': feats['f2'],
                'label_vowel': row['label_vowel'],
                'label_gender': row['label_gender']
            })
        except Exception as e:
            # Si falla un archivo (muy corto o corrupto), lo saltamos
            continue
            
    # Guardar DataFrame
    dsp_df = pd.DataFrame(results)
    dsp_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Dataset DSP guardado en {OUTPUT_CSV}")
    print(dsp_df.head())

if __name__ == "__main__":
    main()
