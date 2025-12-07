import os
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from feature_extractor import get_features

# Configuración
DATASET_DIR = "research/train_lab/dataset/audio"
METADATA_FILE = "research/train_lab/dataset/metadata.csv"
OUTPUT_CSV = "research/dsp_lab/dsp_features.csv"

# Número de archivos por lote (Batch size)
# Aumentar esto reduce el overhead de creación de procesos en Windows
BATCH_SIZE = 100 

def process_batch(batch_subset):
    """
    Procesa un lote completo de archivos.
    Esto se ejecuta en un proceso separado.
    """
    batch_results = []
    
    # Iterar sobre el sub-DataFrame recibido
    for _, row in batch_subset.iterrows():
        fname = row['filename']
        path = os.path.join(DATASET_DIR, fname)
        
        try:
            # Carga optimizada 'kaiser_fast' para velocidad
            y, sr = librosa.load(path, sr=16000, res_type='kaiser_fast')
            
            # Extraer features DSP
            feats = get_features(y, sr)
            
            entry = {
                'filename': fname,
                'f0': feats['f0'],
                'f1': feats['f1'],
                'f2': feats['f2'],
                'label_vowel': row['label_vowel'],
                'label_gender': row['label_gender']
            }
            
            # Añadir MFCCs dinámicamente
            for k, v in feats.items():
                if k.startswith('mfcc_'):
                    entry[k] = v
            
            batch_results.append(entry)
        except Exception:
            # Si un audio falla, lo ignoramos y seguimos con el lote
            continue
            
    return batch_results

def main():
    # Asegurar que se ejecuta solo en el proceso principal
    print(f"Iniciando Procesamiento Turbo en Windows...")
    
    if not os.path.exists(METADATA_FILE):
        print(f"Error: No se encuentra {METADATA_FILE}")
        return

    df = pd.read_csv(METADATA_FILE)
    total_files = len(df)
    
    # Dividir el DataFrame en trozos (chunks)
    chunks = [df.iloc[i:i + BATCH_SIZE] for i in range(0, total_files, BATCH_SIZE)]
    print(f"Total archivos: {total_files}")
    print(f"Total lotes: {len(chunks)} (Tamaño de lote: {BATCH_SIZE})")
    
    all_results = []
    
    # Usar todos los núcleos disponibles
    # En Windows, ProcessPoolExecutor es robusto si se usa dentro de if __name__ == '__main__'
    max_workers = os.cpu_count()
    print(f"Utilizando {max_workers} núcleos lógicos.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Enviar todos los lotes a los workers
        futures = [executor.submit(process_batch, chunk) for chunk in chunks]
        
        # Barra de progreso basada en LOTES completados
        for future in tqdm(as_completed(futures), total=len(futures), unit="lote"):
            try:
                result = future.result()
                all_results.extend(result)
            except Exception as e:
                print(f"Error en un lote: {e}")

    # Guardar resultados finales
    print("Guardando CSV...")
    dsp_df = pd.DataFrame(all_results)
    dsp_df.to_csv(OUTPUT_CSV, index=False)
    print(f"¡Éxito! Dataset guardado en {OUTPUT_CSV}")
    print(f"Procesados correctamente: {len(all_results)} / {total_files}")
    print(dsp_df.head(2))

if __name__ == "__main__":
    main()
