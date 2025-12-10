import os
import glob
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import json # Import json

# Configuración
DATA_ROOT = "data"
OUTPUT_DIR = "train_lab/dataset"
OUTPUT_AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")
SPEAKERS_JSON_FILE = "train_lab/speakers.json" # Define the speakers JSON file
SAMPLE_RATE = 16000

# Vocales objetivo (Nivel T22)
VALID_VOWELS = {'a', 'e', 'i', 'o', 'u'}

# Remove detect_gender function as it's replaced by direct lookup
# def detect_gender(wav_path):
#     """
#     Detecta género estimando el Pitch promedio.
#     < 165Hz = Hombre (M), > 165Hz = Mujer (F)
#     """
#     try:
#         sound = parselmouth.Sound(wav_path)
#         pitch = sound.to_pitch()
#         mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
        
#         if np.isnan(mean_pitch):
#             return "U"
            
#         return "M" if mean_pitch < 165 else "F"
#     except Exception:
#         return "U"

def parse_phn_file(phn_path):
    """
    Parsea el archivo .phn del DIMEx100 (formato T22).
    Retorna lista de tuplas: (start_sec, end_sec, label)
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
        
        if not header_passed:
            continue
            
        if not line: continue
        
        parts = line.split()
        if len(parts) >= 3:
            # Formato: start_ms end_ms label
            # Convertimos a segundos
            try:
                start_sec = float(parts[0]) / 1000.0
                end_sec = float(parts[1]) / 1000.0
                label = parts[2].lower()
                intervals.append((start_sec, end_sec, label))
            except ValueError:
                continue
                
    return intervals

def process_file_pair(wav_path, phn_path, speaker_id, gender, writer):
    try:
        intervals = parse_phn_file(phn_path)
        if not intervals:
            return 0

        # Cargar audio
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
        
        count = 0
        for start, end, label in intervals:
            # Limpieza básica
            clean_label = label.replace("'", "").replace(":", "") 
            
            if clean_label in VALID_VOWELS:
                # Filtrar duraciones muy cortas (< 30ms) que pueden ser ruido
                if (end - start) < 0.03:
                    continue
                
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                
                # Verificar límites
                if end_sample > len(y):
                    end_sample = len(y)
                
                y_slice = y[start_sample:end_sample]
                
                # Guardar
                filename = f"{speaker_id}_{gender}_{clean_label}_{count:04d}.wav"
                out_path = os.path.join(OUTPUT_AUDIO_DIR, filename)
                
                sf.write(out_path, y_slice, sr)
                
                writer.write(f"{filename},{clean_label},{gender},{speaker_id}\n")
                count += 1
        return count
    except Exception as e:
        print(f"Error en {os.path.basename(wav_path)}: {e}")
        return 0

def main():
    if not os.path.exists(OUTPUT_AUDIO_DIR):
        os.makedirs(OUTPUT_AUDIO_DIR)
    
    f_meta = open(METADATA_FILE, 'w')
    f_meta.write("filename,label_vowel,label_gender,speaker_id\n")
    
    # Load speaker gender map
    if not os.path.exists(SPEAKERS_JSON_FILE):
        print(f"Error: No se encontró el archivo de clasificación de locutores: {SPEAKERS_JSON_FILE}")
        print("Por favor, ejecuta el script de clasificación manual primero.")
        return

    with open(SPEAKERS_JSON_FILE, 'r', encoding='utf-8') as f:
        speakers_map = json.load(f)

    speakers = [d for d in os.listdir(DATA_ROOT) if d.startswith('s') and os.path.isdir(os.path.join(DATA_ROOT, d))]
    print(f"Procesando {len(speakers)} locutores...")
    
    total_clips = 0
    
    for spk in tqdm(speakers):
        spk_path = os.path.join(DATA_ROOT, spk)
        
        # 1. Obtener Género desde el mapa
        gender = speakers_map.get(spk, "U") # Default to 'U' (Unknown) if not found in map
        if gender == "U":
            print(f"Advertencia: Locutor {spk} no encontrado en speakers.json. Se usará 'U' para género.")
            
        # 2. Procesar archivos (Comunes e Individuales)
        for sub in ["comunes", "individuales"]:
            wav_dir = os.path.join(spk_path, "audio_editado", sub)
            phn_dir = os.path.join(spk_path, "T22", sub)
            
            if not os.path.exists(wav_dir) or not os.path.exists(phn_dir):
                continue
                
            # Iterar sobre archivos .wav y buscar su par .phn
            wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
            
            for wav_f in wav_files:
                base_name = os.path.basename(wav_f).replace(".wav", "")
                phn_f = os.path.join(phn_dir, base_name + ".phn")
                
                if os.path.exists(phn_f):
                    total_clips += process_file_pair(wav_f, phn_f, spk, gender, f_meta)
                    
    f_meta.close()
    print(f"\nHecho. Total muestras: {total_clips}")

if __name__ == "__main__":
    main()