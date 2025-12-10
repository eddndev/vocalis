import os
import torch
import torchaudio
import soundfile as sf
import pandas as pd
from torch.utils.data import Dataset
import torchaudio.transforms as T

# Mapeos estáticos para asegurar consistencia
VOWEL_TO_IDX = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}
GENDER_TO_IDX = {'M': 0, 'F': 1}

class VocalisDataset(Dataset):
    def __init__(self, metadata_file, audio_dir, target_sample_rate=16000, fixed_length_samples=8000):
        """
        Args:
            metadata_file (string): Ruta al archivo csv.
            audio_dir (string): Directorio con los archivos de audio.
            target_sample_rate (int): Tasa de muestreo deseada (16kHz).
            fixed_length_samples (int): Longitud fija para pad/crop (8000 = 0.5s).
        """
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.fixed_length_samples = fixed_length_samples
        
        # Cargar metadata
        df = pd.read_csv(metadata_file)
        
        # Filtrar géneros desconocidos ('U')
        self.annotations = df[df['label_gender'] != 'U'].reset_index(drop=True)
        
        # Definir transformación a MelSpectrogram
        # Configuración estándar optimizada para voz
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=64
        )
        
        # Convertir a escala logarítmica (dB) es crucial para que la CNN "vea" mejor
        self.amplitude_to_db = T.AmplitudeToDB()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Obtener rutas y etiquetas
        row = self.annotations.iloc[index]
        audio_path = os.path.join(self.audio_dir, row['filename'])
        
        label_vowel_str = row['label_vowel']
        label_gender_str = row['label_gender']
        
        # Cargar audio usando soundfile directamente para evitar error de torchcodec
        # waveform, sr = torchaudio.load(audio_path, backend="soundfile")
        wav_numpy, sr = sf.read(audio_path)
        
        # Convertir a Tensor (soundfile devuelve float64, pasamos a float32)
        waveform = torch.from_numpy(wav_numpy).float()
        
        # Soundfile devuelve (samples,) si es mono o (samples, channels) si es estéreo.
        # Torchaudio espera (channels, samples).
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0) # [samples] -> [1, samples]
        else:
            waveform = waveform.t() # [samples, channels] -> [channels, samples]
        
        # Resample si es necesario (aunque ya preprocesamos a 16k)
        if sr != self.target_sample_rate:
            resampler = T.Resample(sr, self.target_sample_rate)
            waveform = resampler(waveform)
            
        # Asegurar longitud fija (Pad o Crop)
        if waveform.shape[1] > self.fixed_length_samples:
            # Crop (tomar el centro)
            start = (waveform.shape[1] - self.fixed_length_samples) // 2
            waveform = waveform[:, start : start + self.fixed_length_samples]
        elif waveform.shape[1] < self.fixed_length_samples:
            # Pad (rellenar con ceros)
            padding = self.fixed_length_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        # Generar Espectrograma
        # Output shape: [1, n_mels, time_steps] -> [1, 64, 32] para 0.5s
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        # Convertir etiquetas a tensores
        label_vowel = torch.tensor(VOWEL_TO_IDX[label_vowel_str], dtype=torch.long)
        label_gender = torch.tensor(GENDER_TO_IDX[label_gender_str], dtype=torch.long)
        
        return mel_spec_db, label_vowel, label_gender
