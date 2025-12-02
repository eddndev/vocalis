import torch
import torch.nn as nn
import torchaudio.transforms as T
from model import VocalisNet

# Configuración
MODEL_PATH = "train_lab/best_vocalis_model.pth"
ONNX_PATH = "models/vocalis_model.onnx"

class EndToEndVocalis(nn.Module):
    def __init__(self, base_model_path):
        super().__init__()
        # 1. Cargar modelo base
        self.vocalis_net = VocalisNet()
        # Cargar pesos
        self.vocalis_net.load_state_dict(torch.load(base_model_path, weights_only=True))
        self.vocalis_net.eval()
        
        # 2. Definir transformaciones (IGUAL que en dataset.py)
        # Esto incrusta el preprocesamiento en el grafo computacional
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=256,
            n_mels=64
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        
    def forward(self, waveform):
        """
        Args:
            waveform: Tensor de audio crudo [Batch, samples]
                      Longitud esperada ~8000 muestras (0.5s a 16kHz)
        """
        # Calcular MelSpectrogram -> [Batch, n_mels, time]
        mel = self.mel_spectrogram(waveform)
        
        # Convertir a dB
        mel_db = self.amplitude_to_db(mel)
        
        # VocalisNet espera [Batch, 1, n_mels, time] (Formato imagen NCHW)
        # Necesitamos añadir dimensión de canal C=1
        x = mel_db.unsqueeze(1) 
        
        # Inferencia
        return self.vocalis_net(x)

def export_e2e():
    print("Preparando exportación End-to-End...")
    # Crear modelo wrapper
    model_e2e = EndToEndVocalis(MODEL_PATH)
    model_e2e.eval()
    
    # Crear entrada dummy (Audio crudo)
    # [Batch=1, Samples=8000] -> 0.5 segundos a 16kHz
    dummy_waveform = torch.randn(1, 8000)
    
    print(f"Exportando modelo End-to-End a {ONNX_PATH}...")
    
    # opset_version=17 es necesario para operaciones STFT (ShortTimeFourierTransform)
    torch.onnx.export(
        model_e2e,
        dummy_waveform,
        ONNX_PATH,
        verbose=False,
        input_names=['waveform'],
        output_names=['vowel_logits', 'gender_logits'],
        opset_version=17, 
        dynamic_axes={
            'waveform': {0: 'batch_size', 1: 'samples'}, # Longitud variable permitida
            'vowel_logits': {0: 'batch_size'},
            'gender_logits': {0: 'batch_size'}
        }
    )
    print("¡Exportación End-to-End exitosa!")

if __name__ == "__main__":
    export_e2e()
