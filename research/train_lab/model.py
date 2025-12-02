import torch
import torch.nn as nn
import torch.nn.functional as F

class VocalisNet(nn.Module):
    def __init__(self, n_mels=64, time_steps=32):
        super(VocalisNet, self).__init__()
        
        # Bloque Convolucional (Feature Extractor)
        # Entrada esperada: [Batch, 1, 64, 32] (para 0.5s de audio)
        self.features = nn.Sequential(
            # Conv 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> [16, 32, 16]
            
            # Conv 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> [32, 16, 8]
            
            # Conv 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # -> [64, 8, 4]
        )
        
        # Calcular tamaño aplanado
        # 64 canales * 8 mels * 4 time_steps = 2048
        self.flatten_size = 64 * (n_mels // 8) * (time_steps // 8)
        
        # Capa densa compartida
        self.shared_fc = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Cabeza 1: Clasificación de Vocal (5 clases: a,e,i,o,u)
        self.head_vowel = nn.Linear(128, 5)
        
        # Cabeza 2: Clasificación de Género (2 clases: M, F)
        self.head_gender = nn.Linear(128, 2)

    def forward(self, x):
        # Feature Extraction
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        
        # Representación latente compartida
        x = self.shared_fc(x)
        
        # Salidas independientes
        vowel_logits = self.head_vowel(x)
        gender_logits = self.head_gender(x)
        
        return vowel_logits, gender_logits
