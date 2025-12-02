import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import VocalisDataset
from model import VocalisNet
from tqdm import tqdm

# Configuración
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
DATASET_CSV = "train_lab/dataset/metadata.csv"
AUDIO_DIR = "train_lab/dataset/audio"
MODEL_SAVE_PATH = "train_lab/best_vocalis_model.pth"

def train():
    # 1. Preparar Datos
    print("Cargando dataset...")
    full_dataset = VocalisDataset(DATASET_CSV, AUDIO_DIR)
    
    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Dataset cargado. Entrenamiento: {train_size}, Validación: {val_size}")

    # 2. Inicializar Modelo, Loss y Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    model = VocalisNet().to(device)
    
    # Optimizador
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Funciones de Pérdida (CrossEntropy sirve para Multi-Clase y Binaria si son logits)
    criterion_vowel = nn.CrossEntropyLoss()
    criterion_gender = nn.CrossEntropyLoss()

    # 3. Bucle de Entrenamiento
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_vowel = 0
        correct_gender = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for specs, labels_vowel, labels_gender in loop:
            specs = specs.to(device)
            labels_vowel = labels_vowel.to(device)
            labels_gender = labels_gender.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward
            out_vowel, out_gender = model(specs)
            
            # Calcular pérdidas
            loss_v = criterion_vowel(out_vowel, labels_vowel)
            loss_g = criterion_gender(out_gender, labels_gender)
            
            # Pérdida Total (Suma ponderada, aquí 1:1)
            loss = loss_v + loss_g
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Estadísticas
            running_loss += loss.item()
            _, pred_v = torch.max(out_vowel, 1)
            _, pred_g = torch.max(out_gender, 1)
            
            correct_vowel += (pred_v == labels_vowel).sum().item()
            correct_gender += (pred_g == labels_gender).sum().item()
            total += labels_vowel.size(0)
            
            loop.set_postfix(loss=loss.item())
            
        train_acc_v = 100 * correct_vowel / total
        train_acc_g = 100 * correct_gender / total
        avg_train_loss = running_loss / len(train_loader)
        
        # 4. Validación
        model.eval()
        val_loss = 0.0
        val_correct_v = 0
        val_correct_g = 0
        val_total = 0
        
        with torch.no_grad():
            for specs, labels_vowel, labels_gender in val_loader:
                specs = specs.to(device)
                labels_vowel = labels_vowel.to(device)
                labels_gender = labels_gender.to(device)
                
                out_vowel, out_gender = model(specs)
                
                loss_v = criterion_vowel(out_vowel, labels_vowel)
                loss_g = criterion_gender(out_gender, labels_gender)
                
                val_loss += (loss_v + loss_g).item()
                
                _, pred_v = torch.max(out_vowel, 1)
                _, pred_g = torch.max(out_gender, 1)
                
                val_correct_v += (pred_v == labels_vowel).sum().item()
                val_correct_g += (pred_g == labels_gender).sum().item()
                val_total += labels_vowel.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc_v = 100 * val_correct_v / val_total
        val_acc_g = 100 * val_correct_g / val_total
        
        print(f"Epoch {epoch+1} Resumen:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Vowel Acc: {train_acc_v:.2f}% | Gender Acc: {train_acc_g:.2f}%")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Vowel Acc: {val_acc_v:.2f}% | Gender Acc: {val_acc_g:.2f}%")
        
        # Guardar el mejor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> Modelo guardado (Mejor Val Loss)")

if __name__ == "__main__":
    train()
