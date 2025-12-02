import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

INPUT_CSV = "dsp_lab/dsp_features.csv"
MODEL_DIR = "dsp_lab/models"

def train_gender_model(df):
    print("\n--- Entrenando Clasificador de GÉNERO (Basado en F0/Pitch) ---")
    # Filtramos donde F0 es válido (> 0)
    data = df[df['f0'] > 50].copy()
    
    X = data[['f0']] # Usamos solo F0 para género, es la variable física dominante
    y = data['label_gender']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Max depth restringido para mantenerlo explicable
    clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"Precisión (Accuracy): {acc*100:.2f}%")
    print("Reglas del Árbol:")
    print(export_text(clf, feature_names=['f0']))
    
    return clf

def train_vowel_model(df):
    print("\n--- Entrenando Clasificador de VOCALES (Basado en F1, F2) ---")
    # Filtramos donde F1 y F2 son válidos
    data = df[(df['f1'] > 0) & (df['f2'] > 0)].copy()
    
    X = data[['f1', 'f2']] # Usamos los formantes
    y = data['label_vowel']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Un poco más de profundidad para capturar las 5 vocales
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"Precisión (Accuracy): {acc*100:.2f}%")
    print("Importancia de características:")
    for name, imp in zip(['f1', 'f2'], clf.feature_importances_):
        print(f"  {name}: {imp:.4f}")
        
    return clf

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"No se encontró {INPUT_CSV}. Ejecuta build_dsp_dataset.py primero.")
        return
        
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    df = pd.read_csv(INPUT_CSV)
    print(f"Datos cargados: {len(df)} muestras.")
    
    # 1. Género
    gender_model = train_gender_model(df)
    joblib.dump(gender_model, os.path.join(MODEL_DIR, "gender_tree.pkl"))
    
    # 2. Vocales
    vowel_model = train_vowel_model(df)
    joblib.dump(vowel_model, os.path.join(MODEL_DIR, "vowel_tree.pkl"))

if __name__ == "__main__":
    main()
