import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

INPUT_CSV = "dsp_lab/dsp_features.csv"
MODEL_DIR = "dsp_lab/models"

def train_gender_specific_model(df, gender_label):
    print(f"\n{'='*60}")
    print(f"Entrenando modelo SVM para género: {gender_label}")
    print(f"{'='*60}")

    # Filtrar datos por género
    data = df[df['label_gender'] == gender_label].copy()
    
    # Seleccionar features MFCC (mfcc_0 ... mfcc_12)
    mfcc_cols = [col for col in data.columns if col.startswith('mfcc_')]
    X = data[mfcc_cols]
    y = data['label_vowel']
    
    print(f"Muestras: {len(data)}")
    print(f"Features: {len(mfcc_cols)} (MFCCs)")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Pipeline: Scaler + SVM
    # El escalado es CRÍTICO para SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True))
    ])
    
    # Grid Search para optimizar C y Gamma
    # C: Penalización de errores (Alto = menos tolerancia, posible overfitting)
    # Gamma: Influencia de un solo punto (Alto = radio pequeño)
    param_grid = {
        'svm__C': [1, 10, 100],
        'svm__gamma': ['scale', 0.1, 0.01]
    }
    
    print("Buscando hiperparámetros óptimos...")
    grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    print(f"Mejores parámetros: {grid.best_params_}")
    best_model = grid.best_estimator_
    
    # Evaluar
    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n>> Precisión (Test Set) para {gender_label}: {acc*100:.2f}%")
    print(classification_report(y_test, preds))
    
    return best_model

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"No se encontró {INPUT_CSV}. Ejecuta build_dsp_dataset.py primero.")
        return
        
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print("Cargando dataset...")
    df = pd.read_csv(INPUT_CSV)
    
    # Entrenar modelo MASCULINO
    model_m = train_gender_specific_model(df, 'M')
    joblib.dump(model_m, os.path.join(MODEL_DIR, "svm_model_M.pkl"))
    
    # Entrenar modelo FEMENINO
    model_f = train_gender_specific_model(df, 'F')
    joblib.dump(model_f, os.path.join(MODEL_DIR, "svm_model_F.pkl"))

    print("\nEntrenamiento completo. Modelos guardados en dsp_lab/models/")

if __name__ == "__main__":
    main()