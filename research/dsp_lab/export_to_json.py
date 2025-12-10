import joblib
import json
import numpy as np
import os

MODEL_DIR = "research/dsp_lab/models"
OUTPUT_JSON = "research/dsp_lab/models/vocalis_model.json"

def extract_svm_params(pipeline):
    """
    Extrae los parámetros matemáticos crudos de un Pipeline (Scaler + SVC).
    """
    scaler = pipeline.named_steps['scaler']
    svm = pipeline.named_steps['svm']
    
    # 1. Parámetros de Normalización (StandardScaler)
    # x_scaled = (x - mean) / scale
    mean = scaler.mean_.tolist()
    scale = scaler.scale_.tolist()
    
    # 2. Parámetros del SVM (SVC con kernel RBF)
    # Decision function: sum(dual_coef * K(support_vectors, x)) + intercept
    
    # Los coeficientes duales en sklearn vienen con forma [n_classes-1, n_support]
    # Para multiclass 'ovr' (one-vs-rest) o 'ovo' (one-vs-one).
    # Sklearn usa 'ovo' por defecto para multiclass SVC.
    
    params = {
        "scaler": {
            "mean": mean,
            "scale": scale
        },
        "svm": {
            "gamma": svm._gamma,
            "intercept": svm.intercept_.tolist(),
            "dual_coef": svm.dual_coef_.tolist(),
            "support_vectors": svm.support_vectors_.tolist(),
            "n_support": svm.n_support_.tolist(),
            "classes": svm.classes_.tolist()
        }
    }
    
    return params

def main():
    print("Exportando modelos a JSON para Rust...")
    
    # Cargar modelos
    path_m = os.path.join(MODEL_DIR, "svm_model_M.pkl")
    path_f = os.path.join(MODEL_DIR, "svm_model_F.pkl")
    
    if not os.path.exists(path_m) or not os.path.exists(path_f):
        print("Error: No se encuentran los archivos .pkl en", MODEL_DIR)
        return

    model_m = joblib.load(path_m)
    model_f = joblib.load(path_f)
    
    # Extraer datos
    data = {
        "model_male": extract_svm_params(model_m),
        "model_female": extract_svm_params(model_f)
    }
    
    # Guardar JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data, f, indent=None) # Sin indent para ahorrar espacio (será grande)
    
    # Guardar versión legible también para debug
    debug_path = OUTPUT_JSON.replace('.json', '_debug.json')
    with open(debug_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"¡Éxito! Modelo exportado a: {OUTPUT_JSON}")
    
    # Imprimir estadísticas
    n_vec_m = len(data['model_male']['svm']['support_vectors'])
    n_vec_f = len(data['model_female']['svm']['support_vectors'])
    print(f"Vectores de soporte (Hombre): {n_vec_m}")
    print(f"Vectores de soporte (Mujer): {n_vec_f}")

if __name__ == "__main__":
    main()
