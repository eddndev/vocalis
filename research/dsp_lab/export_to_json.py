import joblib
import json
import numpy as np
import os

# Paths relative to research/ directory
MODEL_DIR = "dsp_lab/models"
OUTPUT_JSON = "dsp_lab/models/vocalis_model.json"

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
    
    svm_params = {
        "gamma": svm._gamma,
        "intercept": svm.intercept_.tolist(),
        "dual_coef": svm.dual_coef_.tolist(),
        "support_vectors": svm.support_vectors_.tolist(),
        "n_support": svm.n_support_.tolist(),
        "classes": svm.classes_.tolist()
    }

    # Extract Probability Parameters (Platt Scaling) if available
    if hasattr(svm, 'probA_') and hasattr(svm, 'probB_'):
        svm_params["probA"] = svm.probA_.tolist()
        svm_params["probB"] = svm.probB_.tolist()
    else:
        print(f"WARNING: The model does not have probability calibration (probA_, probB_).")
        svm_params["probA"] = []
        svm_params["probB"] = []

    params = {
        "scaler": {
            "mean": mean,
            "scale": scale
        },
        "svm": svm_params
    }
    
    return params

def load_model_safe(path):
    """Load a model file, return None if not found."""
    if os.path.exists(path):
        return joblib.load(path)
    return None


def main():
    print("=" * 60)
    print("Exporting models to JSON for Rust/WASM...")
    print("=" * 60)
    
    data = {}
    
    # === VOWEL MODELS (existing) ===
    print("\n[1/2] Vowel Models...")
    path_vowel_m = os.path.join(MODEL_DIR, "svm_model_M.pkl")
    path_vowel_f = os.path.join(MODEL_DIR, "svm_model_F.pkl")
    
    model_vowel_m = load_model_safe(path_vowel_m)
    model_vowel_f = load_model_safe(path_vowel_f)
    
    if model_vowel_m:
        data["vowel_male"] = extract_svm_params(model_vowel_m)
        print(f"  ✓ Vowel Male: {len(data['vowel_male']['svm']['support_vectors'])} SVs")
    else:
        print(f"  ✗ Vowel Male: not found")
    
    if model_vowel_f:
        data["vowel_female"] = extract_svm_params(model_vowel_f)
        print(f"  ✓ Vowel Female: {len(data['vowel_female']['svm']['support_vectors'])} SVs")
    else:
        print(f"  ✗ Vowel Female: not found")
    
    # === SYLLABLE MODELS (new) ===
    print("\n[2/2] Syllable Models...")
    
    # Consonant classifiers (use onset MFCCs)
    for gender in ['M', 'F']:
        path_cons = os.path.join(MODEL_DIR, f"svm_consonant_{gender}.pkl")
        model_cons = load_model_safe(path_cons)
        
        if model_cons:
            key = f"consonant_{'male' if gender == 'M' else 'female'}"
            data[key] = extract_svm_params(model_cons)
            print(f"  ✓ Consonant {gender}: {len(data[key]['svm']['support_vectors'])} SVs, classes: {data[key]['svm']['classes']}")
        else:
            print(f"  ✗ Consonant {gender}: not found")
    
    # Vowel classifiers for syllables (use nucleus MFCCs)
    for gender in ['M', 'F']:
        path_vowel_syl = os.path.join(MODEL_DIR, f"svm_vowel_{gender}.pkl")
        model_vowel_syl = load_model_safe(path_vowel_syl)
        
        if model_vowel_syl:
            key = f"syllable_vowel_{'male' if gender == 'M' else 'female'}"
            data[key] = extract_svm_params(model_vowel_syl)
            print(f"  ✓ Syllable Vowel {gender}: {len(data[key]['svm']['support_vectors'])} SVs")
        else:
            print(f"  ✗ Syllable Vowel {gender}: not found")
    
    # Full syllable classifiers (alternative approach)
    for gender in ['M', 'F']:
        path_full = os.path.join(MODEL_DIR, f"svm_syllable_{gender}.pkl")
        model_full = load_model_safe(path_full)
        
        if model_full:
            key = f"syllable_full_{'male' if gender == 'M' else 'female'}"
            data[key] = extract_svm_params(model_full)
            print(f"  ✓ Full Syllable {gender}: {len(data[key]['svm']['support_vectors'])} SVs, classes: {data[key]['svm']['classes']}")
        else:
            print(f"  ✗ Full Syllable {gender}: not found")
    
    # === UNIFIED MODELS (vowels + syllables, 25 classes) ===
    print("\n[3/3] Unified Models (25 classes)...")
    for gender in ['M', 'F']:
        path_unified = os.path.join(MODEL_DIR, f"svm_unified_{gender}.pkl")
        model_unified = load_model_safe(path_unified)
        
        if model_unified:
            key = f"unified_{'male' if gender == 'M' else 'female'}"
            data[key] = extract_svm_params(model_unified)
            print(f"  ✓ Unified {gender}: {len(data[key]['svm']['support_vectors'])} SVs, classes: {data[key]['svm']['classes']}")
        else:
            print(f"  ✗ Unified {gender}: not found")
    
    if not data:
        print("\nError: No models found to export!")
        return
    
    # Save compact JSON
    print(f"\nSaving to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data, f, indent=None)
    
    # Save debug JSON (readable)
    debug_path = OUTPUT_JSON.replace('.json', '_debug.json')
    with open(debug_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Report file size
    size_mb = os.path.getsize(OUTPUT_JSON) / (1024 * 1024)
    print(f"\n{'=' * 60}")
    print(f"Export complete!")
    print(f"Model file: {OUTPUT_JSON}")
    print(f"File size: {size_mb:.2f} MB")
    print(f"Models exported: {list(data.keys())}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
