use hound;
use std::path::Path;
use vocalis_core::{init_vocalis_internal, predict_vowel_internal};

fn main() {
    // 1. Inicializar el motor (carga el modelo JSON)
    println!("Inicializando Vocalis Core...");
    if let Err(e) = init_vocalis_internal() {
        eprintln!("Error inicializando modelo: {:?}", e);
        return;
    }

    // Ruta base al dataset
    let base_path = Path::new("../research/train_lab/dataset/audio");
    
    // Lista de archivos de prueba (algunos ejemplos manuales representando cada vocal)
    // Asegúrate de que estos archivos existan. Si no, ajusta los nombres.
    // Formato esperado: sXXX_G_V_NNNN.wav
    let test_files = vec![
        // Hombre
        "s001_M_a_0001.wav",
        "s001_M_e_0001.wav",
        "s001_M_i_0001.wav",
        "s001_M_o_0001.wav",
        "s001_M_u_0001.wav",
        // Mujer
        "s001_F_a_0001.wav",
        "s001_F_e_0001.wav",
        "s001_F_i_0001.wav",
        "s001_F_o_0001.wav",
        "s001_F_u_0001.wav",
    ];

    println!("{:<20} | {:<10} | {:<10} | {:<10}", "Archivo", "Esperado", "Predicho", "Estado");
    println!("{}", "-".repeat(60));

    let mut correct = 0;
    let mut total = 0;

    for filename in test_files {
        let file_path = base_path.join(filename);
        
        if !file_path.exists() {
            println!("{:<20} | {:<10} | {:<10} | SKIP (No existe)", filename, "-", "-");
            continue;
        }

        // Leer audio WAV
        let reader_result = hound::WavReader::open(&file_path);
        match reader_result {
            Ok(mut reader) => {
                let spec = reader.spec();
                let sample_rate = spec.sample_rate as f32;
                
                // Convertir a f32 (normailizado -1.0 a 1.0 si es necesario, hound lee ints por defecto)
                let samples: Vec<f32> = reader.samples::<i16>()
                    .map(|s| s.unwrap() as f32 / 32768.0)
                    .collect();

                // INFERENCIA USANDO VERSIÓN INTERNA (NO WASM)
                let result_json = predict_vowel_internal(&samples, sample_rate);
                
                match result_json {
                    Ok(json_str) => {
                        // Parsear JSON simple
                        let vowel_start = json_str.find("\"vowel\":\"").unwrap() + 9;
                        let vowel_end = json_str[vowel_start..].find("\"").unwrap();
                        let predicted_vowel = &json_str[vowel_start..vowel_start+vowel_end];

                        // Extraer etiqueta real del nombre del archivo (s001_M_a_...)
                        let parts: Vec<&str> = filename.split('_').collect();
                        let expected_vowel = parts[2];

                        let status = if predicted_vowel == expected_vowel { "OK" } else { "FAIL" };
                        if status == "OK" { correct += 1; }
                        total += 1;

                        println!("{:<20} | {:<10} | {:<10} | {}", filename, expected_vowel, predicted_vowel, status);
                    },
                    Err(e) => {
                         println!("{:<20} | ERROR Inferencia: {:?}", filename, e);
                    }
                }
            },
            Err(e) => println!("Error leyendo {}: {:?}", filename, e),
        }
    }

    println!("{}", "-".repeat(60));
    if total > 0 {
        println!("Precisión del Test: {:.2}% ({}/{})", (correct as f32 / total as f32) * 100.0, correct, total);
    }
}
