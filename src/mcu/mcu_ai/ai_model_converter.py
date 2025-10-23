#!/usr/bin/env python3
import numpy as np
import os
import json
from datetime import datetime

class AIModelConverter:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.model_dir = os.path.join(self.base_dir, "data", "ai_models", "model_simple")
        self.training_dir = os.path.join(self.base_dir, "data", "training_data")  # ← NOUVEAU CHEMIN
        self.output_dir = os.path.join(self.base_dir, "data", "ai_models", "esp32_deployment")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_tflite_model(self):
        try:
            tflite_path = os.path.join(self.model_dir, "ai_autoencoder.tflite")
            with open(tflite_path, 'rb') as f:
                self.model_data = f.read()
            self.model_size_kb = len(self.model_data) / 1024
            print(f"Modèle TFLite chargé: {self.model_size_kb:.1f} KB")
            return True
        except Exception as e:
            print(f"Erreur chargement modèle TFLite: {e}")
            return False
    
    def load_model_metadata(self):
        try:
            # Charger les seuils depuis model_simple
            thresholds_path = os.path.join(self.model_dir, "ai_thresholds.json")
            with open(thresholds_path, 'r') as f:
                thresholds_data = json.load(f)
                self.thresholds = thresholds_data["thresholds"]
            
            # CORRIGÉ : Charger les noms de features depuis training_data
            features_path = os.path.join(self.training_dir, "ai_feature_names.npy")  # ← CHEMIN CORRIGÉ
            self.feature_names = np.load(features_path, allow_pickle=True).tolist()
            print(f"Metadata chargée: {len(self.feature_names)} features")
            return True
        except Exception as e:
            print(f"Erreur chargement metadata: {e}")
            return False
    
    def generate_c_header(self):
        try:
            hex_array = ', '.join(f'0x{byte:02x}' for byte in self.model_data)
            
            header_content = f"""#ifndef EPS_GUARDIAN_AI_MODEL_H
#define EPS_GUARDIAN_AI_MODEL_H

#include <cstdint>
#include <cstddef>

namespace eps_guardian {{
namespace ai_model {{

alignas(8) const uint8_t g_ai_model_data[] = {{
    {hex_array}
}};

const size_t g_ai_model_size = sizeof(g_ai_model_data);

constexpr float NORMAL_THRESHOLD = {self.thresholds['normal']:.6f}f;
constexpr float WARNING_THRESHOLD = {self.thresholds['warning']:.6f}f;
constexpr float CRITICAL_THRESHOLD = {self.thresholds['critical']:.6f}f;

// Features utilisées par le modèle
constexpr int NUM_FEATURES = {len(self.feature_names)};
const char* FEATURE_NAMES[] = {{
    {', '.join(f'"{name}"' for name in self.feature_names)}
}};

}} 
}} 

#endif
"""
            header_path = os.path.join(self.output_dir, "eps_guardian_ai_model.h")
            with open(header_path, 'w') as f:
                f.write(header_content)
            print(f"Header C++ généré: {header_path}")
            return header_path
        except Exception as e:
            print(f"Erreur génération header: {e}")
            return None
    
    def generate_arduino_example(self):
        try:
            arduino_content = f"""#include "eps_guardian_ai_model.h"
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>

constexpr int kTensorArenaSize = 12 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

class EPSGuardianAI {{
private:
    const tflite::Model* model;
    tflite::MicroInterpreter* interpreter;
    TfLiteTensor* input;
    TfLiteTensor* output;
    tflite::AllOpsResolver resolver;

public:
    bool initialize() {{
        model = tflite::GetModel(eps_guardian::ai_model::g_ai_model_data);
        if (model->version() != TFLITE_SCHEMA_VERSION) return false;
        
        static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
        interpreter = &static_interpreter;
        
        if (interpreter->AllocateTensors() != kTfLiteOk) return false;
        
        input = interpreter->input(0);
        output = interpreter->output(0);
        return true;
    }}

    float detectAnomaly(const float features[{len(self.feature_names)}]) {{
        for (int i = 0; i < {len(self.feature_names)}; i++) {{
            input->data.f[i] = features[i];
        }}
        
        if (interpreter->Invoke() != kTfLiteOk) return -1.0f;
        
        float error = 0.0f;
        for (int i = 0; i < {len(self.feature_names)}; i++) {{
            float diff = input->data.f[i] - output->data.f[i];
            error += diff * diff;
        }}
        return error / {len(self.feature_names)}.0f;
    }}

    int getAnomalyLevel(float error) {{
        if (error < eps_guardian::ai_model::NORMAL_THRESHOLD) return 0;    // Normal
        else if (error < eps_guardian::ai_model::WARNING_THRESHOLD) return 1; // Warning
        else return 2; // Critical
    }}
}};

EPSGuardianAI guardianAI;

void setup() {{
    Serial.begin(115200);
    while (!Serial);
    
    Serial.println("Initialisation EPS Guardian AI...");
    Serial.print("Taille modèle: ");
    Serial.print(eps_guardian::ai_model::g_ai_model_size);
    Serial.println(" bytes");
    
    if (!guardianAI.initialize()) {{
        Serial.println("ERREUR: Initialisation IA échouée");
        return;
    }}
    Serial.println("EPS Guardian AI initialisé avec succès");
    
    // Afficher les features utilisées
    Serial.println("Features utilisées:");
    for (int i = 0; i < {len(self.feature_names)}; i++) {{
        Serial.print("  ");
        Serial.print(i);
        Serial.print(": ");
        Serial.println(eps_guardian::ai_model::FEATURE_NAMES[i]);
    }}
}}

void loop() {{
    // Données capteurs simulées (normalisées)
    float sensor_data[{len(self.feature_names)}] = {{
        0.5, 0.5, 0.5, 0.5, 0.5,  // V_batt, I_batt, T_batt, V_bus, I_bus
        0.5, 0.5, 0.5, 0.5, 0.5,  // V_solar, I_solar, SOC, T_eps, P_batt
        0.5, 0.5, 0.5, 0.5, 0.5,  // P_solar, P_bus, converter_ratio, delta_V_batt, delta_I_batt
        0.5, 0.5, 0.5              // delta_T_batt, rolling_std_V_batt, rolling_mean_V_batt
    }};
    
    float error = guardianAI.detectAnomaly(sensor_data);
    
    if (error < 0) {{
        Serial.println("ERREUR: Inference échouée");
    }} else {{
        int level = guardianAI.getAnomalyLevel(error);
        
        Serial.print("Erreur reconstruction: ");
        Serial.print(error, 6);
        Serial.print(" | Niveau anomalie: ");
        
        switch(level) {{
            case 0: Serial.println("NORMAL"); break;
            case 1: Serial.println("WARNING"); break;
            case 2: Serial.println("CRITICAL"); break;
        }}
    }}
    
    delay(2000);
}}
"""
            arduino_path = os.path.join(self.output_dir, "eps_guardian_esp32.ino")
            with open(arduino_path, 'w') as f:
                f.write(arduino_content)
            print(f"Exemple Arduino généré: {arduino_path}")
            return arduino_path
        except Exception as e:
            print(f"Erreur génération Arduino: {e}")
            return None
    
    def run_conversion(self):
        print("Conversion pour ESP32...")
        print(f"Dossier modèles: {self.model_dir}")
        print(f"Dossier données: {self.training_dir}")  # ← AJOUT
        print(f"Dossier sortie: {self.output_dir}")
        
        if not self.load_tflite_model():
            return False
        
        if not self.load_model_metadata():
            return False
        
        header_path = self.generate_c_header()
        arduino_path = self.generate_arduino_example()
        
        if header_path and arduino_path:
            print("Conversion réussie!")
            print(f"Fichiers générés dans: {self.output_dir}")
            return True
        return False

def main():
    converter = AIModelConverter()
    success = converter.run_conversion()
    
    if success:
        print(" Modèle prêt pour déploiement ESP32!")
        print(" Fichiers créés:")
        print("   - eps_guardian_ai_model.h (Header C++)")
        print("   - eps_guardian_esp32.ino (Exemple Arduino)")
    else:
        print("Conversion échouée")

if __name__ == "__main__":
    main()
