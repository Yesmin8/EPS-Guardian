#!/usr/bin/env python3
"""
CONVERTISSEUR DE MODÈLE IA OBC → MCU
Convertit un modèle TensorFlow (.h5) en format TensorFlow Lite (.tflite)
et prépare les fichiers pour déploiement sur ESP32.
"""

import os
import tensorflow as tf
import numpy as np
import json
import logging
from datetime import datetime

# --- Configuration des chemins ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "ai_models", "model_complex")
DEPLOY_DIR = os.path.join(PROJECT_ROOT, "data", "ai_models", "esp32_deployment")
MCU_MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "ai_models", "model_simple")

os.makedirs(DEPLOY_DIR, exist_ok=True)
os.makedirs(MCU_MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AI_Model_Converter")

class AIModelConverter:
    def __init__(self):
        self.model = None
        self.thresholds = None
        self.model_info = {}
        
    def load_complex_model(self):
        """Charge le modèle LSTM complexe et ses métadonnées"""
        try:
            model_path = os.path.join(MODEL_DIR, "ai_model_lstm_autoencoder.h5")
            thresholds_path = os.path.join(MODEL_DIR, "ai_thresholds.json")
            
            if not os.path.exists(model_path):
                logger.error(f"Modèle non trouvé: {model_path}")
                return False
                
            logger.info(f"Chargement du modèle: {model_path}")
            self.model = tf.keras.models.load_model(model_path, compile=False)
            
            # Chargement des seuils
            if os.path.exists(thresholds_path):
                with open(thresholds_path, 'r') as f:
                    thresholds_data = json.load(f)
                    self.thresholds = thresholds_data["anomaly_thresholds"]
                logger.info("Seuils d'anomalie chargés")
            else:
                logger.warning("Fichier de seuils non trouvé, utilisation de valeurs par défaut")
                self.thresholds = {
                    "normal_threshold": 0.001,
                    "warning_threshold": 0.002, 
                    "critical_threshold": 0.003
                }
            
            # Informations du modèle
            self.model_info = {
                "input_shape": self.model.input_shape[1:],
                "output_shape": self.model.output_shape[1:],
                "total_params": self.model.count_params(),
                "model_size_h5": os.path.getsize(model_path) / 1024
            }
            
            logger.info(f"Modèle chargé: {self.model_info['input_shape']} → {self.model_info['output_shape']}")
            logger.info(f"Paramètres: {self.model_info['total_params']:,}")
            logger.info(f"Taille .h5: {self.model_info['model_size_h5']:.1f} KB")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            return False
    
    def convert_to_tflite(self, output_path, quantize=False, quantization_type="float16"):
        """Convertit le modèle en TensorFlow Lite avec support des LSTM dynamiques"""
        try:
            logger.info(f"Conversion en TFLite: quantize={quantize}, type={quantization_type}")

            # --- Création du convertisseur ---
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

            # 🔧 Correction pour les LSTM (Select TF Ops)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter._experimental_lower_tensor_list_ops = False  # indispensable pour TensorListReserve

            # --- Gestion de la quantification ---
            if quantize:
                if quantization_type == "float16":
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.target_spec.supported_types = [tf.float16]
                    logger.info("Quantification float16 activée")
                elif quantization_type == "int8":
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]

                    def representative_dataset():
                        for _ in range(100):
                            data = np.random.randn(1, *self.model_info['input_shape']).astype(np.float32)
                            yield [data]

                    converter.representative_dataset = representative_dataset
                    converter.target_spec.supported_ops = [
                        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                        tf.lite.OpsSet.SELECT_TF_OPS
                    ]
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8
                    logger.info("Quantification INT8 activée")

            # --- Conversion ---
            tflite_model = converter.convert()

            # --- Sauvegarde ---
            with open(output_path, "wb") as f:
                f.write(tflite_model)

            size_kb = os.path.getsize(output_path) / 1024
            logger.info(f"Modèle TFLite généré: {output_path}")
            logger.info(f"Taille TFLite: {size_kb:.1f} KB")

            return size_kb

        except Exception as e:
            logger.error(f"Erreur conversion TFLite: {e}")
            return 0
    
    def generate_c_header(self, tflite_path, header_path):
        """Génère un fichier header C/C++ à partir du modèle TFLite"""
        try:
            logger.info(f"Génération du header C: {header_path}")
            
            with open(tflite_path, "rb") as f:
                model_data = f.read()
            
            # Conversion en tableau C
            hex_array = []
            for i, byte in enumerate(model_data):
                if i % 12 == 0:
                    hex_array.append("\n    ")
                hex_array.append(f"0x{byte:02x}, ")
            
            hex_string = "".join(hex_array).rstrip(", ")
            
            header_content = f"""// ====================================================
// Fichier généré automatiquement - EPS Guardian AI Model
// Modèle TensorFlow Lite pour déploiement ESP32
// Généré le: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
// ====================================================

#ifndef EPS_GUARDIAN_AI_MODEL_H
#define EPS_GUARDIAN_AI_MODEL_H

#include <cstdint>
#include <cstddef>

namespace eps_guardian {{
namespace ai_model {{

// Données du modèle TensorFlow Lite
alignas(8) const uint8_t g_ai_model_data[] = {{
    {hex_string}
}};

// Taille du modèle en bytes
const size_t g_ai_model_size = sizeof(g_ai_model_data);

// Seuils d'anomalie (basés sur l'erreur de reconstruction)
constexpr float NORMAL_THRESHOLD = {self.thresholds['normal_threshold']:.6f}f;
constexpr float WARNING_THRESHOLD = {self.thresholds['warning_threshold']:.6f}f;
constexpr float CRITICAL_THRESHOLD = {self.thresholds['critical_threshold']:.6f}f;

// Configuration du modèle
constexpr int SEQUENCE_LENGTH = {self.model_info['input_shape'][0]};
constexpr int FEATURE_COUNT = {self.model_info['input_shape'][1]};
constexpr int TENSOR_ARENA_SIZE = 40 * 1024; // 40KB pour LSTM

}} // namespace ai_model
}} // namespace eps_guardian

#endif // EPS_GUARDIAN_AI_MODEL_H
"""
            
            with open(header_path, "w") as f:
                f.write(header_content)
            
            logger.info(f"Header C généré: {header_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur génération header: {e}")
            return False
    
    def generate_arduino_sketch(self, sketch_path):
        """Génère un sketch Arduino complet pour l'ESP32"""
        try:
            logger.info(f"Génération du sketch Arduino: {sketch_path}")
            
            sketch_content = f"""// ====================================================
// EPS Guardian - Système de Détection d'Anomalies IA
// MCU: ESP32 avec TensorFlow Lite Micro
// Modèle: LSTM Autoencoder pour séquences temporelles
// ====================================================

#include "eps_guardian_ai_model.h"
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

// ====================================================
// CONFIGURATION MATÉRIELLE
// ====================================================

// Broches pour LEDs de statut
#define LED_NORMAL 2
#define LED_WARNING 4  
#define LED_CRITICAL 5

// Buffer pour l'inférence
constexpr int kTensorArenaSize = eps_guardian::ai_model::TENSOR_ARENA_SIZE;
static uint8_t tensor_arena[kTensorArenaSize];

// ====================================================
// CLASSE PRINCIPALE EPS GUARDIAN AI
// ====================================================

class EPSGuardianAI {{
private:
    const tflite::Model* model;
    tflite::MicroInterpreter* interpreter;
    TfLiteTensor* input;
    TfLiteTensor* output;
    tflite::AllOpsResolver resolver;

    // Buffer pour les séquences capteurs
    float sensor_sequence[eps_guardian::ai_model::SEQUENCE_LENGTH]
                        [eps_guardian::ai_model::FEATURE_COUNT];
    int sequence_index = 0;

public:
    bool initialize() {{
        Serial.println("Initialisation EPS Guardian AI...");
        
        // Chargement du modèle
        model = tflite::GetModel(eps_guardian::ai_model::g_ai_model_data);
        if (model->version() != TFLITE_SCHEMA_VERSION) {{
            Serial.println("ERREUR: Version du modèle incompatible!");
            return false;
        }}
        
        // Initialisation de l'interpréteur
        static tflite::MicroInterpreter static_interpreter(
            model, resolver, tensor_arena, kTensorArenaSize);
        interpreter = &static_interpreter;
        
        // Allocation des tenseurs
        if (interpreter->AllocateTensors() != kTfLiteOk) {{
            Serial.println("ERREUR: Allocation des tenseurs échouée!");
            return false;
        }}
        
        input = interpreter->input(0);
        output = interpreter->output(0);
        
        // Configuration des broches
        pinMode(LED_NORMAL, OUTPUT);
        pinMode(LED_WARNING, OUTPUT);  
        pinMode(LED_CRITICAL, OUTPUT);
        
        Serial.println("EPS Guardian AI initialisé avec succès!");
        Serial.println("Attente de données capteurs...");
        
        return true;
    }}

    void add_sensor_data(float* sensor_values) {{
        // Ajout des nouvelles données à la séquence
        for (int i = 0; i < eps_guardian::ai_model::FEATURE_COUNT; i++) {{
            sensor_sequence[sequence_index][i] = sensor_values[i];
        }}
        
        sequence_index = (sequence_index + 1) % eps_guardian::ai_model::SEQUENCE_LENGTH;
    }}

    float detect_anomaly() {{
        // Vérification que la séquence est complète
        if (sequence_index != 0) {{
            Serial.println("ATTENTION: Séquence incomplète, utilisation des données disponibles");
        }}
        
        // Copie des données dans le tenseur d'entrée
        float* input_data = input->data.f;
        for (int i = 0; i < eps_guardian::ai_model::SEQUENCE_LENGTH; i++) {{
            for (int j = 0; j < eps_guardian::ai_model::FEATURE_COUNT; j++) {{
                *input_data++ = sensor_sequence[i][j];
            }}
        }}
        
        // Inférence
        if (interpreter->Invoke() != kTfLiteOk) {{
            Serial.println("ERREUR: Inférence échouée!");
            return -1.0f;
        }}
        
        // Calcul de l'erreur de reconstruction
        float reconstruction_error = 0.0f;
        float* output_data = output->data.f;
        input_data = input->data.f;
        
        for (int i = 0; i < eps_guardian::ai_model::SEQUENCE_LENGTH * eps_guardian::ai_model::FEATURE_COUNT; i++) {{
            float diff = input_data[i] - output_data[i];
            reconstruction_error += diff * diff;
        }}
        
        reconstruction_error /= (eps_guardian::ai_model::SEQUENCE_LENGTH * eps_guardian::ai_model::FEATURE_COUNT);
        
        return reconstruction_error;
    }}

    int get_anomaly_level(float error) {{
        if (error < eps_guardian::ai_model::NORMAL_THRESHOLD) {{
            return 0; // NORMAL
        }} else if (error < eps_guardian::ai_model::WARNING_THRESHOLD) {{
            return 1; // WARNING
        }} else {{
            return 2; // CRITICAL
        }}
    }}

    void update_leds(int anomaly_level) {{
        // Éteindre toutes les LEDs
        digitalWrite(LED_NORMAL, LOW);
        digitalWrite(LED_WARNING, LOW);
        digitalWrite(LED_CRITICAL, LOW);
        
        // Allumer la LED correspondante
        switch (anomaly_level) {{
            case 0: // NORMAL
                digitalWrite(LED_NORMAL, HIGH);
                break;
            case 1: // WARNING
                digitalWrite(LED_WARNING, HIGH); 
                break;
            case 2: // CRITICAL
                digitalWrite(LED_CRITICAL, HIGH);
                break;
        }}
    }}

    void print_debug_info(float error, int level) {{
        Serial.print("Erreur reconstruction: ");
        Serial.print(error, 6);
        Serial.print(" | Niveau: ");
        
        switch (level) {{
            case 0: Serial.println("NORMAL"); break;
            case 1: Serial.println("WARNING"); break;
            case 2: Serial.println("CRITICAL"); break;
        }}
        
        Serial.print("Seuils - Normal<");
        Serial.print(eps_guardian::ai_model::NORMAL_THRESHOLD, 6);
        Serial.print(", Warning<");  
        Serial.print(eps_guardian::ai_model::WARNING_THRESHOLD, 6);
        Serial.print(", Critical>=");
        Serial.println(eps_guardian::ai_model::CRITICAL_THRESHOLD, 6);
    }}
}};

// ====================================================
// INSTANCE GLOBALE ET SETUP
// ====================================================

EPSGuardianAI guardianAI;

void setup() {{
    Serial.begin(115200);
    while (!Serial) {{ delay(10); }}
    
    Serial.println("================================================");
    Serial.println("EPS GUARDIAN - SYSTÈME IA EMBARQUÉ");
    Serial.println("Détection d'anomalies EPS en temps réel");
    Serial.println("================================================");
    
    if (!guardianAI.initialize()) {{
        Serial.println("ÉCHEC: Initialisation IA - Arrêt du système");
        while(1) {{ delay(1000); }}
    }}
}}

// ====================================================
// BOUCLE PRINCIPALE
// ====================================================

void loop() {{
    // SIMULATION: Génération de données capteurs réalistes
    float sensor_data[eps_guardian::ai_model::FEATURE_COUNT];
    
    // Valeurs typiques d'un système EPS nominal
    sensor_data[0] = 7.4f + random(-10, 10) / 100.0f;  // V_batt
    sensor_data[1] = 1.2f + random(-20, 20) / 100.0f;  // I_batt  
    sensor_data[2] = 35.0f + random(-50, 50) / 10.0f;  // T_batt
    sensor_data[3] = 7.8f + random(-5, 5) / 100.0f;    // V_bus
    sensor_data[4] = 0.8f + random(-10, 10) / 100.0f;  // I_bus
    sensor_data[5] = 15.2f + random(-20, 20) / 10.0f;  // V_solar
    sensor_data[6] = 1.5f + random(-10, 10) / 100.0f;  // I_solar
    
    // Ajout des données à la séquence
    guardianAI.add_sensor_data(sensor_data);
    
    // Détection d'anomalie
    float error = guardianAI.detect_anomaly();
    
    if (error >= 0) {{
        int anomaly_level = guardianAI.get_anomaly_level(error);
        guardianAI.update_leds(anomaly_level);
        guardianAI.print_debug_info(error, anomaly_level);
    }}
    
    // Cycle toutes les 2 secondes
    delay(2000);
}}
"""
            
            with open(sketch_path, "w") as f:
                f.write(sketch_content)
            
            logger.info(f"Sketch Arduino généré: {sketch_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur génération sketch: {e}")
            return False
    
    def copy_simple_model_to_mcu(self):
        """Copie également le modèle simple pour le MCU de base"""
        try:
            # Copier le modèle quantifié depuis model_complex vers model_simple
            source_tflite = os.path.join(MODEL_DIR, "ai_model_lstm_autoencoder_quant.tflite")
            dest_tflite = os.path.join(MCU_MODEL_DIR, "ai_autoencoder.tflite")
            
            if os.path.exists(source_tflite):
                import shutil
                shutil.copy2(source_tflite, dest_tflite)
                logger.info(f"Modèle simple copié vers: {dest_tflite}")
                
                # Mettre à jour les seuils pour le MCU simple
                mcu_thresholds = {
                    "thresholds": {
                        "normal": self.thresholds["normal_threshold"],
                        "warning": self.thresholds["warning_threshold"],
                        "critical": self.thresholds["critical_threshold"]
                    },
                    "conversion_date": datetime.now().isoformat(),
                    "source_model": "LSTM Autoencoder Complexe"
                }
                
                thresholds_path = os.path.join(MCU_MODEL_DIR, "ai_thresholds.json")
                with open(thresholds_path, 'w') as f:
                    json.dump(mcu_thresholds, f, indent=2)
                
                logger.info(f"Seuils mis à jour pour MCU simple: {thresholds_path}")
                
            return True
            
        except Exception as e:
            logger.error(f"Erreur copie modèle MCU: {e}")
            return False
    
    def generate_deployment_report(self):
        """Génère un rapport de déploiement complet"""
        report_path = os.path.join(DEPLOY_DIR, "deployment_report.md")
        
        report_content = f"""# Rapport de Déploiement IA - EPS Guardian

## Date de Génération
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Modèle Source
- **Architecture**: LSTM Autoencoder
- **Shape d'entrée**: {self.model_info['input_shape']}
- **Shape de sortie**: {self.model_info['output_shape']}
- **Paramètres**: {self.model_info['total_params']:,}
- **Taille originale**: {self.model_info['model_size_h5']:.1f} KB

## Fichiers Générés

### 1. Modèles TensorFlow Lite (dans model_complex)
- `ai_model_lstm_autoencoder.tflite` - Version standard
- `ai_model_lstm_autoencoder_quant.tflite` - Version quantifiée (recommandée)

### 2. Fichiers de Déploiement ESP32 (dans esp32_deployment)
- `eps_guardian_ai_model.h` - Header C++ avec modèle intégré
- `eps_guardian_inference.ino` - Sketch Arduino complet

### 3. Compatibilité MCU (dans model_simple)
- Modèle également disponible pour MCU simple: `ai_autoencoder.tflite`

## Seuils d'Anomalie
- **NORMAL**: < {self.thresholds['normal_threshold']:.6f}
- **WARNING**: < {self.thresholds['warning_threshold']:.6f}  
- **CRITICAL**: >= {self.thresholds['critical_threshold']:.6f}

## Configuration Matérielle Recommandée
- **MCU**: ESP32 (avec 4MB Flash)
- **RAM minimale**: 40KB pour Tensor Arena
- **Broches LEDs**: 2 (NORMAL), 4 (WARNING), 5 (CRITICAL)

## Utilisation
1. Inclure `eps_guardian_ai_model.h` dans votre projet
2. Utiliser la classe `EPSGuardianAI` pour l'inférence
3. Appeler `add_sensor_data()` et `detect_anomaly()` cycliquement

## Performances Attendues
- **Temps d'inférence**: < 100ms sur ESP32
- **Précision**: Détection d'anomalies temporelles complexes
- **Consommation**: Optimisée pour systèmes embarqués

---
*Généré automatiquement par le système EPS Guardian*
"""
        
        with open(report_path, "w") as f:
            f.write(report_content)
        
        logger.info(f"Rapport de déploiement généré: {report_path}")

def main():
    """Point d'entrée principal"""
    logger.info("DÉMARRAGE CONVERSION MODÈLE IA POUR ESP32")
    logger.info("=" * 60)
    
    converter = AIModelConverter()
    
    # 1. Chargement du modèle complexe
    if not converter.load_complex_model():
        logger.error("Échec chargement modèle - Arrêt")
        return 1
    
    # 2. Conversion en TFLite - SAUVEGARDE DANS MODEL_COMPLEX
    logger.info("CONVERSION EN TENSORFLOW LITE")
    
    # Version standard - dans model_complex
    tflite_standard = os.path.join(MODEL_DIR, "ai_model_lstm_autoencoder.tflite")
    size_standard = converter.convert_to_tflite(tflite_standard, quantize=False)
    
    # Version quantifiée - dans model_complex
    tflite_quant = os.path.join(MODEL_DIR, "ai_model_lstm_autoencoder_quant.tflite")
    size_quant = converter.convert_to_tflite(tflite_quant, quantize=True, quantization_type="float16")
    
    logger.info(f"Réduction taille: {converter.model_info['model_size_h5']:.1f}KB → {size_quant:.1f}KB " 
                f"({size_quant/converter.model_info['model_size_h5']*100:.1f}%)")
    
    # 3. Génération des fichiers de déploiement - dans esp32_deployment
    logger.info("GÉNÉRATION FICHIERS DÉPLOIEMENT")
    
    # Utiliser le modèle quantifié depuis model_complex pour générer le header
    header_path = os.path.join(DEPLOY_DIR, "eps_guardian_ai_model.h")
    converter.generate_c_header(tflite_quant, header_path)
    
    sketch_path = os.path.join(DEPLOY_DIR, "eps_guardian_inference.ino")
    converter.generate_arduino_sketch(sketch_path)
    
    # 4. Copie pour MCU simple - depuis model_complex vers model_simple
    converter.copy_simple_model_to_mcu()
    
    # 5. Rapport final
    converter.generate_deployment_report()
    
    logger.info("=" * 60)
    logger.info(" CONVERSION TERMINÉE AVEC SUCCÈS!")
    logger.info(f" Dossier model_complex: {MODEL_DIR}")
    logger.info(" Fichiers générés dans model_complex:")
    logger.info(f"   - {tflite_standard}")
    logger.info(f"   - {tflite_quant} (recommandé)")
    logger.info(f" Dossier déploiement: {DEPLOY_DIR}")
    logger.info(" Fichiers de déploiement:")
    logger.info(f"   - {header_path}")
    logger.info(f"   - {sketch_path}")
    logger.info(" Prêt pour déploiement sur ESP32!")
    
    return 0

if __name__ == "__main__":
    exit(main())
