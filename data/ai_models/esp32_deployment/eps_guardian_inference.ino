// ====================================================
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

class EPSGuardianAI {
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
    bool initialize() {
        Serial.println("Initialisation EPS Guardian AI...");
        
        // Chargement du modèle
        model = tflite::GetModel(eps_guardian::ai_model::g_ai_model_data);
        if (model->version() != TFLITE_SCHEMA_VERSION) {
            Serial.println("ERREUR: Version du modèle incompatible!");
            return false;
        }
        
        // Initialisation de l'interpréteur
        static tflite::MicroInterpreter static_interpreter(
            model, resolver, tensor_arena, kTensorArenaSize);
        interpreter = &static_interpreter;
        
        // Allocation des tenseurs
        if (interpreter->AllocateTensors() != kTfLiteOk) {
            Serial.println("ERREUR: Allocation des tenseurs échouée!");
            return false;
        }
        
        input = interpreter->input(0);
        output = interpreter->output(0);
        
        // Configuration des broches
        pinMode(LED_NORMAL, OUTPUT);
        pinMode(LED_WARNING, OUTPUT);  
        pinMode(LED_CRITICAL, OUTPUT);
        
        Serial.println("EPS Guardian AI initialisé avec succès!");
        Serial.println("Attente de données capteurs...");
        
        return true;
    }

    void add_sensor_data(float* sensor_values) {
        // Ajout des nouvelles données à la séquence
        for (int i = 0; i < eps_guardian::ai_model::FEATURE_COUNT; i++) {
            sensor_sequence[sequence_index][i] = sensor_values[i];
        }
        
        sequence_index = (sequence_index + 1) % eps_guardian::ai_model::SEQUENCE_LENGTH;
    }

    float detect_anomaly() {
        // Vérification que la séquence est complète
        if (sequence_index != 0) {
            Serial.println("ATTENTION: Séquence incomplète, utilisation des données disponibles");
        }
        
        // Copie des données dans le tenseur d'entrée
        float* input_data = input->data.f;
        for (int i = 0; i < eps_guardian::ai_model::SEQUENCE_LENGTH; i++) {
            for (int j = 0; j < eps_guardian::ai_model::FEATURE_COUNT; j++) {
                *input_data++ = sensor_sequence[i][j];
            }
        }
        
        // Inférence
        if (interpreter->Invoke() != kTfLiteOk) {
            Serial.println("ERREUR: Inférence échouée!");
            return -1.0f;
        }
        
        // Calcul de l'erreur de reconstruction
        float reconstruction_error = 0.0f;
        float* output_data = output->data.f;
        input_data = input->data.f;
        
        for (int i = 0; i < eps_guardian::ai_model::SEQUENCE_LENGTH * eps_guardian::ai_model::FEATURE_COUNT; i++) {
            float diff = input_data[i] - output_data[i];
            reconstruction_error += diff * diff;
        }
        
        reconstruction_error /= (eps_guardian::ai_model::SEQUENCE_LENGTH * eps_guardian::ai_model::FEATURE_COUNT);
        
        return reconstruction_error;
    }

    int get_anomaly_level(float error) {
        if (error < eps_guardian::ai_model::NORMAL_THRESHOLD) {
            return 0; // NORMAL
        } else if (error < eps_guardian::ai_model::WARNING_THRESHOLD) {
            return 1; // WARNING
        } else {
            return 2; // CRITICAL
        }
    }

    void update_leds(int anomaly_level) {
        // Éteindre toutes les LEDs
        digitalWrite(LED_NORMAL, LOW);
        digitalWrite(LED_WARNING, LOW);
        digitalWrite(LED_CRITICAL, LOW);
        
        // Allumer la LED correspondante
        switch (anomaly_level) {
            case 0: // NORMAL
                digitalWrite(LED_NORMAL, HIGH);
                break;
            case 1: // WARNING
                digitalWrite(LED_WARNING, HIGH); 
                break;
            case 2: // CRITICAL
                digitalWrite(LED_CRITICAL, HIGH);
                break;
        }
    }

    void print_debug_info(float error, int level) {
        Serial.print("Erreur reconstruction: ");
        Serial.print(error, 6);
        Serial.print(" | Niveau: ");
        
        switch (level) {
            case 0: Serial.println("NORMAL"); break;
            case 1: Serial.println("WARNING"); break;
            case 2: Serial.println("CRITICAL"); break;
        }
        
        Serial.print("Seuils - Normal<");
        Serial.print(eps_guardian::ai_model::NORMAL_THRESHOLD, 6);
        Serial.print(", Warning<");  
        Serial.print(eps_guardian::ai_model::WARNING_THRESHOLD, 6);
        Serial.print(", Critical>=");
        Serial.println(eps_guardian::ai_model::CRITICAL_THRESHOLD, 6);
    }
};

// ====================================================
// INSTANCE GLOBALE ET SETUP
// ====================================================

EPSGuardianAI guardianAI;

void setup() {
    Serial.begin(115200);
    while (!Serial) { delay(10); }
    
    Serial.println("================================================");
    Serial.println("EPS GUARDIAN - SYSTÈME IA EMBARQUÉ");
    Serial.println("Détection d'anomalies EPS en temps réel");
    Serial.println("================================================");
    
    if (!guardianAI.initialize()) {
        Serial.println("ÉCHEC: Initialisation IA - Arrêt du système");
        while(1) { delay(1000); }
    }
}

// ====================================================
// BOUCLE PRINCIPALE
// ====================================================

void loop() {
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
    
    if (error >= 0) {
        int anomaly_level = guardianAI.get_anomaly_level(error);
        guardianAI.update_leds(anomaly_level);
        guardianAI.print_debug_info(error, anomaly_level);
    }
    
    // Cycle toutes les 2 secondes
    delay(2000);
}
