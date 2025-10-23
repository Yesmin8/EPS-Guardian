#include "eps_guardian_ai_model.h"
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>

constexpr int kTensorArenaSize = 12 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

class EPSGuardianAI {
private:
    const tflite::Model* model;
    tflite::MicroInterpreter* interpreter;
    TfLiteTensor* input;
    TfLiteTensor* output;
    tflite::AllOpsResolver resolver;

public:
    bool initialize() {
        model = tflite::GetModel(eps_guardian::ai_model::g_ai_model_data);
        if (model->version() != TFLITE_SCHEMA_VERSION) return false;
        
        static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
        interpreter = &static_interpreter;
        
        if (interpreter->AllocateTensors() != kTfLiteOk) return false;
        
        input = interpreter->input(0);
        output = interpreter->output(0);
        return true;
    }

    float detectAnomaly(const float features[18]) {
        for (int i = 0; i < 18; i++) {
            input->data.f[i] = features[i];
        }
        
        if (interpreter->Invoke() != kTfLiteOk) return -1.0f;
        
        float error = 0.0f;
        for (int i = 0; i < 18; i++) {
            float diff = input->data.f[i] - output->data.f[i];
            error += diff * diff;
        }
        return error / 18.0f;
    }

    int getAnomalyLevel(float error) {
        if (error < eps_guardian::ai_model::NORMAL_THRESHOLD) return 0;    // Normal
        else if (error < eps_guardian::ai_model::WARNING_THRESHOLD) return 1; // Warning
        else return 2; // Critical
    }
};

EPSGuardianAI guardianAI;

void setup() {
    Serial.begin(115200);
    while (!Serial);
    
    Serial.println("Initialisation EPS Guardian AI...");
    Serial.print("Taille modèle: ");
    Serial.print(eps_guardian::ai_model::g_ai_model_size);
    Serial.println(" bytes");
    
    if (!guardianAI.initialize()) {
        Serial.println("ERREUR: Initialisation IA échouée");
        return;
    }
    Serial.println("EPS Guardian AI initialisé avec succès");
    
    // Afficher les features utilisées
    Serial.println("Features utilisées:");
    for (int i = 0; i < 18; i++) {
        Serial.print("  ");
        Serial.print(i);
        Serial.print(": ");
        Serial.println(eps_guardian::ai_model::FEATURE_NAMES[i]);
    }
}

void loop() {
    // Données capteurs simulées (normalisées)
    float sensor_data[18] = {
        0.5, 0.5, 0.5, 0.5, 0.5,  // V_batt, I_batt, T_batt, V_bus, I_bus
        0.5, 0.5, 0.5, 0.5, 0.5,  // V_solar, I_solar, SOC, T_eps, P_batt
        0.5, 0.5, 0.5, 0.5, 0.5,  // P_solar, P_bus, converter_ratio, delta_V_batt, delta_I_batt
        0.5, 0.5, 0.5              // delta_T_batt, rolling_std_V_batt, rolling_mean_V_batt
    };
    
    float error = guardianAI.detectAnomaly(sensor_data);
    
    if (error < 0) {
        Serial.println("ERREUR: Inference échouée");
    } else {
        int level = guardianAI.getAnomalyLevel(error);
        
        Serial.print("Erreur reconstruction: ");
        Serial.print(error, 6);
        Serial.print(" | Niveau anomalie: ");
        
        switch(level) {
            case 0: Serial.println("NORMAL"); break;
            case 1: Serial.println("WARNING"); break;
            case 2: Serial.println("CRITICAL"); break;
        }
    }
    
    delay(2000);
}
