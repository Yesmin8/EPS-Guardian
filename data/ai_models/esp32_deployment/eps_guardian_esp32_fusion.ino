// ======================================
// EPS GUARDIAN ESP32
// Règles Déterministes + IA Autoencodeur
// Système Hybride
// ======================================

#include "eps_guardian_ai_model.h"
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

// ======================
// CONFIGURATION MÉMOIRE
// ======================
constexpr int kTensorArenaSize = 12 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

// ======================
// SEUILS RÈGLES DÉTERMINISTES (R1-R7)
// ======================
constexpr float RULE_TEMP_CRITICAL = 60.0f;      // R1 - Surchauffe batterie
constexpr float RULE_CURRENT_CRITICAL = 3.0f;    // R2 - Surcharge courant
constexpr float RULE_VOLTAGE_CRITICAL = 3.2f;    // R3 - Décharge profonde
constexpr float RULE_RATIO_CRITICAL = 0.7f;      // R4 - Ratio DC/DC
constexpr float RULE_SENSOR_FAULT = 120.0f;      // R6 - Défaut capteur
constexpr float RULE_OSCILLATION_V = 0.5f;       // R5 - Oscillation tension
constexpr float RULE_OSCILLATION_T = 5.0f;       // R5 - Oscillation température

// ======================
// PINS ESP32
// ======================
const int LED_NORMAL = 2;
const int LED_WARNING = 4;
const int LED_CRITICAL = 5;
const int MOSFET_PIN = 12;
const int BUZZER_PIN = 13;

// ======================
// CLASSE PRINCIPALE
// ======================
class EPSGuardianFusion {
private:
    // Composants IA
    const tflite::Model* ai_model;
    tflite::MicroInterpreter* ai_interpreter;
    TfLiteTensor* ai_input;
    TfLiteTensor* ai_output;
    tflite::AllOpsResolver resolver;

    // État système et historique
    struct SystemState {
        float v_batt = 7.4f;
        float i_batt = 1.2f;
        float t_batt = 35.0f;
        float v_bus = 7.3f;
        float i_bus = 0.8f;
        float v_solar = 14.5f;
        float i_solar = 1.1f;
        bool mosfet_enabled = true;
        bool charge_enabled = true;
        int anomaly_level = 0; // 0=Normal, 1=Warning, 2=Critical
        unsigned long last_cycle_time = 0;
    } state;

    // Historique pour calcul des deltas
    float prev_v_batt = 7.4f;
    float prev_t_batt = 35.0f;

public:
    // ======================
    // INITIALISATION IA
    // ======================
    bool initializeAI() {
        Serial.println("Initialisation modèle IA...");
        
        // Charger le modèle
        ai_model = tflite::GetModel(eps_guardian::ai_model::g_ai_model_data);
        if (ai_model->version() != TFLITE_SCHEMA_VERSION) {
            Serial.println("Version modèle incompatible");
            return false;
        }

        // Créer l'interpréteur
        static tflite::MicroInterpreter static_interpreter(
            ai_model, resolver, tensor_arena, kTensorArenaSize);
        ai_interpreter = &static_interpreter;

        // Allouer la mémoire
        if (ai_interpreter->AllocateTensors() != kTfLiteOk) {
            Serial.println("Erreur allocation mémoire IA");
            return false;
        }

        ai_input = ai_interpreter->input(0);
        ai_output = ai_interpreter->output(0);
        
        Serial.println("Modèle IA initialisé");
        Serial.print("Taille modèle: ");
        Serial.print(eps_guardian::ai_model::g_ai_model_size);
        Serial.println(" bytes");
        
        return true;
    }

    // ======================
    // RÈGLES DÉTERMINISTES (R1-R7)
    // ======================
    int executeSafetyRules() {
        // R1: Surchauffe batterie → Coupure MOSFET
        if (state.t_batt > RULE_TEMP_CRITICAL) {
            Serial.println("R1: Surchauffe batterie détectée");
            state.mosfet_enabled = false;
            return 2; // CRITICAL
        }
        
        // R2: Surcharge courant → Limitation charge
        if (abs(state.i_batt) > RULE_CURRENT_CRITICAL) {
            Serial.println("R2: Surcharge courant détectée");
            state.charge_enabled = false;
            return 2; // CRITICAL
        }
        
        // R3: Décharge profonde → Isolation batterie
        if (state.v_batt < RULE_VOLTAGE_CRITICAL && state.i_batt < 0) {
            Serial.println("R3: Décharge profonde détectée");
            state.mosfet_enabled = false;
            return 2; // CRITICAL
        }
        
        // R4: Ratio DC/DC anormal → Réduction charge
        float ratio = (state.v_solar > 0.1f) ? state.v_bus / state.v_solar : 0.0f;
        if (ratio < RULE_RATIO_CRITICAL && ratio > 0.01f) {
            Serial.println("R4: Ratio DC/DC anormal");
            state.charge_enabled = false;
            return 1; // WARNING
        }
        
        // R5: Oscillation bus → Logging augmenté
        float delta_v = abs(state.v_batt - prev_v_batt);
        float delta_t = abs(state.t_batt - prev_t_batt);
        
        if (delta_v > RULE_OSCILLATION_V || delta_t > RULE_OSCILLATION_T) {
            Serial.println("R5: Oscillation détectée");
            return 1; // WARNING
        }
        
        // R6: Défaut capteur → Mode sans échec
        if (state.t_batt > RULE_SENSOR_FAULT || state.v_batt > 20.0f) {
            Serial.println("R6: Défaut capteur suspecté");
            return 1; // WARNING
        }
        
        // R7: État normal → LED verte
        return 0; // NORMAL
    }

    // ======================
    // DÉTECTION IA AUTOENCODEUR
    // ======================
    float detectAIAnomaly() {
        // Calculer les deltas pour features temporelles
        float delta_v_batt = state.v_batt - prev_v_batt;
        float delta_t_batt = state.t_batt - prev_t_batt;
        
        // Préparer les 11 features exactement comme l'entraînement
        float features[11] = {
            state.v_batt, state.i_batt, state.t_batt,
            state.v_bus, state.i_bus, 
            state.v_solar, state.i_solar,
            state.v_batt * state.i_batt,                    // P_batt
            (state.v_solar > 0.1f) ? state.v_bus / state.v_solar : 0.0f, // converter_ratio
            delta_v_batt,                                   // delta_V_batt
            delta_t_batt                                    // delta_T_batt
        };
        
        // Copier dans le tensor d'entrée
        for (int i = 0; i < 11; i++) {
            ai_input->data.f[i] = features[i];
        }
        
        // Exécuter l'inférence
        if (ai_interpreter->Invoke() != kTfLiteOk) {
            Serial.println("Erreur inférence IA");
            return -1.0f;
        }
        
        // Calculer l'erreur de reconstruction (MSE)
        float reconstruction_error = 0.0f;
        for (int i = 0; i < 11; i++) {
            float diff = ai_input->data.f[i] - ai_output->data.f[i];
            reconstruction_error += diff * diff;
        }
        reconstruction_error /= 11.0f;
        
        return reconstruction_error;
    }

    int getAIAnomalyLevel(float error) {
        if (error < 0) {
            return -1; // ERREUR
        } else if (error < eps_guardian::ai_model::NORMAL_THRESHOLD) {
            return 0;  // NORMAL
        } else if (error < eps_guardian::ai_model::WARNING_THRESHOLD) {
            return 1;  // WARNING
        } else {
            return 2;  // CRITICAL
        }
    }

    // ======================
    // DÉCISION HYBRIDE
    // ======================
    int hybridDecision() {
        // 1. EXÉCUTER RÈGLES DÉTERMINISTES (Priorité haute)
        int rule_level = executeSafetyRules();
        if (rule_level == 2) { // CRITICAL par règles → Action immédiate
            Serial.println("Décision: Règles critiques → Action immédiate");
            takeEmergencyAction();
            return 2;
        }
        
        // 2. DÉTECTION IA (Patterns complexes)
        float ai_error = detectAIAnomaly();
        int ai_level = getAIAnomalyLevel(ai_error);
        
        if (ai_level == -1) {
            Serial.println("IA en erreur, confiance réduite");
            ai_level = 0; // Fallback vers normal
        }
        
        // 3. DÉCISION FINALE - Le pire des deux
        int final_level = max(rule_level, ai_level);
        state.anomaly_level = final_level;
        
        // Mettre à jour l'historique
        prev_v_batt = state.v_batt;
        prev_t_batt = state.t_batt;
        
        return final_level;
    }

    // ======================
    // ACTIONS PHYSIQUES
    // ======================
    void takeEmergencyAction() {
        // Actions critiques immédiates
        digitalWrite(LED_CRITICAL, HIGH);
        digitalWrite(LED_WARNING, LOW);
        digitalWrite(LED_NORMAL, LOW);
        
        // Coupure de sécurité
        state.mosfet_enabled = false;
        state.charge_enabled = false;
        digitalWrite(MOSFET_PIN, LOW);
        
        // Alarme sonore
        tone(BUZZER_PIN, 1000, 1000);
        
        Serial.println("ACTION D'URGENCE: Coupure sécurité activée");
    }

    void takePreventiveAction(int level) {
        // Réinitialiser toutes les LEDs
        digitalWrite(LED_CRITICAL, LOW);
        digitalWrite(LED_WARNING, LOW);
        digitalWrite(LED_NORMAL, LOW);
        
        switch(level) {
            case 0: // NORMAL
                digitalWrite(LED_NORMAL, HIGH);
                state.mosfet_enabled = true;
                state.charge_enabled = true;
                digitalWrite(MOSFET_PIN, HIGH);
                noTone(BUZZER_PIN);
                break;
                
            case 1: // WARNING
                digitalWrite(LED_WARNING, HIGH);
                state.charge_enabled = false; // Réduction charge
                tone(BUZZER_PIN, 500, 200); // Bip court
                Serial.println("ACTION: Réduction charge active");
                break;
                
            case 2: // CRITICAL  
                takeEmergencyAction();
                break;
        }
    }

    // ======================
    // MISE À JOUR CAPTEURS (Simulation/Réel)
    // ======================
    void updateSensorReadings() {
        // REMPLACER PAR VRAIES LECTURES CAPTEURS
        // Simulation: légères variations aléatoires + scénarios tests
        
        static int cycle_count = 0;
        cycle_count++;
        
        // Scénario test: toutes les 20 cycles, simuler une anomalie
        if (cycle_count % 20 == 0) {
            // Simuler une surchauffe
            state.t_batt = 65.0f;
            state.i_batt = 3.5f;
        } else if (cycle_count % 15 == 0) {
            // Simuler une anomalie subtile (détectable par IA)
            state.v_batt = 5.8f;
            state.v_solar = 25.0f;
        } else {
            // Comportement normal avec bruit
            state.v_batt += random(-10, 10) * 0.001f;
            state.i_batt += random(-5, 5) * 0.01f;
            state.t_batt += random(-3, 3) * 0.05f;
            state.v_bus += random(-10, 10) * 0.001f;
            state.i_bus += random(-2, 2) * 0.01f;
            state.v_solar += random(-20, 20) * 0.002f;
            state.i_solar += random(-10, 10) * 0.01f;
        }
        
        // Limites physiques réalistes
        state.v_batt = constrain(state.v_batt, 2.5f, 8.5f);
        state.i_batt = constrain(state.i_batt, -4.0f, 4.0f);
        state.t_batt = constrain(state.t_batt, -10.0f, 80.0f);
        state.v_bus = constrain(state.v_bus, 5.0f, 9.0f);
        state.v_solar = constrain(state.v_solar, 0.0f, 30.0f);
    }

    // ======================
    // AFFICHAGE STATUT
    // ======================
    void printSystemStatus() {
        Serial.print("Batterie: ");
        Serial.print(state.v_batt, 2);
        Serial.print("V, ");
        Serial.print(state.i_batt, 2);
        Serial.print("A, ");
        Serial.print(state.t_batt, 1);
        Serial.println("°C");
        
        Serial.print("Bus: ");
        Serial.print(state.v_bus, 2);
        Serial.print("V, ");
        Serial.print(state.i_bus, 2);
        Serial.println("A");
        
        Serial.print("Solar: ");
        Serial.print(state.v_solar, 2);
        Serial.print("V, ");
        Serial.print(state.i_solar, 2);
        Serial.println("A");
        
        Serial.print("MOSFET: ");
        Serial.print(state.mosfet_enabled ? "ON" : "OFF");
        Serial.print(" | Charge: ");
        Serial.println(state.charge_enabled ? "ON" : "OFF");
    }
};

// ======================
// INSTANCE GLOBALE
// ======================
EPSGuardianFusion guardian;

// ======================
// SETUP
// ======================
void setup() {
    Serial.begin(115200);
    delay(1000); // Attendre la connexion série
    
    Serial.println("\n\n EPS GUARDIAN FUSION - Démarrage");
    Serial.println("==========================================");
    
    // Configuration des pins
    pinMode(LED_NORMAL, OUTPUT);
    pinMode(LED_WARNING, OUTPUT);
    pinMode(LED_CRITICAL, OUTPUT);
    pinMode(MOSFET_PIN, OUTPUT);
    pinMode(BUZZER_PIN, OUTPUT);
    
    // Séquence de démarrage
    digitalWrite(LED_NORMAL, HIGH);
    delay(300);
    digitalWrite(LED_WARNING, HIGH);
    delay(300);
    digitalWrite(LED_CRITICAL, HIGH);
    delay(300);
    digitalWrite(LED_NORMAL, LOW);
    digitalWrite(LED_WARNING, LOW);
    digitalWrite(LED_CRITICAL, LOW);
    
    // Initialisation IA
    if (guardian.initializeAI()) {
        Serial.println("Système hybride initialisé");
    } else {
        Serial.println("Échec initialisation - Mode sans échec");
        // Continuer avec règles seulement
    }
    
    Serial.println("Prêt pour surveillance temps réel");
    Serial.println("==========================================");
}

// ======================
// BOUCLE PRINCIPALE
// ======================
void loop() {
    Serial.println("\n--- Cycle de surveillance ---");
    
    // 1. Mettre à jour les lectures capteurs
    guardian.updateSensorReadings();
    
    // 2. Décision hybride (Règles + IA)
    int anomaly_level = guardian.hybridDecision();
    
    // 3. Actions selon le niveau
    guardian.takePreventiveAction(anomaly_level);
    
    // 4. Affichage statut
    Serial.print("Décision finale: ");
    switch(anomaly_level) {
        case 0: Serial.println("NORMAL"); break;
        case 1: Serial.println("WARNING"); break;
        case 2: Serial.println("CRITICAL"); break;
        default: Serial.println("INCONNU");
    }
    
    guardian.printSystemStatus();
    
    delay(2000); // Cycle de 2 secondes
}