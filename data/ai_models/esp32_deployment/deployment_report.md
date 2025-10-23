# Rapport de Déploiement IA - EPS Guardian

## Date de Génération
2025-10-23 17:08:49

## Modèle Source
- **Architecture**: LSTM Autoencoder
- **Shape d'entrée**: (30, 7)
- **Shape de sortie**: (30, 7)
- **Paramètres**: 65,511
- **Taille originale**: 827.6 KB

## Fichiers Générés

### 1. Modèles TensorFlow Lite
- `ai_model_lstm_autoencoder.tflite` - Version standard
- `ai_model_lstm_autoencoder_quant.tflite` - Version quantifiée (recommandée)

### 2. Fichiers de Déploiement ESP32
- `eps_guardian_ai_model.h` - Header C++ avec modèle intégré
- `eps_guardian_inference.ino` - Sketch Arduino complet

### 3. Compatibilité MCU
- Modèle également disponible pour MCU simple: `data/ai_models/model_simple/`

## Seuils d'Anomalie
- **NORMAL**: < 0.609123
- **WARNING**: < 0.837789  
- **CRITICAL**: >= 1.066455

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
