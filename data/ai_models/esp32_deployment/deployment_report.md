# Rapport de D�ploiement IA - EPS Guardian

## Date de G�n�ration
2025-10-23 17:08:49

## Mod�le Source
- **Architecture**: LSTM Autoencoder
- **Shape d'entr�e**: (30, 7)
- **Shape de sortie**: (30, 7)
- **Param�tres**: 65,511
- **Taille originale**: 827.6 KB

## Fichiers G�n�r�s

### 1. Mod�les TensorFlow Lite
- `ai_model_lstm_autoencoder.tflite` - Version standard
- `ai_model_lstm_autoencoder_quant.tflite` - Version quantifi�e (recommand�e)

### 2. Fichiers de D�ploiement ESP32
- `eps_guardian_ai_model.h` - Header C++ avec mod�le int�gr�
- `eps_guardian_inference.ino` - Sketch Arduino complet

### 3. Compatibilit� MCU
- Mod�le �galement disponible pour MCU simple: `data/ai_models/model_simple/`

## Seuils d'Anomalie
- **NORMAL**: < 0.609123
- **WARNING**: < 0.837789  
- **CRITICAL**: >= 1.066455

## Configuration Mat�rielle Recommand�e
- **MCU**: ESP32 (avec 4MB Flash)
- **RAM minimale**: 40KB pour Tensor Arena
- **Broches LEDs**: 2 (NORMAL), 4 (WARNING), 5 (CRITICAL)

## Utilisation
1. Inclure `eps_guardian_ai_model.h` dans votre projet
2. Utiliser la classe `EPSGuardianAI` pour l'inf�rence
3. Appeler `add_sensor_data()` et `detect_anomaly()` cycliquement

## Performances Attendues
- **Temps d'inf�rence**: < 100ms sur ESP32
- **Pr�cision**: D�tection d'anomalies temporelles complexes
- **Consommation**: Optimis�e pour syst�mes embarqu�s

---
*G�n�r� automatiquement par le syst�me EPS Guardian*
