#!/usr/bin/env python3
"""
Système d'inférence IA pour EPS GUARDIAN - VERSION CORRIGÉE
"""

import tensorflow as tf
import numpy as np
import os
import json
import joblib
import pandas as pd
from datetime import datetime

class AIModelInference:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.model_dir = os.path.join(self.base_dir, "data", "ai_models", "model_simple")
        self.training_dir = os.path.join(self.base_dir, "data", "training_data")
        self.output_dir = os.path.join(self.base_dir, "data", "mcu", "outputs", "ai_inference_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.scaler = None
        self.feature_names = None
        self.thresholds = None
        self.inference_results = []
        
        print("AIModelInference initialisé")
    
    def load_artifacts(self):
        """Charge tous les artefacts IA"""
        try:
            # Charger le scaler
            scaler_path = os.path.join(self.training_dir, "ai_scaler.pkl")
            self.scaler = joblib.load(scaler_path)
            
            # Charger les noms des features
            features_path = os.path.join(self.training_dir, "ai_feature_names.npy")
            self.feature_names = np.load(features_path, allow_pickle=True).tolist()
            
            # Charger les seuils
            thresholds_path = os.path.join(self.model_dir, "ai_thresholds.json")
            with open(thresholds_path, 'r') as f:
                thresholds_data = json.load(f)
                self.thresholds = thresholds_data["thresholds"]
            
            # Charger le modèle TFLite
            tflite_path = os.path.join(self.model_dir, "ai_autoencoder.tflite")
            self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print("Tous les artefacts chargés avec succès")
            return True
            
        except Exception as e:
            print(f"Erreur chargement artefacts: {e}")
            return False
    
    def predict_from_normalized(self, normalized_features):
        """Prédiction directe avec features déjà normalisées"""
        try:
            # Préparer l'input
            input_data = normalized_features.astype(np.float32)
            
            # Exécuter l'inférence
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Calculer l'erreur de reconstruction
            reconstruction_error = np.mean(np.square(input_data - output_data))
            
            # Déterminer le niveau d'anomalie
            anomaly_level = "NORMAL"
            if reconstruction_error >= self.thresholds["critical"]:
                anomaly_level = "CRITICAL"
            elif reconstruction_error >= self.thresholds["warning"]:
                anomaly_level = "WARNING"
            
            return reconstruction_error, anomaly_level
            
        except Exception as e:
            print(f"Erreur prédiction: {e}")
            return None, None
    
    def test_real_scenarios(self):
        """Test avec des données RÉELLES du dataset d'entraînement"""
        try:
            # Charger les données d'entraînement normalisées
            train_data_path = os.path.join(self.training_dir, "ai_train_data.npy")
            X_train = np.load(train_data_path)
            
            print(f"Données d'entraînement chargées: {X_train.shape}")
            
            # Tester quelques échantillons réels
            test_indices = [0, 100, 500, 1000, 2000]  # Différents échantillons
            
            for i, idx in enumerate(test_indices):
                if idx < len(X_train):
                    sample = X_train[idx:idx+1]  # Prendre un échantillon
                    
                    # Faire la prédiction
                    error, level = self.predict_from_normalized(sample)
                    
                    if error is not None:
                        print(f"\nTest {i+1} (échantillon {idx}):")
                        print(f"  Erreur reconstruction: {error:.6f}")
                        print(f"  Niveau anomalie: {level}")
                        print(f"  Seuils: Normal<{self.thresholds['normal']:.6f}")
                        
                        # Sauvegarder le résultat
                        result = {
                            "timestamp": datetime.now().isoformat(),
                            "sample_index": idx,
                            "reconstruction_error": float(error),
                            "anomaly_level": level,
                            "is_training_sample": True
                        }
                        self.inference_results.append(result)
            
            return True
            
        except Exception as e:
            print(f"Erreur test scénarios réels: {e}")
            return False
    
    def test_synthetic_normal(self):
        """Test avec des données synthétiques NORMALES"""
        try:
            # Créer des features normalisées typiques (proches de la moyenne)
            normal_features = np.random.normal(0.5, 0.1, (1, len(self.feature_names)))
            normal_features = np.clip(normal_features, 0, 1)  # Garder entre 0 et 1
            
            error, level = self.predict_from_normalized(normal_features)
            
            if error is not None:
                print(f"\nTest SYNTHETIQUE NORMAL:")
                print(f"  Erreur reconstruction: {error:.6f}")
                print(f"  Niveau anomalie: {level}")
                
                result = {
                    "timestamp": datetime.now().isoformat(),
                    "sample_type": "synthetic_normal",
                    "reconstruction_error": float(error),
                    "anomaly_level": level,
                    "is_training_sample": False
                }
                self.inference_results.append(result)
            
            return True
            
        except Exception as e:
            print(f"Erreur test synthétique: {e}")
            return False
    
    def save_results(self):
        """Sauvegarde tous les résultats"""
        try:
            # CSV
            csv_path = os.path.join(self.output_dir, "ai_inference_summary.csv")
            df = pd.DataFrame(self.inference_results)
            df.to_csv(csv_path, index=False)
            
            # JSON
            json_path = os.path.join(self.output_dir, "ai_inference_test_cases.json")
            with open(json_path, 'w') as f:
                json.dump(self.inference_results, f, indent=2)
            
            print(f"\nRésultats sauvegardés:")
            print(f"  - CSV: {csv_path}")
            print(f"  - JSON: {json_path}")
            
        except Exception as e:
            print(f"Erreur sauvegarde: {e}")

def main():
    inference = AIModelInference()
    
    if inference.load_artifacts():
        print("\nSYSTEME D'INFERENCE IA PRET")
        print("=" * 40)
        
        # 1. Test avec données RÉELLES d'entraînement
        print("\n1. TESTS AVEC DONNEES REELLES:")
        inference.test_real_scenarios()
        
        # 2. Test avec données synthétiques
        print("\n2. TESTS AVEC DONNEES SYNTHETIQUES:")
        inference.test_synthetic_normal()
        
        # 3. Sauvegarde
        inference.save_results()
        
        print("\nTests d'inference termines avec succes!")
        
    else:
        print("Impossible de charger les artefacts IA")

if __name__ == "__main__":
    main()
