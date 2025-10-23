#!/usr/bin/env python3
"""
INFÉRENCE IA COMPLEXE OBC
Détection d'anomalies avec le modèle LSTM Autoencoder entraîné
"""

import os
import numpy as np
import json
import logging
import joblib

# Configuration des chemins
current_dir = os.path.dirname(os.path.abspath(__file__))
obc_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(obc_dir)
project_root = os.path.dirname(src_dir)

# ========== CHEMINS CORRIGÉS ==========
# Tous les outputs dans obc/
OBC_AI_DIR = os.path.join(project_root, "data", "ai_models", "model_complex")
OBC_LOGS_DIR = os.path.join(project_root, "data", "obc", "logs")
OBC_OUTPUTS_DIR = os.path.join(project_root, "data", "obc", "outputs")
DATA_DIR = os.path.join(project_root, "data", "ai_training_base")  # Corrigé

# Création des dossiers OBC
os.makedirs(OBC_LOGS_DIR, exist_ok=True)
os.makedirs(OBC_OUTPUTS_DIR, exist_ok=True)

# Logging - maintenant dans obc/logs/
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OBC_LOGS_DIR, "obc_ai_inference.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("OBC_AI_Inference")

class OBC_AI:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.thresholds = None
        self.is_loaded = False
        self.simulation_mode = True
        self.model_available = False
        
        logger.info("Initialisation de l'IA OBC")
        logger.info(f"Logs: {OBC_LOGS_DIR}")
        logger.info(f"Outputs: {OBC_OUTPUTS_DIR}")
        self.load_model_and_thresholds()

    def load_model_and_thresholds(self):
        """Charge le modèle et les seuils"""
        try:
            # Chargement des seuils
            thresholds_path = os.path.join(OBC_AI_DIR, "ai_thresholds.json")
            logger.info(f"Recherche des seuils: {thresholds_path}")
            
            if not os.path.exists(thresholds_path):
                logger.error(f"Seuils non trouves: {thresholds_path}")
                # Créer des seuils par défaut
                self.thresholds = {
                    "normal_threshold": 0.001,
                    "warning_threshold": 0.002,
                    "critical_threshold": 0.003,
                    "training_stats": {
                        "mean_error": 0.0005,
                        "std_error": 0.0005,
                        "min_error": 0.0001,
                        "max_error": 0.005
                    }
                }
                logger.info("Seuils par defaut crees")
            else:
                with open(thresholds_path, 'r') as f:
                    thresholds_data = json.load(f)
                self.thresholds = thresholds_data["anomaly_thresholds"]
                logger.info("Seuils d'anomalie charges")
            
            # Chargement du modèle avec compile=False
            model_path = os.path.join(OBC_AI_DIR, "ai_model_lstm_autoencoder.h5")
            
            if os.path.exists(model_path):
                try:
                    from tensorflow.keras.models import load_model
                    
                    print(f"Tentative de chargement du modele: {model_path}")
                    print(f"Taille: {os.path.getsize(model_path)} bytes")
                    
                    # SOLUTION 1: Charger sans compilation
                    try:
                        self.model = load_model(model_path, compile=False)
                        self.simulation_mode = False
                        self.model_available = True
                        print("SUCCES: Modele LSTM charge (compile=False)")
                        logger.info("Modele LSTM Autoencoder charge avec succes (compile=False)")
                        
                    except Exception as e:
                        print(f"Erreur avec compile=False: {e}")
                        self._fallback_to_simulation()
                            
                except ImportError as e:
                    print(f"TensorFlow non disponible: {e}")
                    self._fallback_to_simulation()
                except Exception as e:
                    print(f"Erreur inattendue: {e}")
                    self._fallback_to_simulation()
            else:
                print("Modele non trouve sur le disque")
                self._fallback_to_simulation()
            
            # Chargement ou création du scaler - CHEMIN CORRIGÉ
            scaler_path = os.path.join(DATA_DIR, "ai_sequence_scaler.pkl")
            if os.path.exists(scaler_path):
                try:
                    self.scaler = joblib.load(scaler_path)
                    logger.info("Scaler charge")
                except Exception as e:
                    logger.warning(f"Erreur chargement scaler: {e} - creation d'un scaler par defaut")
                    self._create_default_scaler()
            else:
                logger.warning("Scaler non trouve - creation d'un scaler par defaut")
                self._create_default_scaler()
            
            self.is_loaded = True
            logger.info("IA OBC initialisee avec succes")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'IA: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _fallback_to_simulation(self):
        """Basculer en mode simulation"""
        self.model = None
        self.simulation_mode = True
        self.model_available = False
        print("UTILISATION: Mode simulation active")
        logger.warning("Mode simulation active - modele TensorFlow non disponible")

    def _create_default_scaler(self):
        """Crée un scaler par défaut"""
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        # Fit avec des données simulées réalistes
        dummy_data = np.array([
            [7.4, 1.2, 35.0, 7.8, 0.8, 15.2, 1.5]  # Valeurs typiques
        ] * 100) + np.random.randn(100, 7) * 0.1
        self.scaler.fit(dummy_data)

    def analyze_sequence(self, sequence_data):
        """
        Analyse une séquence temporelle et retourne le niveau d'anomalie
        """
        if not self.is_loaded:
            return self._simulate_analysis(sequence_data)
        
        # CORRECTION: Utiliser le vrai modèle si disponible
        if self.model is not None:
            try:
                # Vérification et préparation des données
                if sequence_data.shape != (30, 7):
                    logger.warning(f"Shape incorrecte: {sequence_data.shape}, attendu: (30, 7)")
                    return self._simulate_analysis(sequence_data)
                
                # Nettoyage des données
                sequence_data = np.nan_to_num(sequence_data, nan=0.0, posinf=0.0, neginf=0.0)
                
                # CORRECTION: Normalisation - méthode corrigée
                print(f"Shape avant normalisation: {sequence_data.shape}")
                
                # Méthode corrigée: Reshape correct pour le scaler
                sequence_normalized = self.scaler.transform(sequence_data.reshape(-1, 7))
                sequence_normalized = sequence_normalized.reshape(1, 30, 7)
                print(f"Shape après normalisation: {sequence_normalized.shape}")
                
                # Prédiction avec le modèle LSTM
                print("UTILISATION DU VRAI MODELE LSTM...")
                reconstructed = self.model.predict(sequence_normalized, verbose=0)
                
                # Calcul de l'erreur de reconstruction
                reconstruction_error = np.mean(np.square(sequence_normalized - reconstructed))
                
                print(f"Erreur reconstruction LSTM: {reconstruction_error:.6f}")
                print(f"Seuil normal: {self.thresholds['normal_threshold']:.6f}")
                print(f"Seuil warning: {self.thresholds['warning_threshold']:.6f}")
                print(f"Seuil critical: {self.thresholds['critical_threshold']:.6f}")
                
                # Détermination du niveau d'anomalie
                ai_level, confidence = self._classify_anomaly(reconstruction_error)
                
                result = {
                    "ai_score": float(reconstruction_error),
                    "ai_level": ai_level,
                    "confidence": confidence,
                    "reconstruction_error": float(reconstruction_error),
                    "thresholds_used": {
                        "normal": self.thresholds["normal_threshold"],
                        "warning": self.thresholds["warning_threshold"],
                        "critical": self.thresholds["critical_threshold"]
                    },
                    "simulated": False,
                    "model_used": "LSTM Autoencoder"
                }
                
                logger.info(f"Analyse IA LSTM - Score: {reconstruction_error:.6f}, Niveau: {ai_level}")
                
                # SAUVEGARDE DU RÉSULTAT DANS OBC/OUTPUTS
                self._save_analysis_result(result, sequence_data)
                
                return result
                
            except Exception as e:
                logger.error(f"Erreur analyse LSTM: {e} - bascule vers simulation")
                print(f"ERREUR LSTM: {e}")
                import traceback
                traceback.print_exc()
                return self._simulate_analysis(sequence_data)
        else:
            # Mode simulation
            print("UTILISATION MODE SIMULATION")
            return self._simulate_analysis(sequence_data)

    def _save_analysis_result(self, result, sequence_data):
        """Sauvegarde le résultat d'analyse dans obc/outputs/"""
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Fichier JSON de résultat
            result_filename = f"obc_ai_analysis_{timestamp}.json"
            result_path = os.path.join(OBC_OUTPUTS_DIR, result_filename)
            
            result_data = {
                "timestamp": datetime.now().isoformat(),
                "analysis_result": result,
                "sequence_info": {
                    "shape": sequence_data.shape,
                    "data_points": len(sequence_data)
                }
            }
            
            with open(result_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            logger.info(f"Resultat sauvegarde: {result_path}")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde resultat: {e}")

    def _simulate_analysis(self, sequence_data):
        """Simule l'analyse IA quand le modèle n'est pas disponible"""
        logger.info("Utilisation de l'analyse simulee")
        
        try:
            # Simulation basée sur la température moyenne et autres métriques
            avg_temp = np.mean(sequence_data[:, 2])  # T_batt
            avg_current = np.mean(sequence_data[:, 1])  # I_batt
            avg_voltage = np.mean(sequence_data[:, 0])  # V_batt
            
            # Logique de détection d'anomalie simulée
            if avg_temp > 60 or avg_current > 3.0:
                result = {
                    "ai_score": 0.95,
                    "ai_level": "CRITICAL",
                    "confidence": "HIGH",
                    "reconstruction_error": 0.95,
                    "simulated": True,
                    "model_used": "Simulation Rules",
                    "metrics": {
                        "avg_temp": float(avg_temp),
                        "avg_current": float(avg_current),
                        "avg_voltage": float(avg_voltage)
                    }
                }
            elif avg_temp > 45 or avg_current > 2.0 or avg_voltage < 3.2:
                result = {
                    "ai_score": 0.75,
                    "ai_level": "WARNING",
                    "confidence": "MEDIUM", 
                    "reconstruction_error": 0.75,
                    "simulated": True,
                    "model_used": "Simulation Rules",
                    "metrics": {
                        "avg_temp": float(avg_temp),
                        "avg_current": float(avg_current),
                        "avg_voltage": float(avg_voltage)
                    }
                }
            else:
                result = {
                    "ai_score": 0.1,
                    "ai_level": "NORMAL",
                    "confidence": "HIGH",
                    "reconstruction_error": 0.1,
                    "simulated": True,
                    "model_used": "Simulation Rules",
                    "metrics": {
                        "avg_temp": float(avg_temp),
                        "avg_current": float(avg_current),
                        "avg_voltage": float(avg_voltage)
                    }
                }
            
            # Sauvegarde même en mode simulation
            self._save_analysis_result(result, sequence_data)
            return result
            
        except Exception as e:
            logger.error(f"Erreur dans l'analyse simulee: {e}")
            # Fallback si erreur
            result = {
                "ai_score": 0.1,
                "ai_level": "NORMAL",
                "confidence": "MEDIUM",
                "reconstruction_error": 0.1,
                "simulated": True,
                "model_used": "Simulation Rules",
                "error": str(e)
            }
            self._save_analysis_result(result, sequence_data)
            return result

    def _classify_anomaly(self, error):
        """Classifie l'anomalie basée sur les seuils"""
        if error <= self.thresholds["normal_threshold"]:
            return "NORMAL", "HIGH"
        elif error <= self.thresholds["warning_threshold"]:
            return "WARNING", "MEDIUM"
        else:
            return "CRITICAL", "HIGH"

    def get_model_info(self):
        """Retourne les informations du modèle"""
        if not self.is_loaded:
            return {"status": "NOT_LOADED"}
        
        if self.model is not None:
            status = "LOADED"
            model_arch = "LSTM Autoencoder"
        else:
            status = "SIMULATION"
            model_arch = "Simulation Rules"
        
        return {
            "status": status,
            "model_architecture": model_arch,
            "input_shape": [30, 7],
            "thresholds": self.thresholds,
            "training_stats": self.thresholds.get("training_stats", {}),
            "scaler_available": self.scaler is not None,
            "model_available": self.model is not None,
            "simulation_mode": self.model is None
        }

# Instance globale pour usage facile
obc_ai = OBC_AI()

def analyze_sensor_sequence(sequence_data):
    """
    Fonction utilitaire pour analyser une séquence de capteurs
    """
    return obc_ai.analyze_sequence(sequence_data)

if __name__ == "__main__":
    # Test de l'IA
    ai = OBC_AI()
    print(f"IA chargee: {ai.is_loaded}")
    print(f"Modele disponible: {ai.model is not None}")
    print(f"Mode simulation: {ai.model is None}")
    print(f"Statut complet: {json.dumps(ai.get_model_info(), indent=2)}")
    
    # Test avec des données réalistes
    print("\nTest avec donnees simulees:")
    test_sequence = np.array([
        [7.4, 1.2, 35.0, 7.8, 0.8, 15.2, 1.5]  # Valeurs de base
    ] * 30) + np.random.randn(30, 7) * 0.1
    
    result = ai.analyze_sequence(test_sequence)
    print("Resultat du test:")
    print(json.dumps(result, indent=2))
