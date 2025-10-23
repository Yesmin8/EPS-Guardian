#!/usr/bin/env python3
"""
TEST RÉEL DE L'IA OBC - AVEC VOTRE MODÈLE RÉEL
Teste le vrai modèle LSTM Autoencoder du dossier model_complex/
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime, timedelta

# Correction des imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interface.obc_message_handler import OBCMessageHandler
from interface.obc_response_generator import OBCResponseGenerator

class RealAI_FusionTest:
    def __init__(self):
        print(" TEST RÉEL DE L'IA OBC - MODÈLE LSTM AUTOENCODEUR")
        print("=" * 60)
        
        # Vérification de l'IA réelle
        self.check_real_ai()
        
        self.handler = OBCMessageHandler()
        self.response_gen = OBCResponseGenerator()
        self.message_count = 0
        self.critical_detections = 0
        self.warning_detections = 0
        self.normal_detections = 0
        
        print(" Test configuré pour utiliser le VRAI modèle IA OBC")

    def check_real_ai(self):
        """Vérifie que le vrai modèle IA est disponible"""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(project_root, "data", "ai_models", "model_complex", "ai_model_lstm_autoencoder.h5")
        thresholds_path = os.path.join(project_root, "data", "ai_models", "model_complex", "ai_thresholds.json")
        
        print(f"  Recherche du modèle IA réel...")
        print(f"   Modèle: {model_path}")
        print(f"   Seuils: {thresholds_path}")
        
        if os.path.exists(model_path):
            print(f" MODÈLE RÉEL TROUVÉ: {os.path.getsize(model_path)/1024/1024:.1f} MB")
        else:
            print(f" MODÈLE NON TROUVÉ - Le test utilisera le mode simulation")
            
        if os.path.exists(thresholds_path):
            with open(thresholds_path, 'r') as f:
                thresholds = json.load(f)
            print(f" SEUILS CHARGÉS: {thresholds['anomaly_thresholds']}")
        else:
            print(" SEUILS NON TROUVÉS")

    def generate_realistic_sensor_sequence(self, anomaly_type="NONE", progression=0):
        """
        Génère une séquence réaliste de 30 points pour le LSTM
        Format requis: (30, 7) - 30 timesteps, 7 features
        """
        sequence = []
        base_time = datetime.now() - timedelta(seconds=30)
        
        for i in range(30):
            timestamp = base_time + timedelta(seconds=i)
            
            # Valeurs de base nominales
            base_values = {
                "V_batt": 7.4,
                "I_batt": 1.2, 
                "T_batt": 35.0,
                "V_bus": 7.8,
                "I_bus": 0.8,
                "V_solar": 15.2,
                "I_solar": 1.5
            }
            
            # Application d'anomalies réalistes
            if anomaly_type == "OVERHEAT":
                # Surchauffe progressive réaliste
                base_values["T_batt"] = 40.0 + (i * 1.0) + (progression * 10)
                base_values["I_batt"] *= 1.2  # Courant augmente avec température
                
            elif anomaly_type == "OVERCURRENT":
                # Surcharge courant réaliste
                base_values["I_batt"] = 2.5 + np.random.normal(0, 0.3)
                base_values["V_batt"] *= 0.95  # Tension baisse légèrement
                
            elif anomaly_type == "UNDERVOLTAGE":
                # Décharge profonde réaliste
                base_values["V_batt"] = 3.5 - (i * 0.05) - (progression * 0.5)
                base_values["I_batt"] = -1.5  # Batterie se décharge
                
            elif anomaly_type == "CONVERTER_FAULT":
                # Défaillance convertisseur réaliste
                base_values["V_bus"] = 5.0 + np.random.normal(0, 0.5)
                base_values["converter_ratio"] = 0.3  # Ratio anormal
                
            # Ajout de bruit réaliste
            noisy_data = {
                "timestamp": timestamp.isoformat() + "Z",
                "V_batt": base_values["V_batt"] + np.random.normal(0, 0.02),
                "I_batt": base_values["I_batt"] + np.random.normal(0, 0.05),
                "T_batt": base_values["T_batt"] + np.random.normal(0, 0.1),
                "V_bus": base_values["V_bus"] + np.random.normal(0, 0.01),
                "I_bus": base_values.get("I_bus", 0.8) + np.random.normal(0, 0.02),
                "V_solar": base_values["V_solar"] + np.random.normal(0, 0.1),
                "I_solar": base_values["I_solar"] + np.random.normal(0, 0.02)
            }
            
            sequence.append(noisy_data)
            
        return sequence

    def create_real_test_message(self, sequence_data, anomaly_type="NONE"):
        """Crée un message de test réaliste pour l'IA"""
        self.message_count += 1
        
        # Détermine le type de message basé sur l'anomalie
        if anomaly_type == "NONE":
            message_type = "SUMMARY"
            priority = "MEDIUM"
        else:
            message_type = "ALERT_CRITICAL" 
            priority = "HIGH"
        
        message = {
            "header": {
                "message_id": self.message_count,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source": "EPS_MCU",
                "message_type": message_type,
                "priority": priority,
                "version": "1.0"
            },
            "payload": {
                "sensor_data": sequence_data[-5:],  # Dernières mesures
                "temporal_window": {
                    "window_size_seconds": 30,
                    "data_points_count": len(sequence_data),
                    "sensor_data": sequence_data  # Séquence complète pour l'IA
                },
                "emergency_level": "HIGH" if anomaly_type != "NONE" else "MEDIUM",
                "test_anomaly_type": anomaly_type  # Pour le debug
            }
        }
        
        return message

    def run_real_ai_test(self, duration_minutes=5):
        """Exécute le test avec le vrai modèle IA"""
        print(f"\n DÉMARRAGE TEST IA RÉELLE - Durée: {duration_minutes} minutes")
        print("Cycle | Anomalie      | Score IA   | Niveau IA  | Décision OBC")
        print("-" * 70)
        
        end_time = time.time() + (duration_minutes * 60)
        cycle = 0
        
        # Scénarios de test réalistes
        test_scenarios = [
            ("NONE", "Système nominal"),
            ("OVERHEAT", "Surchauffe progressive"), 
            ("OVERCURRENT", "Surcharge courant"),
            ("UNDERVOLTAGE", "Décharge batterie"),
            ("CONVERTER_FAULT", "Défaillance convertisseur"),
            ("NONE", "Retour à la normale")
        ]
        
        while time.time() < end_time and cycle < len(test_scenarios):
            cycle += 1
            anomaly_type, description = test_scenarios[cycle - 1]
            
            print(f"\n TEST {cycle}: {description}")
            
            # Génération séquence réaliste
            sequence = self.generate_realistic_sensor_sequence(
                anomaly_type=anomaly_type, 
                progression=cycle/len(test_scenarios)
            )
            
            # Création message de test
            test_message = self.create_real_test_message(sequence, anomaly_type)
            
            # Traitement par l'OBC avec la VRAIE IA
            start_time = time.time()
            obc_response = self.handler.process_mcu_message(test_message)
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # Extraction résultats IA
            ai_analysis = obc_response.get('ai_analysis', {})
            ai_score = ai_analysis.get('ai_score', 0)
            ai_level = ai_analysis.get('ai_level', 'UNKNOWN')
            confidence = ai_analysis.get('confidence', 'LOW')
            simulated = ai_analysis.get('simulated', True)
            
            # Statistiques
            if ai_level == 'CRITICAL':
                self.critical_detections += 1
            elif ai_level == 'WARNING':
                self.warning_detections += 1
            else:
                self.normal_detections += 1
            
            # Affichage résultats détaillés
            print(f"    Score IA: {ai_score:.6f}")
            print(f"    Niveau: {ai_level} (Confiance: {confidence})")
            print(f"    Modèle: {'SIMULATION' if simulated else 'LSTM RÉEL'}")
            print(f"    Temps traitement: {processing_time:.1f}ms")
            print(f"    Décision OBC: {obc_response['decision']} -> {obc_response['action']}")
            
            # Vérification cohérence détection
            if anomaly_type != "NONE" and ai_level == "NORMAL":
                print(f"   ️  ATTENTION: Anomalie '{anomaly_type}' non détectée!")
            elif anomaly_type == "NONE" and ai_level != "NORMAL":
                print(f"   ️  ATTENTION: Faux positif! Normal détecté comme {ai_level}")
            
            time.sleep(3)  # Pause pour analyse

        self._print_real_test_summary()

    def _print_real_test_summary(self):
        """Affiche le résumé détaillé du test réel"""
        print("\n" + "=" * 70)
        print(" RAPPORT FINAL - TEST IA RÉELLE OBC")
        print("=" * 70)
        
        total_tests = self.critical_detections + self.warning_detections + self.normal_detections
        
        print(f"Tests effectués: {total_tests}")
        print(f"Détections CRITICAL: {self.critical_detections}")
        print(f"Détections WARNING: {self.warning_detections}") 
        print(f"Détections NORMAL: {self.normal_detections}")
        
        if total_tests > 0:
            print(f"Taux CRITICAL: {(self.critical_detections/total_tests)*100:.1f}%")
            print(f"Taux WARNING: {(self.warning_detections/total_tests)*100:.1f}%")
            print(f"Taux NORMAL: {(self.normal_detections/total_tests)*100:.1f}%")
        
        # Évaluation performance
        print("\n ÉVALUATION IA RÉELLE:")
        if self.critical_detections > 0:
            print(" L'IA détecte les anomalies critiques")
        else:
            print(" Aucune anomalie critique détectée - vérifiez le modèle")
            
        if self.normal_detections > 0:
            print(" L'IA reconnaît le comportement normal")
        else:
            print(" Aucun comportement normal reconnu - vérifiez les seuils")

def main():
    """Point d'entrée principal"""
    print(" SYSTÈME DE TEST IA RÉELLE - EPS GUARDIAN OBC")
    print("Ce test utilise le VRAI modèle LSTM Autoencoder")
    print("=" * 60)
    
    # Demander la durée à l'utilisateur
    try:
        duration = int(input("Durée du test (minutes, défaut: 5): ") or "5")
    except:
        duration = 5
    
    # Lancer le test
    real_tester = RealAI_FusionTest()
    real_tester.run_real_ai_test(duration)
    
    print(f"\n TEST TERMINÉ! Vérifiez les logs dans data/obc/logs/")

if __name__ == "__main__":
    main()
