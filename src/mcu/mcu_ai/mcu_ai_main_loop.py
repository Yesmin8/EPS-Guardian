#!/usr/bin/env python3
"""
MCU AI MAIN LOOP – Version finale avec buffer temporel 30s
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque

# === CORRECTION DES IMPORTS ===
# Remonter d'un niveau pour atteindre le dossier mcu/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_mcu_dir = os.path.dirname(current_dir)  # D:\Challenge AESS&IES\src\mcu\
sys.path.insert(0, parent_mcu_dir)

# Maintenant importer depuis le dossier mcu/
from mcu_logger import MCULogger
from mcu_rule_engine import MCU_RuleEngine, obc_interface
from mcu_data_interface import DataInterface

class MCUAI_MainLoop:
    def __init__(self):
        self.logger = MCULogger("mcu_ai_validation.log")
        self.logger.info("DÉMARRAGE SYSTÈME HYBRIDE AVEC BUFFER TEMPOREL 30s")

        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        data_path = os.path.join(self.base_dir, "data", "dataset", "pro_eps_dataset.csv")

        self.data_interface = DataInterface(source="csv", path=data_path)
        self.rule_engine = RuleEngine()
        self.obc_interface = obc_interface

        self.model_dir = os.path.join(self.base_dir, "data", "ai_models", "model_simple")
        self.training_data_dir = os.path.join(self.base_dir, "data", "training_data")
        self.output_dir = os.path.join(self.base_dir, "data", "mcu", "outputs", "hybrid_simulation")
        os.makedirs(self.output_dir, exist_ok=True)

        self.load_ai_model()
        self.results = []
        self.cycle_count = 0
        self.ai_alert_buffer = deque(maxlen=5)

        self.logger.info("Système hybride initialisé avec buffer temporel 30s")
    def load_ai_model(self):
        """Charge le modèle IA avec seuils calibrés"""
        try:
            import joblib
            # Correction des chemins - utilisation de training_data/
            self.scaler = joblib.load(os.path.join(self.training_data_dir, "ai_scaler.pkl"))
            self.feature_names = np.load(os.path.join(self.training_data_dir, "ai_feature_names.npy"), allow_pickle=True).tolist()
            
            # Charger les données d'entraînement pour la simulation
            self.train_data = np.load(os.path.join(self.training_data_dir, "ai_train_data.npy"))

            self.thresholds = {
                "normal": 0.35,
                "warning": 0.55, 
                "critical": 0.75
            }

            self.logger.info(f"IA chargée ({len(self.feature_names)} features, seuils calibrés)")
        except Exception as e:
            self.logger.error(f"ERREUR chargement IA: {e}")
            raise

    def prepare_features(self, sensor_data):
        """Prépare les features pour l'IA"""
        features = {}
        base_features = ["V_batt", "I_batt", "T_batt", "V_bus", "I_bus", "V_solar", "I_solar"]

        for f in base_features:
            features[f] = sensor_data.get(f, 0.0)

        try:
            features["P_batt"] = features["V_batt"] * features["I_batt"]
            v_solar = features["V_solar"]
            features["converter_ratio"] = features["V_bus"] / v_solar if v_solar > 0.1 else 0.0
        except Exception:
            features["P_batt"], features["converter_ratio"] = 0.0, 0.0

        features["delta_V_batt"] = 0.0
        features["delta_T_batt"] = 0.0
        return features

    def normalize_features(self, features_dict):
        """Normalise les features"""
        try:
            df = pd.DataFrame([features_dict], columns=self.feature_names)
            normalized = self.scaler.transform(df)
            return normalized.reshape(-1)
        except Exception as e:
            self.logger.error(f"Erreur normalisation: {e}")
            return None

    def simulate_ai_inference(self, normalized_features):
        """Simule l'inférence IA"""
        try:
            distances = np.sqrt(np.sum((self.train_data - normalized_features) ** 2, axis=1))
            min_d, avg_d = np.min(distances), np.mean(distances)
            simulated_error = min_d * 0.25 + avg_d * 0.20
            return float(np.clip(simulated_error, 0.001, 1.0))
        except Exception as e:
            self.logger.error(f"Erreur simulation IA: {e}")
            return 0.5

    def ai_anomaly_detection(self, sensor_data):
        """Détection d'anomalie IA"""
        features = self.prepare_features(sensor_data)
        normalized = self.normalize_features(features)
        if normalized is None:
            return {"ai_error": -1, "ai_level": "ERROR"}

        ai_error = self.simulate_ai_inference(normalized)

        if ai_error < self.thresholds["normal"]:
            ai_level = "NORMAL"
        elif ai_error < self.thresholds["warning"]:
            ai_level = "WARNING"
        else:
            ai_level = "CRITICAL"

        self.logger.info(f"[IA] Erreur={ai_error:.4f} → {ai_level}")
        return {"ai_error": ai_error, "ai_level": ai_level, "features_used": len(self.feature_names)}

    def hybrid_decision_fusion(self, ai_result, rule_result):
        """Fusion IA + règles"""
        ai_level = ai_result["ai_level"]
        rule_level = rule_result["max_level"]

        if rule_level == ai_level:
            return ai_level, "AGREEMENT", "HIGH"
        if rule_level == "CRITICAL":
            return "CRITICAL", "RULE_DOMINANT", "HIGH"
        if ai_level == "CRITICAL" and rule_level == "NORMAL":
            return "CRITICAL", "IA_DOMINANT", "HIGH"
        if ai_level == "WARNING" and rule_level == "NORMAL":
            return "WARNING", "IA_DOMINANT", "MEDIUM"
        if rule_level == "WARNING" and ai_level == "NORMAL":
            return "WARNING", "RULE_DOMINANT", "MEDIUM"
        if rule_level == "WARNING" and ai_level == "CRITICAL":
            return "CRITICAL", "FUSION", "HIGH"
        return rule_level, "RULE_DEFAULT", "LOW"

    def hybrid_decision_pipeline(self, sensor_data):
        """Pipeline de décision hybride avec transmission OBC"""
        self.cycle_count += 1
        
        # Ajouter données au buffer temporel (via rule_engine)
        self.obc_interface.add_sensor_data(sensor_data)
        
        # 1. Évaluation règles MCU
        actions, message_type, rule_details = self.rule_engine.apply_rules(sensor_data)
        rule_result = self._convert_rule_result(actions, message_type, rule_details)
        
        # 2. Détection IA
        ai_result = self.ai_anomaly_detection(sensor_data)
        
        # 3. Fusion
        final_decision, decision_source, confidence = self.hybrid_decision_fusion(ai_result, rule_result)
        
        # 4. Transmission OBC pour décisions IA (avec données de fenêtre)
        if final_decision == "CRITICAL":
            self.obc_interface.send_to_obc("ALERT_CRITICAL", {
                "detection_type": "AI_ANOMALY",
                "final_decision": final_decision,
                "decision_source": decision_source,
                "ai_confidence": ai_result.get("ai_error", 0),
                "ai_level": ai_result.get("ai_level", "UNKNOWN"),
                "rule_triggered": rule_result["details"].get("rule_triggered"),
                "emergency_level": "HIGH"
            }, include_window_data=True)
            
        elif final_decision == "WARNING":
            self.obc_interface.send_to_obc("SUMMARY", {
                "detection_type": "AI_ANOMALY",
                "final_decision": final_decision, 
                "decision_source": decision_source,
                "ai_confidence": ai_result.get("ai_error", 0),
                "ai_level": ai_result.get("ai_level", "UNKNOWN"),
                "rule_triggered": rule_result["details"].get("rule_triggered"),
                "emergency_level": "MEDIUM"
            }, include_window_data=True)
        
        # 5. Log décision finale
        if final_decision != "NORMAL":
            status_text = "[CRITIQUE]" if final_decision == "CRITICAL" else "[WARNING]"
            self.logger.info(f"{status_text} Decision: {final_decision} (Source: {decision_source})")

        return {
            "cycle": self.cycle_count,
            "timestamp": sensor_data.get("timestamp", datetime.now().isoformat()),
            "final_decision": final_decision,
            "decision_source": decision_source,
            "confidence": confidence,
            "rule_result": rule_result,
            "ai_result": ai_result
        }

    def _convert_rule_result(self, actions, message_type, rule_details):
        """Convertit le résultat des règles"""
        if message_type == "ALERT_CRITICAL":
            return {"max_level": "CRITICAL", "actions": actions, "details": rule_details}
        elif message_type == "SUMMARY":
            return {"max_level": "WARNING", "actions": actions, "details": rule_details}
        else:
            return {"max_level": "NORMAL", "actions": actions, "details": rule_details}

    def run_hybrid_simulation(self, max_cycles=50):
        """Exécute une simulation complète"""
        self.logger.info(f"Démarrage simulation hybride ({max_cycles} cycles)")

        stats = {
            "NORMAL": 0, "WARNING": 0, "CRITICAL": 0,
            "agreement": 0, "ai_dominant": 0, "rules_dominant": 0, "fusion": 0
        }

        for _ in range(max_cycles):
            data = self.data_interface.get_next()
            if data is None:
                break
            result = self.hybrid_decision_pipeline(data)
            self.results.append(result)
            stats[result["final_decision"]] += 1

        self.save_results(stats)
        return stats

    def save_results(self, stats):
        """Sauvegarde les résultats"""
        # Données détaillées
        df = pd.DataFrame([{
            "cycle": r["cycle"],
            "timestamp": r["timestamp"],
            "final_decision": r["final_decision"],
            "decision_source": r["decision_source"],
            "confidence": r["confidence"],
            "ai_level": r["ai_result"]["ai_level"],
            "ai_error": r["ai_result"]["ai_error"],
            "rule_level": r["rule_result"]["max_level"]
        } for r in self.results])

        csv_path = os.path.join(self.output_dir, "mcu_ai_detailed_results.csv")
        df.to_csv(csv_path, index=False)

        # Résumé
        total = len(df)
        summary_data = [
            {"Type": "NORMAL", "Nombre": stats["NORMAL"], "Pourcentage": f"{(stats['NORMAL']/total)*100:.1f}%"},
            {"Type": "WARNING", "Nombre": stats["WARNING"], "Pourcentage": f"{(stats['WARNING']/total)*100:.1f}%"},
            {"Type": "CRITICAL", "Nombre": stats["CRITICAL"], "Pourcentage": f"{(stats['CRITICAL']/total)*100:.1f}%"},
        ]
        summary = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.output_dir, "simulation_summary.csv")
        summary.to_csv(summary_path, index=False)

        # Sauvegarder les statistiques détaillées
        stats_path = os.path.join(self.output_dir, "hybrid_simulation_stats.json")
        with open(stats_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_cycles": total,
                "decisions": stats,
                "ai_thresholds": self.thresholds
            }, f, indent=2)

        self.logger.info("=== SIMULATION TERMINÉE ===")
        for _, row in summary.iterrows():
            self.logger.info(f"{row['Type']}: {row['Nombre']} cycles ({row['Pourcentage']})")

        print(f"\nRésultats sauvegardés dans: {self.output_dir}")

def main():
    try:
        system = MCUAI_MainLoop()
        system.run_hybrid_simulation(max_cycles=50)
    except Exception as e:
        print(f"Erreur critique: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
