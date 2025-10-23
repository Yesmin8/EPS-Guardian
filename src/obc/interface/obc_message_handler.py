#!/usr/bin/env python3
"""
GESTIONNAIRE DE MESSAGES OBC
Reçoit et traite les messages du MCU
"""

import json
import logging
import sys
import os
import numpy as np
from datetime import datetime

# Configuration des chemins
current_dir = os.path.dirname(os.path.abspath(__file__))
obc_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(obc_dir)
project_root = os.path.dirname(src_dir)

sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, obc_dir)

try:
    from ai.ai_complex_inference import obc_ai
    AI_AVAILABLE = True
    print("INFO: IA OBC chargee avec succes")
except ImportError as e:
    print(f"ATTENTION: IA non disponible - {e}")
    AI_AVAILABLE = False

class OBCMessageHandler:
    def __init__(self):
        self.logger = logging.getLogger("OBC_MessageHandler")
        self.logger.info("Initialisation du gestionnaire de messages OBC")
        self.ai_available = AI_AVAILABLE

    def process_mcu_message(self, message_json):
        """
        Traite un message JSON reçu du MCU
        """
        try:
            # Conversion si nécessaire
            if isinstance(message_json, str):
                message = json.loads(message_json)
            else:
                message = message_json
            
            self.logger.info(f"Message recu: {message.get('message_type', 'UNKNOWN')}")
            
            # Extraction des données de fenêtre temporelle
            temporal_window = self._extract_temporal_data(message)
            
            if temporal_window is None:
                return self._generate_error_response("Donnees temporelles manquantes")
            
            # Analyse IA complexe (si disponible)
            if self.ai_available:
                ai_result = obc_ai.analyze_sequence(temporal_window)
            else:
                ai_result = self._simulate_ai_analysis(temporal_window)
            
            # Génération de la réponse
            response = self._generate_obc_response(message, ai_result)
            
            self.logger.info(f"Reponse generee: {response.get('decision', 'UNKNOWN')}")
            return response
            
        except Exception as e:
            self.logger.error(f"Erreur traitement message: {e}")
            return self._generate_error_response(str(e))

    def _extract_temporal_data(self, message):
        """Extrait les données de la fenêtre temporelle"""
        try:
            payload = message.get('payload', {})
            
            # Recherche des données temporelles
            temporal_data = None
            
            # Cas 1: Données dans temporal_window
            if 'temporal_window' in payload:
                window_data = payload['temporal_window'].get('sensor_data', [])
                if window_data and len(window_data) >= 30:
                    temporal_data = np.array([[
                        point.get('V_batt', 0), point.get('I_batt', 0),
                        point.get('T_batt', 0), point.get('V_bus', 0),
                        point.get('I_bus', 0), point.get('V_solar', 0),
                        point.get('I_solar', 0)
                    ] for point in window_data])
                    return temporal_data
            
            # Cas 2: Données directes dans sensor_data
            elif 'sensor_data' in payload:
                sensor_data = payload['sensor_data']
                if isinstance(sensor_data, list) and len(sensor_data) >= 30:
                    temporal_data = np.array([[
                        point.get('V_batt', 0), point.get('I_batt', 0),
                        point.get('T_batt', 0), point.get('V_bus', 0),
                        point.get('I_bus', 0), point.get('V_solar', 0),
                        point.get('I_solar', 0)
                    ] for point in sensor_data[-30:]])  # Derniers 30 points
                    return temporal_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erreur extraction donnees: {e}")
            return None

    def _simulate_ai_analysis(self, temporal_window):
        """Simule l'analyse IA quand le modèle n'est pas disponible"""
        # Simulation simple basée sur la température
        avg_temp = np.mean(temporal_window[:, 2])  # T_batt
        
        if avg_temp > 60:
            return {
                "ai_score": 0.95,
                "ai_level": "CRITICAL",
                "confidence": "HIGH",
                "reconstruction_error": 0.95,
                "simulated": True
            }
        elif avg_temp > 45:
            return {
                "ai_score": 0.75,
                "ai_level": "WARNING", 
                "confidence": "MEDIUM",
                "reconstruction_error": 0.75,
                "simulated": True
            }
        else:
            return {
                "ai_score": 0.1,
                "ai_level": "NORMAL",
                "confidence": "HIGH",
                "reconstruction_error": 0.1,
                "simulated": True
            }

    def _generate_obc_response(self, original_message, ai_result):
        """Génère la réponse de l'OBC"""
        message_type = original_message.get('message_type', 'UNKNOWN')
        ai_level = ai_result.get('ai_level', 'ERROR')
        
        response = {
            "timestamp": datetime.now().isoformat() + "Z",
            "original_message_id": original_message.get('header', {}).get('message_id'),
            "original_type": message_type,
            "ai_analysis": ai_result,
            "decision": "PENDING",
            "action": "NONE",
            "confidence": ai_result.get('confidence', 'LOW'),
            "notes": ""
        }
        
        # Logique de décision basée sur l'analyse IA
        if ai_level == "CRITICAL":
            response.update({
                "decision": "CONFIRM",
                "action": "ISOLATE_BATTERY",
                "notes": "Anomalie critique confirmee par IA OBC"
            })
        elif ai_level == "WARNING":
            response.update({
                "decision": "MONITOR", 
                "action": "INCREASE_SAMPLING",
                "notes": "Anomalie warning detectee - surveillance renforcee"
            })
        elif ai_level == "NORMAL":
            response.update({
                "decision": "IGNORE",
                "action": "NONE",
                "notes": "Aucune anomalie detectee par IA OBC"
            })
        else:  # ERROR
            response.update({
                "decision": "REQUEST_RETRANSMISSION",
                "action": "NONE",
                "notes": "Erreur analyse IA - donnees invalides"
            })
        
        return response

    def _generate_error_response(self, error_msg):
        """Génère une réponse d'erreur"""
        return {
            "timestamp": datetime.now().isoformat() + "Z",
            "decision": "ERROR",
            "action": "NONE",
            "error": error_msg,
            "notes": "Erreur lors du traitement du message"
        }

# Instance globale
message_handler = OBCMessageHandler()

def process_incoming_message(message_json):
    """
    Fonction utilitaire pour traiter un message MCU
    """
    return message_handler.process_mcu_message(message_json)