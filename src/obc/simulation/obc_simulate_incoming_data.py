#!/usr/bin/env python3
"""
SIMULATION DE DONNEES ENTRANTES OBC
Teste le système OBC avec des messages MCU simulés
"""

import os
import sys
import json
import numpy as np
import logging
from datetime import datetime, timedelta

# Configuration des chemins
current_dir = os.path.dirname(os.path.abspath(__file__))
obc_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(obc_dir)
project_root = os.path.dirname(src_dir)

# ========== CHEMINS CORRIGÉS ==========
# Tous les logs dans obc/
OBC_LOGS_DIR = os.path.join(project_root, "data", "obc", "logs")
os.makedirs(OBC_LOGS_DIR, exist_ok=True)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OBC_LOGS_DIR, "obc_simulation_test.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("OBC_Simulation_Test")

sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, obc_dir)

from interface.obc_message_handler import OBCMessageHandler
from interface.obc_response_generator import OBCResponseGenerator

def create_sample_mcu_message(message_type="SUMMARY", include_window=True):
    """Crée un message MCU simulé"""
    
    # Données capteurs simulées - CORRIGÉ: 30 points exactement
    sensor_data = []
    base_time = datetime.now() - timedelta(seconds=30)
    
    for i in range(30):  # Exactement 30 points
        timestamp = base_time + timedelta(seconds=i)
        sensor_data.append({
            "timestamp": timestamp.isoformat() + "Z",
            "V_batt": 7.4 + np.random.normal(0, 0.1),
            "I_batt": 1.2 + np.random.normal(0, 0.2),
            "T_batt": 35.0 + np.random.normal(0, 0.5),
            "V_bus": 7.8 + np.random.normal(0, 0.05),
            "I_bus": 0.8 + np.random.normal(0, 0.1),
            "V_solar": 15.2 + np.random.normal(0, 0.3),
            "I_solar": 1.5 + np.random.normal(0, 0.1)
        })
    
    message = {
        "header": {
            "message_id": np.random.randint(1000, 9999),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source": "EPS_MCU",
            "message_type": message_type,
            "priority": "HIGH" if message_type == "ALERT_CRITICAL" else "MEDIUM",
            "version": "1.0"
        },
        "payload": {
            "rule_triggered": "R4" if message_type == "SUMMARY" else "R1",
            "sensor_data": sensor_data[-10:],  # Derniers 10 points
            "actions_taken": ["REDUCE_LOAD", "LED_YELLOW"],
            "emergency_level": "HIGH" if message_type == "ALERT_CRITICAL" else "MEDIUM"
        }
    }
    
    if include_window:
        message["payload"]["temporal_window"] = {
            "window_size_seconds": 30,
            "data_points_count": 30,
            "sensor_data": sensor_data  # TOUS les 30 points
        }
    
    return message

def simulate_critical_anomaly():
    """Crée un message avec anomalie critique simulée"""
    sensor_data = []
    base_time = datetime.now() - timedelta(seconds=30)
    
    # Création d'une séquence avec surchauffe progressive
    for i in range(30):  # Exactement 30 points
        timestamp = base_time + timedelta(seconds=i)
        # Simulation de surchauffe progressive
        temp = 40.0 + (i * 1.0)  # De 40°C à 70°C
        
        sensor_data.append({
            "timestamp": timestamp.isoformat() + "Z",
            "V_batt": 7.4 + np.random.normal(0, 0.1),
            "I_batt": 1.2 + np.random.normal(0, 0.2),
            "T_batt": temp + np.random.normal(0, 0.5),
            "V_bus": 7.8 + np.random.normal(0, 0.05),
            "I_bus": 0.8 + np.random.normal(0, 0.1),
            "V_solar": 15.2 + np.random.normal(0, 0.3),
            "I_solar": 1.5 + np.random.normal(0, 0.1)
        })
    
    message = {
        "header": {
            "message_id": np.random.randint(1000, 9999),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source": "EPS_MCU",
            "message_type": "ALERT_CRITICAL",
            "priority": "HIGH",
            "version": "1.0"
        },
        "payload": {
            "rule_triggered": "R1",
            "sensor_data": sensor_data[-10:],
            "actions_taken": ["CUT_POWER", "LED_RED", "BUZZER_ALARM"],
            "emergency_level": "HIGH",
            "temporal_window": {
                "window_size_seconds": 30,
                "data_points_count": 30,
                "sensor_data": sensor_data  # TOUS les 30 points
            }
        }
    }
    
    return message

def save_test_results(test_name, message, response):
    """Sauvegarde les résultats des tests dans data/obc/logs/"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"obc_test_{test_name}_{timestamp}.json"
        filepath = os.path.join(OBC_LOGS_DIR, filename)
        
        test_result = {
            "test_name": test_name,
            "timestamp": datetime.now().isoformat(),
            "message_sent": {
                "message_type": message.get('header', {}).get('message_type'),
                "message_id": message.get('header', {}).get('message_id'),
                "rule_triggered": message.get('payload', {}).get('rule_triggered')
            },
            "response_received": response,
            "test_summary": {
                "decision": response.get('decision'),
                "action": response.get('action'),
                "ai_level": response.get('ai_analysis', {}).get('ai_level'),
                "ai_score": response.get('ai_analysis', {}).get('ai_score')
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(test_result, f, indent=2)
            
        logger.info(f"Resultat test sauvegarde: {filename}")
        
    except Exception as e:
        logger.error(f"Erreur sauvegarde resultats: {e}")

def test_obc_system():
    """Test complet du système OBC"""
    logger.info("DEBUT TEST SYSTÈME OBC")
    print("TEST DU SYSTÈME OBC")
    print("=" * 50)
    print(f"Logs: {OBC_LOGS_DIR}")
    print("=" * 50)
    
    handler = OBCMessageHandler()
    response_gen = OBCResponseGenerator()
    
    # Test 1: Message SUMMARY normal
    print("\n1. TEST MESSAGE SUMMARY (normal)")
    logger.info("Test 1: Message SUMMARY normal")
    summary_msg = create_sample_mcu_message("SUMMARY")
    
    # DEBUG: Vérifier la structure du message
    print(f"DEBUG: Message keys: {list(summary_msg.keys())}")
    print(f"DEBUG: Payload keys: {list(summary_msg['payload'].keys())}")
    if 'temporal_window' in summary_msg['payload']:
        window_data = summary_msg['payload']['temporal_window']['sensor_data']
        print(f"DEBUG: Points dans temporal_window: {len(window_data)}")
    
    response = handler.process_mcu_message(summary_msg)
    print(f"   Message: {summary_msg['header']['message_type']}")
    print(f"   Reponse: {response['decision']} - {response['action']}")
    
    # Gestion sécurisée de ai_analysis
    ai_analysis = response.get('ai_analysis', {})
    ai_level = ai_analysis.get('ai_level', 'UNKNOWN')
    ai_score = ai_analysis.get('ai_score', -1)
    error_msg = ai_analysis.get('error', '')
    print(f"   Analyse IA: {ai_level} (score: {ai_score:.6f})")
    if error_msg:
        print(f"   Erreur IA: {error_msg}")
    
    # Sauvegarde du résultat
    save_test_results("summary_normal", summary_msg, response)
    
    # Test 2: Message ALERT_CRITICAL avec anomalie
    print("\n2. TEST MESSAGE ALERT_CRITICAL (anomalie)")
    logger.info("Test 2: Message ALERT_CRITICAL avec anomalie")
    critical_msg = simulate_critical_anomaly()
    response = handler.process_mcu_message(critical_msg)
    print(f"   Message: {critical_msg['header']['message_type']}")
    print(f"   Reponse: {response['decision']} - {response['action']}")
    
    ai_analysis = response.get('ai_analysis', {})
    ai_level = ai_analysis.get('ai_level', 'UNKNOWN')
    ai_score = ai_analysis.get('ai_score', -1)
    error_msg = ai_analysis.get('error', '')
    print(f"   Analyse IA: {ai_level} (score: {ai_score:.6f})")
    if error_msg:
        print(f"   Erreur IA: {error_msg}")
    
    # Sauvegarde du résultat
    save_test_results("critical_anomaly", critical_msg, response)
    
    # Test 3: Message sans données temporelles
    print("\n3. TEST MESSAGE SANS DONNEES TEMPORELLES")
    logger.info("Test 3: Message sans données temporelles")
    incomplete_msg = create_sample_mcu_message("SUMMARY", include_window=False)
    response = handler.process_mcu_message(incomplete_msg)
    print(f"   Message: {incomplete_msg['header']['message_type']}")
    print(f"   Reponse: {response['decision']} - {response.get('error', 'Aucune erreur')}")
    
    # Sauvegarde du résultat
    save_test_results("no_temporal_data", incomplete_msg, response)
    
    # Test 4: Génération réponse structurée (seulement si ai_analysis existe)
    print("\n4. TEST GENERATION REPONSE STRUCTUREE")
    logger.info("Test 4: Génération réponse structurée")
    
    # Réutiliser la réponse du test 2 qui devrait avoir des données
    test_response = handler.process_mcu_message(critical_msg)
    
    if 'ai_analysis' in test_response:
        structured_response = response_gen.generate_response(
            critical_msg, 
            test_response['ai_analysis'],
            {"decision": test_response['decision'], "action": test_response['action'], "notes": test_response.get('notes', '')}
        )
        print(f"   Reponse structuree generee: {structured_response['header']['message_type']}")
        print(f"   Priorite: {structured_response['header']['priority']}")
        
        # Sauvegarde de la réponse structurée
        save_test_results("structured_response", critical_msg, structured_response)
    else:
        print("   Impossible de generer la reponse structuree - analyse IA manquante")
        print(f"   Keys dans la reponse: {list(test_response.keys())}")
        logger.warning("Impossible de generer reponse structuree - analyse IA manquante")
    
    print("\n" + "=" * 50)
    print("TEST TERMINE")
    logger.info("TEST SYSTÈME OBC TERMINE AVEC SUCCES")

def main():
    """Point d'entrée principal avec gestion des arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test du système OBC")
    parser.add_argument("--test", choices=["all", "summary", "critical", "notemporal"], 
                       default="all", help="Type de test à exécuter")
    
    args = parser.parse_args()
    
    logger.info(f"Lancement test OBC - Type: {args.test}")
    print(f"Configuration logs: {OBC_LOGS_DIR}")
    
    test_obc_system()

if __name__ == "__main__":
    main()
