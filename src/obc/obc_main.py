#!/usr/bin/env python3
"""
POINT D'ENTREE PRINCIPAL DU SYSTÈME OBC
Cerveau principal du module On-Board Computer
"""

import os
import sys
import time
import json
import logging
from datetime import datetime

# Configuration des chemins
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, current_dir)

from interface.obc_message_handler import OBCMessageHandler
from interface.obc_response_generator import OBCResponseGenerator
from ai.ai_complex_inference import obc_ai

# ========== CHEMINS CORRIGÉS ==========
# Tous les outputs dans obc/
OBC_LOGS_DIR = os.path.join(project_root, "data", "obc", "logs")
OBC_OUTPUTS_DIR = os.path.join(project_root, "data", "obc", "outputs")
OBC_SYSTEM_DIR = os.path.join(project_root, "data", "obc", "system")

# Création des dossiers OBC
os.makedirs(OBC_LOGS_DIR, exist_ok=True)
os.makedirs(OBC_OUTPUTS_DIR, exist_ok=True)
os.makedirs(OBC_SYSTEM_DIR, exist_ok=True)

# Configuration des logs - maintenant dans obc/logs/
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OBC_LOGS_DIR, "obc_main_system.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("OBC_Main")

class OBCSystem:
    def __init__(self):
        self.logger = logger
        self.is_running = False
        self.processed_messages = 0
        self.start_time = None
        self.system_id = f"OBC_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Import des composants
        try:
            self.message_handler = OBCMessageHandler()
            self.response_generator = OBCResponseGenerator()
            self.ai_system = obc_ai
            
            # Vérifier le statut de l'IA
            ai_status = self.ai_system.get_model_info()
            self.logger.info(f"Systeme OBC initialise avec succes")
            self.logger.info(f"Repertoire logs: {OBC_LOGS_DIR}")
            self.logger.info(f"Repertoire outputs: {OBC_OUTPUTS_DIR}")
            self.logger.info(f"Statut IA: {ai_status['status']}")
            
        except ImportError as e:
            self.logger.error(f"Erreur import composants: {e}")
            raise

    def start_system(self):
        """Démarre le système OBC"""
        self.is_running = True
        self.start_time = datetime.now()
        self.processed_messages = 0
        
        ai_status = self.ai_system.get_model_info()
        
        self.logger.info("DEMARRAGE SYSTÈME OBC")
        self.logger.info(f"System ID: {self.system_id}")
        self.logger.info(f"Heure debut: {self.start_time}")
        self.logger.info(f"Statut IA: {ai_status['status']}")
        self.logger.info("En attente de messages MCU...")
        
        # Sauvegarde du statut de démarrage
        self._save_system_status()
        
        return True

    def process_single_message(self, message_json):
        """
        Traite un message unique (mode manuel)
        """
        if not self.is_running:
            self.logger.warning("Systeme OBC non demarre - Demarrage automatique")
            self.start_system()
        
        try:
            # Traitement du message
            response = self.message_handler.process_mcu_message(message_json)
            self.processed_messages += 1
            
            # Génération réponse structurée
            structured_response = self.response_generator.generate_response(
                message_json,
                response['ai_analysis'],
                {
                    "decision": response['decision'],
                    "action": response['action'],
                    "notes": response.get('notes', '')
                }
            )
            
            self.logger.info(f"Message #{self.processed_messages} traite - Decision: {response['decision']}")
            
            # Sauvegarde de la réponse dans obc/outputs/
            self._save_response(structured_response, message_json)
            
            return structured_response
            
        except Exception as e:
            self.logger.error(f"Erreur traitement message: {e}")
            error_response = self.response_generator.generate_error_response(
                message_json if isinstance(message_json, dict) else {"header": {}},
                str(e)
            )
            self._save_response(error_response, message_json, is_error=True)
            return error_response

    def _save_response(self, response, original_message, is_error=False):
        """Sauvegarde la réponse dans obc/outputs/"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = "error" if is_error else "response"
            filename = f"obc_{prefix}_{timestamp}_{self.processed_messages:06d}.json"
            filepath = os.path.join(OBC_OUTPUTS_DIR, filename)
            
            response_data = {
                "system_id": self.system_id,
                "timestamp": datetime.now().isoformat(),
                "processed_count": self.processed_messages,
                "response": response,
                "original_message_summary": {
                    "message_id": original_message.get('header', {}).get('message_id', 'unknown'),
                    "message_type": original_message.get('header', {}).get('message_type', 'unknown'),
                    "source": original_message.get('header', {}).get('source', 'unknown')
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(response_data, f, indent=2)
                
            self.logger.debug(f"Reponse sauvegardee: {filename}")
            
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde reponse: {e}")

    def _save_system_status(self):
        """Sauvegarde le statut du système dans obc/system/"""
        try:
            status = self.get_system_status()
            filename = f"obc_system_status_{self.system_id}.json"
            filepath = os.path.join(OBC_SYSTEM_DIR, filename)
            
            with open(filepath, 'w') as f:
                json.dump(status, f, indent=2)
                
            self.logger.debug(f"Statut systeme sauvegarde: {filename}")
            
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde statut: {e}")

    def run_continuous_mode(self, message_callback=None):
        """
        Exécute le mode continu (écoute permanente)
        """
        if not self.start_system():
            return
        
        self.logger.info("MODE CONTINU ACTIVE")
        
        try:
            cycle_count = 0
            while self.is_running:
                cycle_count += 1
                
                # Sauvegarde périodique du statut
                if cycle_count % 10 == 0:
                    self._log_system_status()
                    self._save_system_status()
                    
                    # Simulation: générer un message de test périodiquement
                    from simulation.obc_simulate_incoming_data import create_sample_mcu_message
                    test_message = create_sample_mcu_message("SUMMARY")
                    self.process_single_message(test_message)
                
                # Attente avant prochaine vérification
                time.sleep(2)
                
        except KeyboardInterrupt:
            self.logger.info("Arret demande par l'utilisateur")
        except Exception as e:
            self.logger.error(f"Erreur mode continu: {e}")
        finally:
            self.stop_system()

    def stop_system(self):
        """Arrête le système OBC"""
        self.is_running = False
        end_time = datetime.now()
        runtime = end_time - self.start_time if self.start_time else None
        
        # Sauvegarde finale du statut
        final_status = self.get_system_status()
        final_status["shutdown_time"] = end_time.isoformat()
        final_status["total_runtime_seconds"] = runtime.total_seconds() if runtime else 0
        
        final_filename = f"obc_shutdown_report_{self.system_id}.json"
        final_filepath = os.path.join(OBC_SYSTEM_DIR, final_filename)
        
        try:
            with open(final_filepath, 'w') as f:
                json.dump(final_status, f, indent=2)
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde rapport final: {e}")
        
        self.logger.info("ARRET SYSTÈME OBC")
        self.logger.info(f"System ID: {self.system_id}")
        self.logger.info(f"Messages traites: {self.processed_messages}")
        if runtime:
            self.logger.info(f"Temps d'execution: {runtime}")
        self.logger.info(f"Rapport final: {final_filename}")

    def get_system_status(self):
        """Retourne le statut du système"""
        ai_info = self.ai_system.get_model_info() if hasattr(self, 'ai_system') else {"status": "UNKNOWN"}
        
        status = {
            "system_id": self.system_id,
            "is_running": self.is_running,
            "ai_loaded": self.ai_system.is_loaded if hasattr(self, 'ai_system') else False,
            "ai_status": ai_info['status'],
            "processed_messages": self.processed_messages,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "current_time": datetime.now().isoformat(),
            "directories": {
                "logs": OBC_LOGS_DIR,
                "outputs": OBC_OUTPUTS_DIR,
                "system": OBC_SYSTEM_DIR
            }
        }
        
        if hasattr(self, 'ai_system') and self.ai_system.is_loaded:
            status.update({
                "ai_model_info": self.ai_system.get_model_info()
            })
            
        return status

    def _log_system_status(self):
        """Log le statut périodique du système"""
        status = self.get_system_status()
        self.logger.info(f"Statut systeme - Messages: {status['processed_messages']}, "
                        f"IA: {status['ai_status']}")

# Instance globale
obc_system = OBCSystem()

def main():
    """Point d'entrée principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Systeme OBC EPS Guardian")
    parser.add_argument("--mode", choices=["single", "continuous"], default="single",
                       help="Mode d'execution (single/continuous)")
    parser.add_argument("--message", type=str, help="Message JSON a traiter (mode single)")
    parser.add_argument("--message-file", type=str, help="Fichier JSON contenant le message")
    
    args = parser.parse_args()
    
    system = OBCSystem()
    
    try:
        if args.mode == "single":
            # Mode traitement unique
            message_data = None
            
            if args.message_file:
                # Charger depuis fichier
                with open(args.message_file, 'r') as f:
                    message_data = json.load(f)
            elif args.message:
                # Charger depuis string JSON
                message_data = json.loads(args.message)
            else:
                # Message par défaut
                from simulation.obc_simulate_incoming_data import create_sample_mcu_message
                message_data = create_sample_mcu_message("SUMMARY")
            
            print("MODE TRAITEMENT UNIQUE OBC")
            print(f"Repertoire outputs: {OBC_OUTPUTS_DIR}")
            response = system.process_single_message(message_data)
            print("\nReponse generee:")
            print(json.dumps(response, indent=2))
            
        elif args.mode == "continuous":
            # Mode continu
            print("MODE CONTINU OBC")
            print(f"System ID: {system.system_id}")
            print(f"Repertoire logs: {OBC_LOGS_DIR}")
            print(f"Repertoire outputs: {OBC_OUTPUTS_DIR}")
            print("Ctrl+C pour arreter")
            system.run_continuous_mode()
            
        else:
            # Mode démonstration
            print("SYSTÈME OBC EPS GUARDIAN")
            print("=" * 50)
            print(f"Repertoire logs: {OBC_LOGS_DIR}")
            print(f"Repertoire outputs: {OBC_OUTPUTS_DIR}")
            print(f"Repertoire system: {OBC_SYSTEM_DIR}")
            print("=" * 50)
            
            # Test avec un message exemple
            from simulation.obc_simulate_incoming_data import create_sample_mcu_message
            test_message = create_sample_mcu_message("SUMMARY")
            
            print("Test automatique avec message exemple:")
            response = system.process_single_message(test_message)
            print("\nReponse generee:")
            print(json.dumps(response, indent=2))
            
            print("\nStatut systeme complet:")
            print(json.dumps(system.get_system_status(), indent=2))
            
    except Exception as e:
        logger.error(f"Erreur execution: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
