import logging
import math
import json
import os
from datetime import datetime, timedelta
from collections import deque
from mcu_logger import MCULogger

# ======================================================
# INTERFACE DE COMMUNICATION MCU → OBC AVEC BUFFER TEMPOREL
# ======================================================
class OBCInterface:
    def __init__(self, window_size_seconds=30, max_data_points=100):
        self.logger = MCULogger("mcu_obc_interface.log")
        self.window_size = timedelta(seconds=window_size_seconds)
        self.data_buffer = deque(maxlen=max_data_points)  # Mémoire circulaire
        self.message_counter = 0
        
        # Créer le dossier de sortie pour les messages OBC
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.output_dir = os.path.join(BASE_DIR, "data", "mcu", "outputs", "obc_messages")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger.info(f"Interface OBC initialisée (buffer: {window_size_seconds}s)")
    
    def add_sensor_data(self, sensor_data):
        """Ajoute les données capteurs au buffer temporel avec timestamp"""
        data_point = {
            "timestamp": datetime.now(),
            "data": sensor_data.copy()
        }
        self.data_buffer.append(data_point)
        
        # Nettoyage des données trop anciennes
        self._clean_old_data()
    
    def _clean_old_data(self):
        """Supprime les données plus vieilles que la fenêtre temporelle"""
        cutoff_time = datetime.now() - self.window_size
        while self.data_buffer and self.data_buffer[0]["timestamp"] < cutoff_time:
            self.data_buffer.popleft()
    
    def get_window_data(self):
        """Retourne les données de la fenêtre temporelle (30 dernières secondes)"""
        self._clean_old_data()
        return [point["data"] for point in self.data_buffer]
    
    def send_to_obc(self, message_type, payload, include_window_data=True):
        """Envoie un message à l'OBC avec données de fenêtre optionnelles"""
        try:
            self.message_counter += 1
            
            # Préparer le message complet
            obc_message = {
                "header": {
                    "message_id": self.message_counter,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "source": "EPS_MCU",
                    "message_type": message_type,
                    "priority": "HIGH" if "CRITICAL" in message_type else "MEDIUM",
                    "version": "1.0"
                },
                "payload": payload
            }
            
            # Ajouter les données de fenêtre pour les SUMMARY et ALERT
            if include_window_data and message_type in ["SUMMARY", "ALERT_CRITICAL"]:
                window_data = self.get_window_data()
                obc_message["payload"]["temporal_window"] = {
                    "window_size_seconds": self.window_size.total_seconds(),
                    "data_points_count": len(window_data),
                    "sensor_data": window_data[-10:]  # Derniers 10 points pour éviter overflow
                }
            
            # Sauvegarder le message dans un fichier
            message_filename = f"obc_message_{self.message_counter:06d}_{message_type}.json"
            message_path = os.path.join(self.output_dir, message_filename)
            
            with open(message_path, 'w') as f:
                json.dump(obc_message, f, indent=2, default=str)
            
            # Simulation console
            print(f"\n{'='*60}")
            print(f"📡 [OBC-TX] Message #{self.message_counter}")
            print(f"   Type: {message_type} | Priorité: {obc_message['header']['priority']}")
            print(f"   Timestamp: {obc_message['header']['timestamp']}")
            print(f"   Fenêtre: {len(obc_message['payload'].get('temporal_window', {}).get('sensor_data', []))} points")
            print(f"   Sauvegardé: {message_filename}")
            print(f"{'='*60}")
            
            self.logger.info(f"OBC_TX [{message_type}] → {len(window_data) if include_window_data else 0} pts données")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur envoi OBC: {e}")
            return False

# Instance globale
obc_interface = OBCInterface(window_size_seconds=30)

def send_to_obc(message_type, payload):
    """Fonction legacy pour compatibilité"""
    return obc_interface.send_to_obc(message_type, payload)

class MCU_RuleEngine:
    def __init__(self, window_size=30):
        self.logger = MCULogger("mcu_rule_engine.log")
        self.window_size = window_size
        self.obc_interface = obc_interface  # Utilise l'interface globale

        # Buffers pour persistance des règles
        self.overcurrent_buffer = deque(maxlen=3)
        self.ratio_buffer = deque(maxlen=10)
        self.oscillation_buffer = deque(maxlen=10)
        self.heartbeat_counter = 0

        # État système
        self.last_alert_time = {}
        self.alert_throttle_interval = 60
        self.system_state = "NORMAL"
        self.actions_taken = []

        # Seuils
        self.thresholds = {
            "T_batt_critical": 60.0,
            "I_batt_overcurrent": 3.0,  
            "V_batt_undervoltage": 3.2,
            "ratio_min": 0.7,
            "T_batt_sensor_fault": 120.0,
            "V_batt_sensor_min": 0.0,
            "V_batt_sensor_max": 20.0,
            "I_batt_sensor_max": 10.0,
        }

        self.logger.info("MCU_RuleEngine initialisé avec buffer temporel 30s")

    def trigger_action(self, action_type, details=None):
        """Déclenche une action physique et la logge"""
        mapping = {
            "CUT_POWER": "Coupure d'urgence de l'alimentation",
            "ACTIVATE_COOLING": "Activation du refroidissement", 
            "REDUCE_LOAD": "Réduction de la charge",
            "ISOLATE_BATTERY": "Isolation de la batterie",
            "LED_RED": "LED rouge activée",
            "LED_YELLOW": "LED jaune activée", 
            "LED_GREEN": "LED verte activée",
            "BUZZER_ALARM": "Alarme sonore activée",
            "INCREASE_LOGGING": "Logging intensif activé"
        }
        
        msg = mapping.get(action_type, "Action inconnue")
        log_msg = f"ACTION: {msg}"
        if details:
            log_msg += f" - Détails: {details}"
            
        self.logger.info(log_msg)
        self.actions_taken.append({
            "action": action_type,
            "timestamp": datetime.now().isoformat(),
            "details": details
        })
        return True

    # ======================================================
    # RÈGLES R1-R7 (inchangées mais optimisées)
    # ======================================================
    def _check_sensor_fault(self, s):
        faults = []
        for key, value in s.items():
            if value is None or (isinstance(value, float) and math.isnan(value)):
                faults.append(f"{key}_NaN")
        if s["T_batt"] > self.thresholds["T_batt_sensor_fault"]:
            faults.append("T_batt_out_of_range")
        if not (self.thresholds["V_batt_sensor_min"] <= s["V_batt"] <= self.thresholds["V_batt_sensor_max"]):
            faults.append("V_batt_out_of_range") 
        if abs(s["I_batt"]) > self.thresholds["I_batt_sensor_max"]:
            faults.append("I_batt_out_of_range")
        return faults

    def _check_overheat(self, s): 
        return s["T_batt"] > self.thresholds["T_batt_critical"]

    def _check_overcurrent(self, s):
        self.overcurrent_buffer.append(abs(s["I_batt"]) > self.thresholds["I_batt_overcurrent"])
        return len(self.overcurrent_buffer) == 3 and all(self.overcurrent_buffer)

    def _check_deep_discharge(self, s):
        return (s["V_batt"] < self.thresholds["V_batt_undervoltage"] and s["I_batt"] < 0)

    def _check_dcdc_ratio(self, s):
        ratio = s["V_bus"] / s["V_solar"] if s["V_solar"] > 0.1 else 0
        self.ratio_buffer.append(ratio)
        if len(self.ratio_buffer) >= 5:
            avg_ratio = sum(self.ratio_buffer) / len(self.ratio_buffer)
            return avg_ratio < self.thresholds["ratio_min"]
        return False

    def _check_oscillation(self, s):
        self.oscillation_buffer.append(s["V_bus"])
        if len(self.oscillation_buffer) >= 5:
            mean = sum(self.oscillation_buffer) / len(self.oscillation_buffer)
            variance = sum((x - mean) ** 2 for x in self.oscillation_buffer) / len(self.oscillation_buffer)
            return variance > 0.1
        return False

    def _throttle_alert(self, alert_type):
        """Anti-spam des alertes"""
        current_time = datetime.now().timestamp()
        last_time = self.last_alert_time.get(alert_type, 0)
        
        if current_time - last_time < self.alert_throttle_interval:
            return False
            
        self.last_alert_time[alert_type] = current_time
        return True

    # ======================================================
    # APPLICATION DES RÈGLES AVEC TRANSMISSION OBC COMPLÈTE
    # ======================================================
    def apply_rules(self, s):
        """Applique les règles R1-R7 avec transmission OBC et données temporelles"""
        # Ajouter les données au buffer temporel
        self.obc_interface.add_sensor_data(s)
        
        # Réinitialiser les actions pour ce cycle
        self.actions_taken = []
        actions = []
        message_type = "STATUS_OK"
        details = {"rule_triggered": None, "values": {}}

        # R6: Vérification défaut capteur en premier
        sensor_faults = self._check_sensor_fault(s)
        if sensor_faults:
            self.trigger_action("LED_YELLOW", "Défaut capteur détecté")
            self.trigger_action("INCREASE_LOGGING", f"Faults: {sensor_faults}")
            message_type = "DIAGNOSTIC_SENSOR"
            details.update({
                "sensor_faults": sensor_faults,
                "actions_taken": [action["action"] for action in self.actions_taken]
            })
            
            # Envoi OBC SANS données de fenêtre (défaut instantané)
            self.obc_interface.send_to_obc(message_type, details, include_window_data=False)
            self.logger.warning(f"Défaut capteur: {sensor_faults}")
            return actions, message_type, details

        # R1: Surchauffe batterie (critique immédiat)
        if self._check_overheat(s):
            if self._throttle_alert("R1"):
                self.trigger_action("CUT_POWER", f"Température critique: {s['T_batt']}°C")
                self.trigger_action("LED_RED")
                self.trigger_action("BUZZER_ALARM")
                message_type = "ALERT_CRITICAL"
                details.update({
                    "rule_triggered": "R1", 
                    "T_batt": s["T_batt"],
                    "actions_taken": [action["action"] for action in self.actions_taken],
                    "emergency_level": "HIGH"
                })
                
                # Envoi OBC AVEC données de fenêtre
                self.obc_interface.send_to_obc(message_type, details, include_window_data=True)
                self.logger.critical(f"R1 - Surchauffe batterie: {s['T_batt']}°C")
                self.system_state = "CRITICAL"
                return actions, message_type, details

        # R3: Décharge profonde (critique immédiat)
        if self._check_deep_discharge(s):
            if self._throttle_alert("R3"):
                self.trigger_action("ISOLATE_BATTERY", f"Tension basse: {s['V_batt']}V")
                self.trigger_action("LED_RED")
                self.trigger_action("BUZZER_ALARM")
                message_type = "ALERT_CRITICAL"
                details.update({
                    "rule_triggered": "R3",
                    "V_batt": s["V_batt"],
                    "I_batt": s["I_batt"],
                    "actions_taken": [action["action"] for action in self.actions_taken],
                    "emergency_level": "HIGH"
                })
                
                self.obc_interface.send_to_obc(message_type, details, include_window_data=True)
                self.logger.critical(f"R3 - Décharge profonde: V={s['V_batt']}V")
                self.system_state = "CRITICAL"
                return actions, message_type, details

        # R2: Surcharge courant (persistance 3s)
        if self._check_overcurrent(s):
            if self._throttle_alert("R2"):
                self.trigger_action("REDUCE_LOAD", f"Courant élevé: {s['I_batt']}A")
                self.trigger_action("LED_RED")
                message_type = "ALERT_CRITICAL"
                details.update({
                    "rule_triggered": "R2",
                    "I_batt": s["I_batt"],
                    "actions_taken": [action["action"] for action in self.actions_taken],
                    "emergency_level": "HIGH"
                })
                
                self.obc_interface.send_to_obc(message_type, details, include_window_data=True)
                self.logger.critical(f"R2 - Surcharge courant: {s['I_batt']}A")
                self.system_state = "CRITICAL"
                return actions, message_type, details

        # R4: Ratio DC/DC anormal
        if self._check_dcdc_ratio(s):
            ratio_value = s["V_bus"] / s["V_solar"] if s["V_solar"] > 0.1 else 0
            self.trigger_action("REDUCE_LOAD", f"Ratio anormal: {ratio_value:.2f}")
            self.trigger_action("LED_YELLOW")
            message_type = "SUMMARY"
            details.update({
                "rule_triggered": "R4",
                "ratio": ratio_value,
                "actions_taken": [action["action"] for action in self.actions_taken],
                "emergency_level": "MEDIUM"
            })
            
            self.obc_interface.send_to_obc(message_type, details, include_window_data=True)
            self.logger.warning(f"R4 - Ratio DC/DC anormal: {ratio_value:.2f}")

        # R5: Oscillation bus
        elif self._check_oscillation(s):
            self.trigger_action("INCREASE_LOGGING", "Oscillation détectée")
            self.trigger_action("LED_YELLOW")
            message_type = "SUMMARY"
            details.update({
                "rule_triggered": "R5",
                "actions_taken": [action["action"] for action in self.actions_taken],
                "emergency_level": "MEDIUM"
            })
            
            self.obc_interface.send_to_obc(message_type, details, include_window_data=True)
            self.logger.warning("R5 - Oscillation bus détectée")

        # R7: État normal + heartbeat périodique
        else:
            self.trigger_action("LED_GREEN", "Système nominal")
            message_type = "STATUS_HEARTBEAT"
            details.update({"rule_triggered": "R7"})
            
            # Heartbeat toutes les 10 cycles (sans données de fenêtre)
            self.heartbeat_counter += 1
            if self.heartbeat_counter % 10 == 0:
                self.obc_interface.send_to_obc("STATUS_HEARTBEAT", {
                    "system_status": "NORMAL",
                    "uptime_cycles": self.heartbeat_counter,
                    "rule_engine_status": "OPERATIONAL"
                }, include_window_data=False)
                
            self.system_state = "NORMAL"

        details["actions_triggered"] = self.actions_taken.copy()
        return actions, message_type, details

    def get_system_state(self):
        return self.system_state

    def get_recent_actions(self):
        return self.actions_taken