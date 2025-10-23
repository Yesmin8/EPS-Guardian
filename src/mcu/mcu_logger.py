import logging
import sys
import os
from datetime import datetime

class MCULogger:
    """
    Gestionnaire de logs universel pour le système MCU + IA.
    Compatible avec setup_logger() des anciennes versions.
    Supporte UTF-8 et évite la duplication des handlers.
    """
    def __init__(self, log_file="mcu_rule_engine.log", name="MCU_Logger"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Empêche d'ajouter plusieurs fois les mêmes handlers
        if not self.logger.handlers:
            # Format du log avec date et millisecondes
            formatter = logging.Formatter(
                "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

            # --- Dossier de logs ---
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            log_dir = os.path.join(BASE_DIR, "data", "mcu", "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, log_file)

            # --- Handler fichier (UTF-8) ---
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)

            # --- Handler console (UTF-8 si supporté) ---
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            try:
                # Python 3.7+ compatible
                console_handler.stream.reconfigure(encoding="utf-8")
            except Exception:
                pass

            # Ajouter les handlers
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

            print(f"Logging initialisé → {log_path}")

    # === Méthodes de niveau ===
    def info(self, message):
        try:
            self.logger.info(message)
        except UnicodeEncodeError:
            self.logger.info(message.encode("utf-8", "ignore").decode("utf-8"))

    def warning(self, message):
        try:
            self.logger.warning(message)
        except UnicodeEncodeError:
            self.logger.warning(message.encode("utf-8", "ignore").decode("utf-8"))

    def error(self, message):
        try:
            self.logger.error(message)
        except UnicodeEncodeError:
            self.logger.error(message.encode("utf-8", "ignore").decode("utf-8"))

    def critical(self, message):
        try:
            self.logger.critical(message)
        except UnicodeEncodeError:
            self.logger.critical(message.encode("utf-8", "ignore").decode("utf-8"))


# === Fonction de compatibilité rétro ===
def setup_logger(log_file="mcu_rule_engine.log"):
    """
    Compatibilité avec l'ancien code :
    MCU_MainLoop utilise setup_logger(), donc on retourne une instance de MCULogger.
    """
    return MCULogger(log_file)