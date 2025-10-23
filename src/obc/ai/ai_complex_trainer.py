#!/usr/bin/env python3
"""
ENTRAÎNEMENT DU MODÈLE IA COMPLEXE OBC
LSTM Autoencoder pour détection d'anomalies temporelles
"""

import os
import numpy as np
import json
import logging
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib

def find_project_root():
    """Trouve le répertoire racine du projet de manière robuste"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Essayer différents niveaux de profondeur
    possible_roots = [
        os.path.dirname(os.path.dirname(os.path.dirname(current_dir))),  # src/obc/ai → racine
        os.path.join(current_dir, "..", "..", ".."),  # Alternative
        current_dir  # Dernier recours
    ]
    
    for root in possible_roots:
        root = os.path.abspath(root)
        # Vérifier si data/ai_training_base existe
        data_training = os.path.join(root, "data", "ai_training_base")
        if os.path.exists(data_training):
            return root
    
    # Si rien trouvé, utiliser le répertoire courant
    return current_dir

# Détermination des chemins
PROJECT_ROOT = find_project_root()
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "ai_training_base")  # CORRIGÉ: ai_training_base au lieu de mcu/training
OBC_AI_DIR = os.path.join(PROJECT_ROOT, "data", "ai_models", "model_complex")
OBC_LOGS_DIR = os.path.join(PROJECT_ROOT, "data", "mcu", "logs")

# Création des dossiers
os.makedirs(OBC_AI_DIR, exist_ok=True)
os.makedirs(OBC_LOGS_DIR, exist_ok=True)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OBC_LOGS_DIR, "obc_ai_training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("OBC_AI_Trainer")

logger.info(f"Repertoire racine detecte: {PROJECT_ROOT}")
logger.info(f"Dossier donnees: {DATA_DIR}")
logger.info(f"Dossier OBC AI: {OBC_AI_DIR}")

class OBCAITrainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.training_history = None
        self.anomaly_thresholds = {}
        
        # Initialisation des attributs pour eviter les erreurs
        self.sequences = None
        self.labels = None
        self.dataset_config = None
        self.training_sequences = None
        self.x_train = None
        self.x_val = None
        
        logger.info("Initialisation de l'entraineur IA OBC")

    def load_training_data(self):
        """Charge les donnees d'entrainement depuis ai_training_base"""
        logger.info("Chargement des donnees d'entrainement...")
        
        try:
            # Verifier que le dossier existe
            if not os.path.exists(DATA_DIR):
                logger.error(f"Dossier donnees non trouve: {DATA_DIR}")
                logger.info("Contenu du repertoire racine:")
                for item in os.listdir(PROJECT_ROOT):
                    logger.info(f"  - {item}")
                return False
            
            # Verifier les fichiers dans le dossier
            data_files = os.listdir(DATA_DIR)
            logger.info(f"Fichiers trouves dans {DATA_DIR}: {data_files}")
            
            # Chargement des sequences normalisees
            sequences_path = os.path.join(DATA_DIR, "ai_sequence_data.npy")
            if not os.path.exists(sequences_path):
                logger.error(f"Fichier non trouve: {sequences_path}")
                return False
                
            self.sequences = np.load(sequences_path)
            logger.info(f"Sequences chargees: {self.sequences.shape}")
            
            # Chargement des labels
            labels_path = os.path.join(DATA_DIR, "ai_sequence_labels.npy")
            if not os.path.exists(labels_path):
                logger.error(f"Fichier non trouve: {labels_path}")
                return False
                
            self.labels = np.load(labels_path)
            logger.info(f"Labels charges: {len(self.labels)}")
            
            # Analyse des labels
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            logger.info("Distribution des labels:")
            for label, count in zip(unique_labels, counts):
                logger.info(f"  {label}: {count} sequences ({count/len(self.labels)*100:.1f}%)")
            
            # Chargement du scaler
            scaler_path = os.path.join(DATA_DIR, "ai_sequence_scaler.pkl")
            if not os.path.exists(scaler_path):
                logger.error(f"Fichier non trouve: {scaler_path}")
                return False
                
            self.scaler = joblib.load(scaler_path)
            logger.info("Scaler charge")
            
            # Chargement de la configuration
            config_path = os.path.join(DATA_DIR, "dataset_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.dataset_config = json.load(f)
                logger.info("Configuration du dataset chargee")
            else:
                logger.warning("Configuration du dataset non trouvee")
                self.dataset_config = {}
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des donnees: {e}")
            return False

    def prepare_data(self):
        """Prepare les donnees pour l'entrainement"""
        logger.info("Preparation des donnees...")
        
        # Verifier que les donnees sont chargees
        if self.sequences is None or self.labels is None:
            logger.error("Donnees non chargees - Appelez load_training_data() d'abord")
            return False
        
        # Utiliser seulement les sequences NORMALES pour l'entrainement autoencoder
        normal_indices = np.where(self.labels == "NORMAL")[0]
        
        if len(normal_indices) == 0:
            logger.warning("Aucune sequence NORMAL trouvee, utilisation de toutes les donnees")
            normal_indices = np.arange(len(self.sequences))
        
        self.training_sequences = self.sequences[normal_indices]
        logger.info(f"Sequences d'entrainement (normales): {self.training_sequences.shape}")
        
        # Separation train/validation (80/20)
        split_idx = int(0.8 * len(self.training_sequences))
        self.x_train = self.training_sequences[:split_idx]
        self.x_val = self.training_sequences[split_idx:]
        
        logger.info(f"Train: {self.x_train.shape}, Validation: {self.x_val.shape}")
        return True

    def build_lstm_autoencoder(self):
        """Construit le modele LSTM Autoencoder"""
        logger.info("Construction du modele LSTM Autoencoder...")
        
        # Parametres du modele
        timesteps = self.sequences.shape[1]
        n_features = self.sequences.shape[2]
        encoding_dim = 32  # Dimension de l'espace latent
        
        logger.info(f"Architecture: {timesteps} timesteps x {n_features} features")
        logger.info(f"Dimension latente: {encoding_dim}")
        
        # Encoder
        inputs = Input(shape=(timesteps, n_features))
        encoded = LSTM(64, activation='relu', return_sequences=True, name="encoder_lstm1")(inputs)
        encoded = LSTM(32, activation='relu', return_sequences=False, name="encoder_lstm2")(encoded)
        encoded = Dense(encoding_dim, activation='relu', name="bottleneck")(encoded)
        
        # Decoder
        decoded = RepeatVector(timesteps, name="repeat_vector")(encoded)
        decoded = LSTM(32, activation='relu', return_sequences=True, name="decoder_lstm1")(decoded)
        decoded = LSTM(64, activation='relu', return_sequences=True, name="decoder_lstm2")(decoded)
        decoded = TimeDistributed(Dense(n_features), name="output")(decoded)
        
        # Modele complet
        self.model = Model(inputs, decoded, name="lstm_autoencoder")
        
        # Compilation
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("Modele compile avec succes")
        
        # Resume du modele
        total_params = self.model.count_params()
        logger.info(f"Nombre total de parametres: {total_params:,}")
        
        return True

    def train_model(self, epochs=100, batch_size=32):
        """Entraine le modele"""
        logger.info("Debut de l'entrainement...")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
        ]
        
        logger.info(f"Parametres d'entrainement: {epochs} epochs, batch_size={batch_size}")
        
        # Entrainement
        self.training_history = self.model.fit(
            self.x_train, self.x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.x_val, self.x_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Analyse des resultats
        final_train_loss = self.training_history.history['loss'][-1]
        final_val_loss = self.training_history.history['val_loss'][-1]
        final_epochs = len(self.training_history.history['loss'])
        
        logger.info(f"Entrainement termine apres {final_epochs} epochs")
        logger.info(f"Loss finale - Train: {final_train_loss:.6f}, Validation: {final_val_loss:.6f}")
        
        return True

    def calculate_anomaly_thresholds(self):
        """Calcule les seuils d'anomalie bases sur l'erreur de reconstruction"""
        logger.info("Calcul des seuils d'anomalie...")
        
        # Predictions sur les donnees d'entrainement
        logger.info("Calcul des erreurs de reconstruction...")
        train_predictions = self.model.predict(self.x_train, verbose=0)
        train_errors = np.mean(np.square(self.x_train - train_predictions), axis=(1, 2))
        
        # Calcul des statistiques
        mean_error = np.mean(train_errors)
        std_error = np.std(train_errors)
        min_error = np.min(train_errors)
        max_error = np.max(train_errors)
        
        logger.info(f"Statistiques des erreurs - Mean: {mean_error:.6f}, Std: {std_error:.6f}")
        logger.info(f"Plage des erreurs - Min: {min_error:.6f}, Max: {max_error:.6f}")
        
        # Calcul des seuils
        self.anomaly_thresholds = {
            "normal_threshold": float(mean_error + std_error),
            "warning_threshold": float(mean_error + 2 * std_error),
            "critical_threshold": float(mean_error + 3 * std_error),
            "training_stats": {
                "mean_error": float(mean_error),
                "std_error": float(std_error),
                "min_error": float(min_error),
                "max_error": float(max_error),
                "training_samples": len(train_errors)
            }
        }
        
        logger.info("Seuils calcules:")
        logger.info(f"  NORMAL  < {self.anomaly_thresholds['normal_threshold']:.6f}")
        logger.info(f"  WARNING < {self.anomaly_thresholds['warning_threshold']:.6f}") 
        logger.info(f"  CRITICAL >= {self.anomaly_thresholds['critical_threshold']:.6f}")
        
        return True

    def save_model_and_thresholds(self):
        """Sauvegarde le modele et les seuils"""
        logger.info("Sauvegarde du modele et des seuils...")
        
        try:
            # Sauvegarde du modele
            model_path = os.path.join(OBC_AI_DIR, "ai_model_lstm_autoencoder.h5")
            self.model.save(model_path)
            logger.info(f"Modele sauvegarde: {model_path}")
            
            # Sauvegarde des seuils
            thresholds_path = os.path.join(OBC_AI_DIR, "ai_thresholds.json")
            thresholds_data = {
                "anomaly_thresholds": self.anomaly_thresholds,
                "training_date": datetime.now().isoformat(),
                "model_architecture": "LSTM Autoencoder",
                "input_shape": self.sequences.shape[1:],
                "dataset_info": self.dataset_config,
                "training_samples": len(self.x_train),
                "validation_samples": len(self.x_val)
            }
            
            with open(thresholds_path, 'w') as f:
                json.dump(thresholds_data, f, indent=2)
            logger.info(f"Seuils sauvegardes: {thresholds_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
            return False

    def generate_training_report(self):
        """Genere un rapport d'entrainement"""
        logger.info("Generation du rapport d'entrainement...")
        
        try:
            # Graphiques de performance
            plt.figure(figsize=(15, 5))
            
            # Loss
            plt.subplot(1, 3, 1)
            plt.plot(self.training_history.history['loss'], label='Train Loss', linewidth=2)
            plt.plot(self.training_history.history['val_loss'], label='Val Loss', linewidth=2)
            plt.title('Evolution de la Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # MAE
            plt.subplot(1, 3, 2)
            plt.plot(self.training_history.history['mae'], label='Train MAE', linewidth=2)
            plt.plot(self.training_history.history['val_mae'], label='Val MAE', linewidth=2)
            plt.title('Evolution du MAE')
            plt.xlabel('Epochs')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Distribution des erreurs
            plt.subplot(1, 3, 3)
            train_predictions = self.model.predict(self.x_train, verbose=0)
            train_errors = np.mean(np.square(self.x_train - train_predictions), axis=(1, 2))
            
            plt.hist(train_errors, bins=50, alpha=0.7, edgecolor='black')
            plt.axvline(self.anomaly_thresholds['normal_threshold'], color='orange', linestyle='--', label='Seuil Normal')
            plt.axvline(self.anomaly_thresholds['warning_threshold'], color='red', linestyle='--', label='Seuil Warning')
            plt.axvline(self.anomaly_thresholds['critical_threshold'], color='darkred', linestyle='--', label='Seuil Critical')
            plt.title('Distribution des erreurs')
            plt.xlabel('Erreur de reconstruction')
            plt.ylabel('Frequence')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            report_path = os.path.join(OBC_LOGS_DIR, "training_report.png")
            plt.savefig(report_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Rapport d'entrainement genere: {report_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur generation rapport: {e}")
            return False

    def run_training_pipeline(self):
        """Execute le pipeline complet d'entrainement"""
        logger.info("DEMARRAGE PIPELINE D'ENTRAINEMENT OBC IA")
        
        try:
            # Pipeline sequentiel avec verification
            steps = [
                ("Chargement des donnees", self.load_training_data),
                ("Preparation des donnees", self.prepare_data),
                ("Construction du modele", self.build_lstm_autoencoder),
                ("Entrainement", lambda: self.train_model(epochs=50)),  # Reduit pour test
                ("Calcul des seuils", self.calculate_anomaly_thresholds),
                ("Sauvegarde", self.save_model_and_thresholds),
                ("Generation rapport", self.generate_training_report)
            ]
            
            for step_name, step_func in steps:
                logger.info(f"ETAPE: {step_name}")
                if not step_func():
                    logger.error(f"Echec de l'etape: {step_name}")
                    return False
                logger.info(f"ETAPE TERMINEE: {step_name}")
            
            logger.info("PIPELINE D'ENTRAINEMENT TERMINE AVEC SUCCES")
            return True
            
        except Exception as e:
            logger.error(f"ERREUR DURANT L'ENTRAINEMENT: {e}")
            return False

def main():
    """Point d'entree principal"""
    logger.info("=" * 60)
    logger.info("SYSTEME D'ENTRAINEMENT IA OBC - EPS GUARDIAN")
    logger.info("=" * 60)
    
    trainer = OBCAITrainer()
    success = trainer.run_training_pipeline()
    
    if success:
        logger.info("ENTRAINEMENT REUSSI - Le modele OBC est pret!")
        return 0
    else:
        logger.error("ENTRAINEMENT ECHOUE - Verifiez les donnees et les logs")
        return 1

if __name__ == "__main__":
    exit(main())
