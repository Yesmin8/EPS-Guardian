#!/usr/bin/env python3
"""
Entraîneur de modèle IA pour EPS GUARDIAN
Entraîne un autoencodeur léger sur les données normales pour détection d'anomalies
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import sys
from datetime import datetime

# Désactiver les warnings TensorFlow pour une sortie plus propre
tf.get_logger().setLevel('ERROR')

class AIModelTrainer:
    def __init__(self):
        self.model = None
        self.history = None
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        # CHEMINS CORRIGÉS
        self.model_dir = os.path.join(self.base_dir, "data", "ai_models", "model_simple")
        self.training_dir = os.path.join(self.base_dir, "data", "training_data")  # ← CORRIGÉ ICI
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.training_dir, exist_ok=True)
        
        self.thresholds = {
            "normal": 0.05,
            "warning": 0.15,
            "critical": 0.25
        }
        print(f"Base directory: {self.base_dir}")
        print(f"Model directory: {self.model_dir}")
        print(f"Training directory: {self.training_dir}")
        
    def load_training_data(self):
        """Charge les données d'entraînement préparées"""
        data_path = os.path.join(self.training_dir, "ai_train_data.npy")
        feature_path = os.path.join(self.training_dir, "ai_feature_names.npy")
        
        print(f"Recherche des données: {data_path}")
        
        if not os.path.exists(data_path):
            # Lister les fichiers pour debug
            parent_dir = os.path.dirname(data_path)
            if os.path.exists(parent_dir):
                print(f"Fichiers dans {parent_dir}:")
                for f in os.listdir(parent_dir):
                    print(f"   - {f}")
            raise FileNotFoundError(f"Données d'entraînement introuvables: {data_path}")
        
        print("Chargement des données d'entraînement...")
        X_train = np.load(data_path)
        feature_names = np.load(feature_path, allow_pickle=True)
        
        print(f"Données chargées: {X_train.shape[0]} échantillons, {X_train.shape[1]} features")
        print(f"Features: {list(feature_names)}")
        
        return X_train, feature_names
    
    def create_autoencoder(self, input_dim):
        """
        Crée un autoencodeur léger pour ESP32
        Architecture: 10 → 8 → 4 → 2 → 4 → 8 → 10
        """
        print(f" Création de l'autoencodeur (input_dim: {input_dim})")
        
        # Encoder
        encoder = keras.Sequential([
            layers.Dense(8, activation='relu', input_shape=(input_dim,)),
            layers.Dense(4, activation='relu'),
            layers.Dense(2, activation='relu', name='bottleneck')  # Compression forte
        ], name='encoder')
        
        # Decoder
        decoder = keras.Sequential([
            layers.Dense(4, activation='relu', input_shape=(2,)),
            layers.Dense(8, activation='relu'),
            layers.Dense(input_dim, activation='sigmoid')  # Sigmoid car données normalisées [0,1]
        ], name='decoder')
        
        # Autoencodeur complet
        autoencoder = keras.Sequential([
            encoder,
            decoder
        ], name='autoencoder')
        
        # Compilation
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',  # Mean Squared Error pour reconstruction
            metrics=['mae']
        )
        
        # Affichage de l'architecture
        autoencoder.summary()
        
        return autoencoder, encoder, decoder
    
    def train_model(self, X_train, validation_split=0.2, epochs=100, batch_size=32):
        """Entraîne le modèle d'autoencodeur"""
        print("\n Début de l'entraînement...")
        print(f"   - Échantillons: {X_train.shape[0]}")
        print(f"   - Features: {X_train.shape[1]}")
        print(f"   - Validation split: {validation_split}")
        print(f"   - Epochs: {epochs}")
        print(f"   - Batch size: {batch_size}")
        
        # Création du modèle
        self.model, self.encoder, self.decoder = self.create_autoencoder(X_train.shape[1])
        
        # Callbacks pour un meilleur entraînement
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Entraînement
        self.history = self.model.fit(
            X_train, X_train,  # Autoencodeur: input = output
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        print("Entraînement terminé!")
        return self.history
    
    def calculate_anomaly_threshold(self, X_train):
        """Calcule le seuil d'anomalie basé sur l'erreur de reconstruction"""
        print("\n Calcul du seuil d'anomalie...")
        
        # Prédictions sur les données d'entraînement
        predictions = self.model.predict(X_train, verbose=0)
        
        # Erreur de reconstruction (MSE)
        reconstruction_errors = np.mean(np.square(X_train - predictions), axis=1)
        
        # Statistiques des erreurs
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        
        # Seuils basés sur la distribution normale
        self.thresholds = {
            "normal": mean_error + std_error,      # 1 sigma
            "warning": mean_error + 2 * std_error, # 2 sigma
            "critical": mean_error + 3 * std_error # 3 sigma
        }
        
        print(f"Statistiques des erreurs de reconstruction:")
        print(f"   - Moyenne: {mean_error:.6f}")
        print(f"   - Écart-type: {std_error:.6f}")
        print(f"   - Min: {np.min(reconstruction_errors):.6f}")
        print(f"   - Max: {np.max(reconstruction_errors):.6f}")
        print(f"Seuils d'anomalie:")
        print(f"   - Normal: < {self.thresholds['normal']:.6f}")
        print(f"   - Warning: {self.thresholds['normal']:.6f} - {self.thresholds['warning']:.6f}")
        print(f"   - Critical: > {self.thresholds['warning']:.6f}")
        
        return reconstruction_errors
    
    def save_model(self, feature_names):
        """Sauvegarde le modèle et les artefacts dans le dossier dédié"""
        print(f"Dossier modèles: {self.model_dir}")
        
        # 1. Sauvegarde du modèle complet Keras
        model_path = os.path.join(self.model_dir, "ai_autoencoder.h5")
        self.model.save(model_path)
        print(f"Modèle Keras sauvegardé: {model_path}")
        
        # 2. Sauvegarde de l'encodeur (utile pour l'extraction de features)
        encoder_path = os.path.join(self.model_dir, "ai_encoder.h5")
        self.encoder.save(encoder_path)
        print(f"Encodeur sauvegardé: {encoder_path}")
        
        # 3. Conversion en TensorFlow Lite
        tflite_path = os.path.join(self.model_dir, "ai_autoencoder.tflite")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Optimisations pour ESP32
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]  # Réduction précision
        
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Vérification de la taille du modèle
        model_size_kb = len(tflite_model) / 1024
        print(f"Modèle TFLite sauvegardé: {tflite_path}")
        print(f"Taille du modèle TFLite: {model_size_kb:.1f} KB")
        
        # 4. Sauvegarde des seuils
        thresholds_path = os.path.join(self.model_dir, "ai_thresholds.json")
        thresholds_data = {
            "thresholds": {k: float(v) for k, v in self.thresholds.items()},
            "calculation_date": datetime.now().isoformat(),
            "model_size_kb": float(model_size_kb),
            "features": list(feature_names)
        }
        
        with open(thresholds_path, 'w') as f:
            json.dump(thresholds_data, f, indent=2)
        print(f"Seuils sauvegardés: {thresholds_path}")
        
        # 5. Sauvegarde des informations du modèle
        model_info_path = os.path.join(self.model_dir, "ai_model_info.json")
        model_info = {
            "model_name": "EPS_Guardian_Autoencoder",
            "version": "1.0.0",
            "creation_date": datetime.now().isoformat(),
            "architecture": {
                "input_dim": self.model.input_shape[1],
                "encoder_layers": [8, 4, 2],
                "decoder_layers": [4, 8, self.model.input_shape[1]],
                "total_parameters": self.model.count_params()
            },
            "training_config": {
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "loss_function": "mse",
                "metrics": ["mae"]
            },
            "features_used": list(feature_names),
            "model_size_kb": float(model_size_kb),
            "esp32_compatible": model_size_kb <= 50
        }
        
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"Informations modèle sauvegardées: {model_info_path}")
        
        # 6. Sauvegarde du résumé du modèle (CORRIGÉ)
        summary_path = os.path.join(self.model_dir, "ai_model_summary.txt")
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:  # ← ENCODING UTF-8 AJOUTÉ
                self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            print(f"Résumé modèle sauvegardé: {summary_path}")
        except Exception as e:
            print(f"Note: Résumé non sauvegardé (erreur encodage: {e})")
            # Créer un résumé simplifié
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"Autoencodeur EPS Guardian\n")
                f.write(f"Input: {self.model.input_shape[1]} features\n")
                f.write(f"Architecture: 18 -> 8 -> 4 -> 2 -> 4 -> 8 -> 18\n")
                f.write(f"Paramètres: {self.model.count_params()}\n")
                f.write(f"Taille TFLite: {model_size_kb:.1f} KB\n")
            print(f"Résumé simplifié sauvegardé: {summary_path}")
        
        return model_path, tflite_path
    
    def plot_training_history(self):
        """Génère des graphiques de l'entraînement"""
        if self.history is None:
            print("Aucun historique d'entraînement disponible")
            return
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Loss
            ax1.plot(self.history.history['loss'], label='Train Loss')
            ax1.plot(self.history.history['val_loss'], label='Val Loss')
            ax1.set_title('Model Loss')
            ax1.set_ylabel('Loss')
            ax1.set_xlabel('Epoch')
            ax1.legend()
            ax1.grid(True)
            
            # MAE
            ax2.plot(self.history.history['mae'], label='Train MAE')
            ax2.plot(self.history.history['val_mae'], label='Val MAE')
            ax2.set_title('Model MAE')
            ax2.set_ylabel('MAE')
            ax2.set_xlabel('Epoch')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Sauvegarde
            plot_path = os.path.join(self.model_dir, "ai_training_history.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Graphiques d'entraînement sauvegardés: {plot_path}")
        except Exception as e:
            print(f"Erreur lors de la création des graphiques: {e}")
    
    def evaluate_model_size(self):
        """Évalue si le modèle est adapté pour ESP32"""
        tflite_path = os.path.join(self.model_dir, "ai_autoencoder.tflite")
        
        if not os.path.exists(tflite_path):
            print("Modèle TFLite non trouvé pour l'évaluation de taille")
            return False
        
        model_size_kb = os.path.getsize(tflite_path) / 1024
        target_size_kb = 50  # Objectif ESP32
        
        print(f"\n ÉVALUATION TAILLE MODÈLE:")
        print(f"   - Taille actuelle: {model_size_kb:.1f} KB")
        print(f"   - Objectif ESP32: {target_size_kb} KB")
        
        if model_size_kb <= target_size_kb:
            print("  MODÈLE ADAPTÉ POUR ESP32! ")
            return True
        else:
            print(f" Modèle trop grand: {model_size_kb - target_size_kb:.1f} KB au-dessus de la cible")
            print("Suggestions: Réduire l'architecture ou utiliser l'optimisation INT8")
            return False
    
    def run(self, epochs=100, batch_size=32):
        """Exécute l'entraînement complet"""
        print("Démarrage de l'entraînement du modèle IA")
        print("=" * 60)
        
        try:
            # 1. Chargement des données
            X_train, feature_names = self.load_training_data()
            
            # 2. Entraînement du modèle
            history = self.train_model(X_train, epochs=epochs, batch_size=batch_size)
            
            # 3. Calcul des seuils d'anomalie
            reconstruction_errors = self.calculate_anomaly_threshold(X_train)
            
            # 4. Sauvegarde des modèles
            model_path, tflite_path = self.save_model(feature_names)
            
            # 5. Visualisation
            self.plot_training_history()
            
            # 6. Évaluation taille
            esp32_compatible = self.evaluate_model_size()
            
            print("\n Entraînement IA terminé avec succès!")
            
            # Rapport final
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            
            print(f"\n RAPPORT FINAL:")
            print(f"   - Loss finale: {final_loss:.6f}")
            print(f"   - Validation loss: {final_val_loss:.6f}")
            print(f"   - Features utilisées: {len(feature_names)}")
            print(f"   - Modèle sauvegardé dans: {self.model_dir}")
            print(f"   - Compatible ESP32: {'OUI' if esp32_compatible else 'NON'}")
            
            return True
            
        except Exception as e:
            print(f" Erreur lors de l'entraînement: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Fonction principale"""
    trainer = AIModelTrainer()
    
    # Configuration d'entraînement (adaptable)
    success = trainer.run(
        epochs=100,        # Nombre d'époques
        batch_size=32      # Taille de batch
    )
    
    if success:
        print("\n L'entraînement IA est terminé avec succès!")
        print(" Tous les fichiers sont organisés dans: data/ai_models/model_simple/")
        print(" Vous pouvez maintenant tester l'inférence avec ai_model_inference.py")
    else:
        print("\n L'entraînement IA a échoué.")

if __name__ == "__main__":
    main()
