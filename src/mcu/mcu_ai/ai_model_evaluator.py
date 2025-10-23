#!/usr/bin/env python3
"""
Évaluateur de modèle IA pour EPS GUARDIAN
Évalue les performances du modèle sur le dataset complet avec métriques détaillées
"""
import numpy as np
import pandas as pd
import os
import json
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns
from datetime import datetime

class AIModelEvaluator:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.model_dir = os.path.join(self.base_dir, "data", "ai_models", "model_simple")
        self.training_dir = os.path.join(self.base_dir, "data", "training_data")  # ← NOUVEAU CHEMIN
        self.evaluation_dir = os.path.join(self.base_dir, "data", "mcu", "evaluation")
        os.makedirs(self.evaluation_dir, exist_ok=True)
        
        # Données et artefacts
        self.dataset = None
        self.scaler = None
        self.feature_names = None
        self.thresholds = None
        self.model = None
        
        # Résultats d'évaluation
        self.evaluation_results = {}
        
        print("AIModelEvaluator initialisé")
    
    def load_dataset_and_artifacts(self):
        """Charge le dataset complet et les artefacts IA"""
        try:
            # 1. Charger le dataset original
            dataset_path = os.path.join(self.base_dir, "data", "dataset", "pro_eps_dataset.csv")
            if not os.path.exists(dataset_path):
                print(f" Dataset non trouvé: {dataset_path}")
                return False
                
            self.dataset = pd.read_csv(dataset_path)
            print(f" Dataset chargé: {len(self.dataset)} échantillons")
            
            # 2. Charger les artefacts IA
            # CORRIGÉ : Charger depuis training_data
            scaler_path = os.path.join(self.training_dir, "ai_scaler.pkl")  # ← CHEMIN CORRIGÉ
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print(" Scaler chargé")
            else:
                print(f" Scaler non trouvé: {scaler_path}")
                return False
            
            # CORRIGÉ : Charger depuis training_data  
            features_path = os.path.join(self.training_dir, "ai_feature_names.npy")  # ← CHEMIN CORRIGÉ
            if os.path.exists(features_path):
                self.feature_names = np.load(features_path, allow_pickle=True).tolist()
                print(f"Features chargées: {len(self.feature_names)} features")
            else:
                print(f"Features non trouvées: {features_path}")
                return False
            
            # Charger les seuils depuis model_simple
            thresholds_path = os.path.join(self.model_dir, "ai_thresholds.json")
            if os.path.exists(thresholds_path):
                with open(thresholds_path, 'r') as f:
                    thresholds_data = json.load(f)
                    self.thresholds = thresholds_data["thresholds"]
                print("Seuils chargés")
            else:
                print(f"Seuils non trouvés: {thresholds_path}")
                return False
            
            # 3. Charger le modèle TFLite
            import tensorflow as tf
            tflite_path = os.path.join(self.model_dir, "ai_autoencoder.tflite")
            if os.path.exists(tflite_path):
                self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print("Modèle TFLite chargé")
            else:
                print(f"Modèle TFLite non trouvé: {tflite_path}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Erreur chargement: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def prepare_features(self, df):
        """Prépare les features pour l'évaluation"""
        try:
            # Calculer les features manquantes
            df_processed = df.copy()
            
            # P_batt = V_batt * I_batt
            if "P_batt" not in df.columns and "V_batt" in df.columns and "I_batt" in df.columns:
                df_processed["P_batt"] = df_processed["V_batt"] * df_processed["I_batt"]
            
            # P_solar = V_solar * I_solar
            if "P_solar" not in df.columns and "V_solar" in df.columns and "I_solar" in df.columns:
                df_processed["P_solar"] = df_processed["V_solar"] * df_processed["I_solar"]
            
            # P_bus = V_bus * I_bus
            if "P_bus" not in df.columns and "V_bus" in df.columns and "I_bus" in df.columns:
                df_processed["P_bus"] = df_processed["V_bus"] * df_processed["I_bus"]
            
            # converter_ratio = V_bus / V_solar
            if "converter_ratio" not in df.columns and "V_bus" in df.columns and "V_solar" in df.columns:
                df_processed["converter_ratio"] = df_processed["V_bus"] / df_processed["V_solar"]
                df_processed["converter_ratio"] = df_processed["converter_ratio"].replace([np.inf, -np.inf], 0)
            
            # Deltas (simplifiés pour l'évaluation)
            for col in ["V_batt", "I_batt", "T_batt"]:
                delta_col = f"delta_{col}"
                if delta_col not in df.columns and col in df.columns:
                    df_processed[delta_col] = df_processed[col].diff().fillna(0)
            
            # Rolling statistics (simplifiées)
            if "rolling_std_V_batt" not in df.columns and "V_batt" in df.columns:
                df_processed["rolling_std_V_batt"] = df_processed["V_batt"].rolling(window=5, min_periods=1).std().fillna(0)
            
            if "rolling_mean_V_batt" not in df.columns and "V_batt" in df.columns:
                df_processed["rolling_mean_V_batt"] = df_processed["V_batt"].rolling(window=5, min_periods=1).mean().fillna(0)
            
            # Sélectionner et ordonner les features
            features_data = []
            missing_features = []
            
            for feature_name in self.feature_names:
                if feature_name in df_processed.columns:
                    features_data.append(df_processed[feature_name].values)
                else:
                    # Remplir avec des zéros si feature manquante
                    features_data.append(np.zeros(len(df_processed)))
                    missing_features.append(feature_name)
            
            if missing_features:
                print(f"Features manquantes remplacées par zéros: {missing_features}")
            
            # Créer la matrice de features
            X = np.column_stack(features_data)
            
            # Nettoyer les NaN
            X = np.nan_to_num(X)
            
            print(f" Features préparées: {X.shape}")
            return X, df_processed
            
        except Exception as e:
            print(f" Erreur préparation features: {e}")
            return None, None
    
    def calculate_reconstruction_errors(self, X):
        """Calcule les erreurs de reconstruction pour un dataset"""
        try:
            # Normaliser les données
            X_normalized = self.scaler.transform(X)
            
            errors = []
            batch_size = 32  # Traitement par lots pour plus de rapidité
            
            for i in range(0, len(X_normalized), batch_size):
                batch_end = min(i + batch_size, len(X_normalized))
                batch_data = X_normalized[i:batch_end].astype(np.float32)
                
                # Préparer l'input
                self.interpreter.resize_tensor_input(self.input_details[0]['index'], [len(batch_data), len(self.feature_names)])
                self.interpreter.allocate_tensors()
                
                # Exécuter l'inférence
                self.interpreter.set_tensor(self.input_details[0]['index'], batch_data)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                
                # Calculer l'erreur pour chaque échantillon
                batch_errors = np.mean(np.square(batch_data - output_data), axis=1)
                errors.extend(batch_errors)
            
            return np.array(errors)
            
        except Exception as e:
            print(f" Erreur calcul reconstruction: {e}")
            return None
    
    def evaluate_model_performance(self):
        """Évalue les performances du modèle sur le dataset complet"""
        print("\n ÉVALUATION DES PERFORMANCES DU MODÈLE")
        print("=" * 50)
        
        try:
            # Préparer les features
            X, df_processed = self.prepare_features(self.dataset)
            if X is None:
                return False
            
            # Calculer les erreurs de reconstruction
            print("Calcul des erreurs de reconstruction...")
            reconstruction_errors = self.calculate_reconstruction_errors(X)
            if reconstruction_errors is None:
                return False
            
            # Préparer les labels (si disponibles)
            has_anomaly_labels = "anomaly_type" in self.dataset.columns or "anomaly" in self.dataset.columns
            y_true = None
            
            if has_anomaly_labels:
                if "anomaly_type" in self.dataset.columns:
                    # Convertir les labels en binaire (normal=0, anomalie=1)
                    y_true = (self.dataset["anomaly_type"] != "normal").astype(int).values
                elif "anomaly" in self.dataset.columns:
                    y_true = self.dataset["anomaly"].astype(int).values
                
                print(f"Labels d'anomalie disponibles: {np.sum(y_true)} anomalies sur {len(y_true)} échantillons")
            else:
                print("Aucun label d'anomalie disponible - évaluation non supervisée seulement")
            
            # Calculer les métriques de base
            self.evaluation_results = {
                "evaluation_date": datetime.now().isoformat(),
                "dataset_size": len(self.dataset),
                "features_used": len(self.feature_names),
                "feature_names": self.feature_names,
                "reconstruction_errors": {
                    "mean": float(np.mean(reconstruction_errors)),
                    "std": float(np.std(reconstruction_errors)),
                    "min": float(np.min(reconstruction_errors)),
                    "max": float(np.max(reconstruction_errors)),
                    "median": float(np.median(reconstruction_errors)),
                    "q95": float(np.quantile(reconstruction_errors, 0.95))
                },
                "thresholds": self.thresholds
            }
            
            # Métriques avec labels si disponibles
            if has_anomaly_labels and y_true is not None:
                self._calculate_detection_metrics(reconstruction_errors, y_true)
            
            # Générer les visualisations
            self._generate_evaluation_plots(reconstruction_errors, y_true)
            
            # Sauvegarder les résultats
            self._save_evaluation_results()
            
            return True
            
        except Exception as e:
            print(f" Erreur évaluation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _calculate_detection_metrics(self, errors, y_true):
        """Calcule les métriques de détection d'anomalies"""
        try:
            # Utiliser le seuil "warning" pour la détection
            threshold = self.thresholds["warning"]
            y_pred = (errors >= threshold).astype(int)
            
            # Matrice de confusion
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Métriques de performance
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Courbe ROC
            fpr, tpr, roc_thresholds = roc_curve(y_true, errors)
            roc_auc = auc(fpr, tpr)
            
            # Courbe Precision-Recall
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, errors)
            pr_auc = auc(recall_curve, precision_curve)
            
            self.evaluation_results["detection_metrics"] = {
                "confusion_matrix": {
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_positives": int(tp)
                },
                "performance_metrics": {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1_score),
                    "roc_auc": float(roc_auc),
                    "pr_auc": float(pr_auc)
                },
                "detection_threshold": float(threshold)
            }
            
            print("Métriques de détection calculées:")
            print(f"   - Accuracy: {accuracy:.3f}")
            print(f"   - Precision: {precision:.3f}")
            print(f"   - Recall: {recall:.3f}")
            print(f"   - F1-Score: {f1_score:.3f}")
            print(f"   - AUC-ROC: {roc_auc:.3f}")
            
        except Exception as e:
            print(f" Erreur calcul métriques: {e}")
    
    def _generate_evaluation_plots(self, errors, y_true=None):
        """Génère les graphiques d'évaluation"""
        try:
            # 1. Distribution des erreurs de reconstruction
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
            plt.axvline(self.thresholds["normal"], color='green', linestyle='--', label='Seuil Normal', linewidth=2)
            plt.axvline(self.thresholds["warning"], color='orange', linestyle='--', label='Seuil Warning', linewidth=2)
            plt.axvline(self.thresholds["critical"], color='red', linestyle='--', label='Seuil Critical', linewidth=2)
            plt.xlabel('Erreur de Reconstruction')
            plt.ylabel('Fréquence')
            plt.title('Distribution des Erreurs de Reconstruction')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 2. Erreurs au cours du temps
            plt.subplot(1, 3, 2)
            plt.plot(errors, alpha=0.7, linewidth=0.5)
            plt.axhline(self.thresholds["normal"], color='green', linestyle='--', label='Normal')
            plt.axhline(self.thresholds["warning"], color='orange', linestyle='--', label='Warning')
            plt.axhline(self.thresholds["critical"], color='red', linestyle='--', label='Critical')
            plt.xlabel('Échantillon')
            plt.ylabel('Erreur de Reconstruction')
            plt.title('Erreurs de Reconstruction dans le Temps')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 3. Courbe ROC si labels disponibles
            if y_true is not None:
                plt.subplot(1, 3, 3)
                fpr, tpr, _ = roc_curve(y_true, errors)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Courbe ROC - Détection d\'Anomalies')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(self.evaluation_dir, "evaluation_summary.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f" Graphiques sauvegardés: {plot_path}")
            
            # 4. Matrice de confusion si labels disponibles
            if y_true is not None and "detection_metrics" in self.evaluation_results:
                cm = self.evaluation_results["detection_metrics"]["confusion_matrix"]
                cm_array = np.array([[cm["true_negatives"], cm["false_positives"]],
                                   [cm["false_negatives"], cm["true_positives"]]])
                
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['Normal Prédit', 'Anomalie Prédite'],
                           yticklabels=['Normal Réel', 'Anomalie Réelle'])
                plt.title('Matrice de Confusion - Détection d\'Anomalies')
                plt.tight_layout()
                
                cm_path = os.path.join(self.evaluation_dir, "confusion_matrix.png")
                plt.savefig(cm_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Matrice de confusion sauvegardée: {cm_path}")
                
        except Exception as e:
            print(f" Erreur génération graphiques: {e}")
    
    def _save_evaluation_results(self):
        """Sauvegarde les résultats d'évaluation"""
        try:
            results_path = os.path.join(self.evaluation_dir, "ai_evaluation_results.json")
            with open(results_path, 'w') as f:
                json.dump(self.evaluation_results, f, indent=2)
            
            print(f" Résultats d'évaluation sauvegardés: {results_path}")
            
            # Résumé console
            print("\n RAPPORT D'ÉVALUATION:")
            print("=" * 50)
            errors = self.evaluation_results["reconstruction_errors"]
            print(f"Erreurs de reconstruction:")
            print(f"   - Moyenne: {errors['mean']:.6f}")
            print(f"   - Écart-type: {errors['std']:.6f}")
            print(f"   - Médiane: {errors['median']:.6f}")
            print(f"   - 95e percentile: {errors['q95']:.6f}")
            print(f"   - Plage: [{errors['min']:.6f}, {errors['max']:.6f}]")
            
            print(f"\nSeuils de détection:")
            print(f"   - Normal: < {self.thresholds['normal']:.6f}")
            print(f"   - Warning: {self.thresholds['normal']:.6f} - {self.thresholds['warning']:.6f}")
            print(f"   - Critical: > {self.thresholds['warning']:.6f}")
            
            if "detection_metrics" in self.evaluation_results:
                metrics = self.evaluation_results["detection_metrics"]["performance_metrics"]
                print(f"\n Métriques de détection:")
                print(f"   - Accuracy: {metrics['accuracy']:.3f}")
                print(f"   - Precision: {metrics['precision']:.3f}")
                print(f"   - Recall: {metrics['recall']:.3f}")
                print(f"   - F1-Score: {metrics['f1_score']:.3f}")
                print(f"   - AUC-ROC: {metrics['roc_auc']:.3f}")
                
        except Exception as e:
            print(f"Erreur sauvegarde résultats: {e}")
    
    def run_evaluation(self):
        """Exécute l'évaluation complète"""
        print("Démarrage de l'évaluation du modèle IA")
        print("=" * 50)
        
        if self.load_dataset_and_artifacts():
            success = self.evaluate_model_performance()
            
            if success:
                print("\n Évaluation terminée avec succès!")
                print(f"Résultats disponibles dans: {self.evaluation_dir}")
            else:
                print("\n Évaluation échouée")
        else:
            print("Impossible de charger les données et artefacts")

def main():
    """Fonction principale"""
    evaluator = AIModelEvaluator()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
