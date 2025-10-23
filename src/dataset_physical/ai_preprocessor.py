#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
import sys

class AIPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.features = None
        self.base_dir = r"D:\Challenge AESS&IES"
        self.data_dir = os.path.join(self.base_dir, "data")
        self.dataset_dir = os.path.join(self.data_dir, "dataset")
        self.training_data_dir = os.path.join(self.data_dir, "training_data")
        self.analyse_dir = os.path.join(self.data_dir, "analyse")
        
    def load_dataset(self):
        data_path = os.path.join(self.dataset_dir, "pro_eps_dataset.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset introuvable: {data_path}")
        print(f"Chargement du dataset: {data_path}")
        df = pd.read_csv(data_path)
        print(f"Dataset chargé: {len(df)} échantillons, {len(df.columns)} colonnes")
        return df
    
    def select_features(self, df):
        base_features = ["V_batt", "I_batt", "T_batt", "V_bus", "I_bus", "V_solar", "I_solar", "SOC", "T_eps"]
        available_features = [f for f in base_features if f in df.columns]
        print(f"Features sélectionnées ({len(available_features)}):")
        for feature in available_features: 
            print(f"  - {feature}")
        return available_features
    
    def prepare_training_data(self, df, features):
        if "anomaly_type" in df.columns:
            normal_data = df[df["anomaly_type"] == "normal"].copy()
            print(f"Données normales pour entraînement: {len(normal_data)} échantillons")
        else:
            normal_data = df.copy()
            print(f"Colonne 'anomaly_type' non trouvée, utilisation de tout le dataset")
        
        normal_data_processed = normal_data.copy()
        
        # Features de base
        final_features = [f for f in features if f in normal_data_processed.columns]
        
        # Features calculées existantes
        calculated_features = ["P_batt", "P_solar", "P_bus", "converter_ratio", 
                              "delta_V_batt", "delta_I_batt", "delta_T_batt",
                              "rolling_std_V_batt", "rolling_mean_V_batt"]
        
        for feature in calculated_features:
            if feature in normal_data_processed.columns and feature not in final_features:
                final_features.append(feature)
        
        print(f"Features finales ({len(final_features)}):")
        for feature in final_features: 
            print(f"  - {feature}")
        
        X = normal_data_processed[final_features]
    
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.interpolate(method='linear', limit_direction='forward')
                
        X_clean = X_clean.bfill().ffill()
        
        if X_clean.isna().any().any():
            X_clean = X_clean.fillna(0)
            
        print(f"Données nettoyées: {X_clean.shape}")
        return X_clean, final_features
    
    def normalize_data(self, X):
        print("\nNormalisation des données...")
        X_scaled = self.scaler.fit_transform(X)
        print(f"Plage des données normalisées: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
        return X_scaled
    
    def save_processed_data(self, X_scaled, feature_names):
        """Sauvegarde les données traitées dans training_data/"""
        os.makedirs(self.training_data_dir, exist_ok=True)
        
        # Sauvegarder les données d'entraînement
        train_data_path = os.path.join(self.training_data_dir, "ai_train_data.npy")
        np.save(train_data_path, X_scaled)
        print(f" Données d'entraînement sauvegardées: {train_data_path}")
        
        # Sauvegarder les noms des features
        feature_path = os.path.join(self.training_data_dir, "ai_feature_names.npy")
        np.save(feature_path, np.array(feature_names))
        print(f" Noms des features sauvegardés: {feature_path}")
        
        # Sauvegarder le scaler
        scaler_path = os.path.join(self.training_data_dir, "ai_scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        print(f" Scaler sauvegardé: {scaler_path}")
        
        # Sauvegarder un résumé CSV dans analyse/ pour vérification
        summary_path = os.path.join(self.analyse_dir, "ai_training_summary.csv")
        summary_df = pd.DataFrame(X_scaled, columns=feature_names)
        summary_df.to_csv(summary_path, index=False)
        print(f" Résumé CSV sauvegardé: {summary_path}")
        
        return train_data_path
    
    def run(self):
        print(" Démarrage du préprocessing IA")
        print("=" * 60)
        try:
            # Afficher la structure des chemins
            print(f"Base directory: {self.base_dir}")
            print(f"Dataset source: {self.dataset_dir}")
            print(f"Training data output: {self.training_data_dir}")
            print(f"Analyse reports: {self.analyse_dir}")
            print("-" * 40)
            
            df = self.load_dataset()
            features = self.select_features(df)
            X, final_features = self.prepare_training_data(df, features)
            X_scaled = self.normalize_data(X)
            output_path = self.save_processed_data(X_scaled, final_features)
            
            print("\n Préprocessing IA terminé avec succès!")
            print(f" Données finales: {X_scaled.shape[0]} échantillons, {X_scaled.shape[1]} features")
            print(f" Fichiers TensorFlow dans: {self.training_data_dir}")
            print(f" Rapport dans: {self.analyse_dir}")
            
            return True
            
        except Exception as e:
            print(f" Erreur lors du préprocessing: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    preprocessor = AIPreprocessor()
    success = preprocessor.run()
    if success:
        print("\n Le préprocessing IA est terminé!")
        print(" Vous pouvez maintenant passer à l'entraînement du modèle TensorFlow.")
        print(f" Données d'entraînement: D:\\Challenge AESS&IES\\data\\training_data\\")
        print(f" Rapport: D:\\Challenge AESS&IES\\data\\analyse\\ai_training_summary.csv")
    else:
        print("\n Le préprocessing IA a échoué.")

if __name__ == "__main__":
    main()
