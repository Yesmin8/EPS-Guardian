#!/usr/bin/env python3
"""
Vérification de la base de données améliorée avec normalisation
Version corrigée avec les bons chemins pour votre structure
"""

import os
import numpy as np
import pandas as pd
import json
import joblib

# Chemins corrigés pour votre structure
PROJECT_ROOT = r"D:\Challenge AESS&IES"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "ai_training_base")
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "dataset")
ANALYSE_DIR = os.path.join(PROJECT_ROOT, "data", "analyse")

def verify_improved_datasets():
    """Vérifie l'intégrité des datasets générés avec les améliorations"""
    print("VÉRIFICATION DES DATASETS AMÉLIORÉS")
    print("=" * 60)
    
    # Définition des fichiers avec leurs emplacements CORRECTS
    files_to_check = {
        os.path.join(DATASET_DIR, 'pro_eps_extended.csv'): 'CSV étendu (dataset/)',
        os.path.join(OUTPUT_DIR, 'ai_sequence_data.npy'): 'Séquences temporelles normalisées (ai_training_base/)', 
        os.path.join(OUTPUT_DIR, 'ai_sequence_labels.npy'): 'Labels des séquences (ai_training_base/)',
        os.path.join(OUTPUT_DIR, 'ai_sequence_features.npy'): 'Features dérivées normalisées (ai_training_base/)',
        os.path.join(OUTPUT_DIR, 'ai_sequence_scaler.pkl'): 'Scaler pour séquences (ai_training_base/)',
        os.path.join(OUTPUT_DIR, 'ai_features_scaler.pkl'): 'Scaler pour features (ai_training_base/)',
        os.path.join(ANALYSE_DIR, 'extended_summary_stats.csv'): 'Statistiques (analyse/)',
        os.path.join(ANALYSE_DIR, 'dataset_config.json'): 'Configuration avec traçabilité (analyse/)'
    }
    
    for filepath, description in files_to_check.items():
        if os.path.exists(filepath):
            print(f"{description}: {os.path.basename(filepath)}")
            
            # Informations supplémentaires selon le type de fichier
            try:
                if filepath.endswith('.npy'):
                    data = np.load(filepath)
                    print(f"   Shape: {data.shape} | dtype: {data.dtype}")
                elif filepath.endswith('.csv'):
                    df = pd.read_csv(filepath)
                    print(f"   Lignes: {len(df)} | Colonnes: {len(df.columns)}")
                elif filepath.endswith('.json'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    print(f"   Normalisation: {config.get('normalization_applied', 'N/A')}")
                    print(f"   Séquences: {config.get('dataset_shape', {}).get('sequences', 'N/A')}")
                elif filepath.endswith('.pkl'):
                    scaler = joblib.load(filepath)
                    print(f"   Type: {type(scaler).__name__} | Features: {len(scaler.mean_)}")
                    
            except Exception as e:
                print(f"   Erreur lecture: {e}")
        else:
            print(f"{description}: FICHIER MANQUANT")
    
    print("=" * 60)
    
    # Vérification de cohérence et normalisation
    try:
        sequences_path = os.path.join(OUTPUT_DIR, 'ai_sequence_data.npy')
        labels_path = os.path.join(OUTPUT_DIR, 'ai_sequence_labels.npy')
        features_path = os.path.join(OUTPUT_DIR, 'ai_sequence_features.npy')
        
        if all(os.path.exists(p) for p in [sequences_path, labels_path, features_path]):
            sequences = np.load(sequences_path)
            labels = np.load(labels_path)
            features = np.load(features_path)
            
            print("VÉRIFICATION COHÉRENCE:")
            print(f"  Séquences: {sequences.shape[0]}")
            print(f"  Labels: {labels.shape[0]}")
            print(f"  Features: {features.shape[0]}")
            
            if sequences.shape[0] == labels.shape[0] == features.shape[0]:
                print(" Dimensions cohérentes")
            else:
                print(" Dimensions incohérentes!")
            
            # Vérification de la normalisation
            sequence_mean = np.mean(sequences)
            sequence_std = np.std(sequences)
            print(f"\nVÉRIFICATION NORMALISATION:")
            print(f"  Mean: {sequence_mean:.6f} (attendu ~0.0)")
            print(f"  Std:  {sequence_std:.6f} (attendu ~1.0)")
            
            if abs(sequence_mean) < 0.1 and abs(sequence_std - 1.0) < 0.2:
                print("  Normalisation correcte")
            else:
                print("  Normalisation potentiellement problématique")
                
        else:
            print("Fichiers manquants pour la vérification de cohérence")
            
    except Exception as e:
        print(f"Erreur lors de la vérification: {e}")

if __name__ == "__main__":
    verify_improved_datasets()
