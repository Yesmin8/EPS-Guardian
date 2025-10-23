#!/usr/bin/env python3
"""
DIAGNOSTIC DE LA NORMALISATION - Investigation du problème
"""

import os
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = r"D:\Challenge AESS&IES"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "ai_training_base")

def diagnose_normalization_issue():
    """Investigue le problème de normalisation trop parfaite"""
    print(" DIAGNOSTIC DU PROBLÈME DE NORMALISATION")
    print("=" * 60)
    
    try:
        # Charger les données
        sequences = np.load(os.path.join(OUTPUT_DIR, 'ai_sequence_data.npy'))
        features = np.load(os.path.join(OUTPUT_DIR, 'ai_sequence_features.npy'))
        scaler_seq = joblib.load(os.path.join(OUTPUT_DIR, 'ai_sequence_scaler.pkl'))
        
        print("1. ANALYSE DES SÉQUENCES ORIGINALES (avant normalisation):")
        print("=" * 50)
        
        # Vérifier si on a les données originales
        original_data_path = os.path.join(PROJECT_ROOT, "data", "dataset", "pro_eps_dataset.csv")
        if os.path.exists(original_data_path):
            df_original = pd.read_csv(original_data_path)
            print(f"Données originales - Shape: {df_original.shape}")
            print("\nStatistiques des données originales:")
            for col in ['V_batt', 'I_batt', 'T_batt', 'V_bus', 'I_bus', 'V_solar', 'I_solar']:
                if col in df_original.columns:
                    print(f"  {col}: mean={df_original[col].mean():.3f}, std={df_original[col].std():.3f}")
        
        print("\n2. ANALYSE DES SÉQUENCES NORMALISÉES:")
        print("=" * 50)
        
        # Analyser les séquences normalisées
        print(f"Shape des séquences: {sequences.shape}")
        
        # Vérifier par canal
        print("\nPar canal (moyenne sur toutes les séquences):")
        channels = ['V_batt', 'I_batt', 'T_batt', 'V_bus', 'I_bus', 'V_solar', 'I_solar']
        for i, channel in enumerate(channels):
            channel_data = sequences[:, :, i]
            print(f"  {channel}: mean={np.mean(channel_data):.6f}, std={np.std(channel_data):.6f}")
        
        print("\n3. VÉRIFICATION DES VALEURS EXTRAORDINAIRES:")
        print("=" * 50)
        
        # Vérifier les min/max
        print(f"Min global: {np.min(sequences):.6f}")
        print(f"Max global: {np.max(sequences):.6f}")
        
        # Vérifier les valeurs uniques
        unique_vals = np.unique(sequences)
        print(f"Nombre de valeurs uniques: {len(unique_vals)}")
        if len(unique_vals) < 20:
            print(f"Valeurs uniques: {unique_vals}")
        
        print("\n4. TEST DE RÉALISME:")
        print("=" * 50)
        
        # Prendre un échantillon aléatoire
        sample_seq = sequences[np.random.randint(0, len(sequences))]
        print("Séquence échantillon (premières 10 valeurs du premier canal):")
        print(sample_seq[:10, 0])
        
        # Vérifier la variance temporelle
        temporal_variance = np.var(sequences, axis=1)  # Variance le long du temps
        avg_temporal_var = np.mean(temporal_variance)
        print(f"\nVariance temporelle moyenne: {avg_temporal_var:.6f}")
        
        if avg_temporal_var < 0.01:
            print("  ATTENTION: Variance temporelle très faible!")
            print("   Les séquences pourraient être trop constantes.")
        
        print("\n5. SCALER INFORMATION:")
        print("=" * 50)
        print(f"Scaler mean: {scaler_seq.mean_}")
        print(f"Scaler scale: {scaler_seq.scale_}")
        
        # Vérifier si le scaler a des scales très petits (problème de division)
        if np.any(scaler_seq.scale_ < 1e-6):
            print(" PROBLÈME: Certains scales sont presque nuls!")
            problematic_indices = np.where(scaler_seq.scale_ < 1e-6)[0]
            for idx in problematic_indices:
                print(f"  Canal {channels[idx]}: scale={scaler_seq.scale_[idx]}")
        
    except Exception as e:
        print(f"Erreur lors du diagnostic: {e}")

def check_data_generation_process():
    """Vérifie le processus de génération des données"""
    print("\n VÉRIFICATION DU PROCESSUS DE GÉNÉRATION")
    print("=" * 60)
    
    # Vérifier si on utilise des données simulées
    simulated_path = os.path.join(PROJECT_ROOT, "data", "dataset", "simulated_base_data.csv")
    if os.path.exists(simulated_path):
        print("  UTILISATION DE DONNÉES SIMULÉES DÉTECTÉE")
        df_sim = pd.read_csv(simulated_path)
        print(f"Fichier: {simulated_path}")
        print(f"Shape: {df_sim.shape}")
        
        # Analyser la qualité des données simulées
        print("\nQualité des données simulées:")
        for col in ['V_batt', 'I_batt', 'T_batt']:
            if col in df_sim.columns:
                unique_ratio = df_sim[col].nunique() / len(df_sim)
                print(f"  {col}: {df_sim[col].nunique()} valeurs uniques ({unique_ratio:.1%})")

if __name__ == "__main__":
    diagnose_normalization_issue()
    check_data_generation_process()