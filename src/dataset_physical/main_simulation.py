#!/usr/bin/env python3
import os
import sys
import logging
import time
import argparse
from datetime import datetime
import pandas as pd

# === CONFIGURATION CORRIGÉE DES CHEMINS ===
BASE_DIR = r"D:\Challenge AESS&IES"
DATA_DIR = os.path.join(BASE_DIR, "data")
ANALYSE_DIR = os.path.join(DATA_DIR, "analyse")
VISUALIZATIONS_DIR = os.path.join(ANALYSE_DIR, "visualizations")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
DATA_TRAIN_DIR = os.path.join(DATA_DIR, "data_train")
LOGS_DIR = os.path.join(DATA_DIR, "logs")

# SUPPRIMER TOUTES LES LIGNES os.makedirs() - VOUS AVEZ DÉJÀ LES DOSSIERS

# Configuration du logging - UNIQUEMENT CONSOLE
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # SUPPRIMER FileHandler
)

logger = logging.getLogger(__name__)

# Ajouter le chemin actuel pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def parse_arguments():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description='Simulation EPS Guardian - Surveillance energetique CubeSat')
    parser.add_argument('--fast', action='store_true', help='Mode rapide avec moins d echantillons')
    parser.add_argument('--debug', action='store_true', help='Mode debug avec logging detaille')
    parser.add_argument('--seed', type=int, default=42, help='Seed RNG pour reproductibilite')
    return parser.parse_args()

def print_banner():
    """Affiche une banniere simplifiee"""
    banner = """
=================================================================
        EPS GUARDIAN - SYSTEME DE SURVEILLANCE ENERGETIQUE
                   POUR CUBESAT AESS/IES
=================================================================
    """
    print(banner)

def check_environment():
    """Verifie l'environnement et les dependances"""
    logger.info("Verification de l'environnement...")
    
    # VERIFICATION SEULEMENT - DOSSIERS CRITIQUES UNIQUEMENT
    required_dirs = {
        "Data": DATA_DIR,
        "Analyse": ANALYSE_DIR,
        "Visualizations": VISUALIZATIONS_DIR,
        "Dataset": DATASET_DIR,
    }
    
    missing_dirs = []
    for name, dir_path in required_dirs.items():
        if not os.path.exists(dir_path):
            missing_dirs.append(f"{name}: {dir_path}")
    
    if missing_dirs:
        logger.error(f"Dossiers manquants: {missing_dirs}")
        return False
    
    # Verification modules
    required_modules = ['numpy', 'pandas', 'matplotlib', 'seaborn', 'openpyxl']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"Modules manquants: {missing_modules}")
        return False
    
    logger.info("Environnement verifie")
    return True


def phase_generation(args):
    """Phase 1: Generation des donnees EPS"""
    print("\n" + "="*70)
    print("PHASE 1: GENERATION DU DATASET EPS")
    print("="*70)
    
    try:
        # Import direct pour éviter les problèmes de chemin
        from pro_eps_data_generator import ProEPSSensorDataGenerator
        
        logger.info("Initialisation du generateur...")
        generator = ProEPSSensorDataGenerator(random_seed=args.seed)
        
        # Configuration de la generation
        if args.fast:
            config = {
                'num_normal': 500,
                'num_anomalies': 100,
                'duration_hours': 24,
                'low_res': True
            }
            print("Mode rapide active")
        else:
            config = {
                'num_normal': 5000,
                'num_anomalies': 1000,
                'duration_hours': 24,
                'low_res': False
            }
        
        print(f"Configuration generation:")
        print(f"   Echantillons normaux: {config['num_normal']}")
        print(f"   Anomalies: {config['num_anomalies']}")
        print(f"   Seed RNG: {args.seed}")
        
        logger.info(f"Debut generation: {config}")
        
        # Generation du dataset
        start_time = time.time()
        df = generator.generate_dataset(**config)
        generation_time = time.time() - start_time
    
        print(f"Generation terminee en {generation_time:.1f}s")
        print(f"Echantillons generes: {len(df)}")
        
        # Calcul des features derivees
        df = generator.calculate_derived_features(df)
        
        # Sauvegarde du dataset
        metadata = generator.save_dataset(df, "pro_eps_dataset.csv")
        
        if metadata:
            print(f"Dataset sauvegarde: {os.path.join(DATASET_DIR, 'pro_eps_dataset.csv')}")
            
            # Statistiques de generation
            print(f"\nStatistiques generation:")
            print(f"   Total: {metadata['dataset_info']['total_samples']} echantillons")
            print(f"   Normaux: {metadata['dataset_info']['normal_samples']}")
            print(f"   Anomalies: {metadata['dataset_info']['anomaly_samples']}")
            print(f"   Types d'anomalies: {len(metadata['anomaly_distribution'])}")
        
        return df
        
    except Exception as e:
        logger.error(f"Erreur generation: {e}")
        print(f"Erreur generation: {e}")
        return None

def phase_analysis(df, args):
    """Phase 2: Analyse complete des donnees"""
    print("\n" + "="*70)
    print("PHASE 2: ANALYSE COMPLETE")
    print("="*70)
    
    try:
        from pro_eps_analyzer import ProEPSAnalyzer
        
        logger.info("Initialisation de l'analyseur...")
        analyzer = ProEPSAnalyzer()
        
        print("Debut de l'analyse...")
        start_time = time.time()
        
        # Analyse complete
        analyzer.analyze_dataset("pro_eps_dataset.csv")
        
        analysis_time = time.time() - start_time
        print(f"Analyse terminee en {analysis_time:.1f}s")
        
        # Verification des fichiers generes
        output_files = [
            'eps_main_dashboard_realistic.png',
            'eps_timeseries_detailed_realistic.png', 
            'eps_anomaly_analysis_realistic.png',
            'eps_distributions_realistic.png'
        ]
        
        json_files = [
            'eps_obc_summary_realistic.json'
        ]
        
        excel_files = [
            'eps_summary_stats.xlsx'
        ]
        
        print(f"\nFichiers generes dans {VISUALIZATIONS_DIR}:")
        for file in output_files:
            file_path = os.path.join(VISUALIZATIONS_DIR, file)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024
                print(f"   {file} ({file_size:.1f} KB)")
        
        print(f"\nFichiers generes dans {ANALYSE_DIR}:")
        for file in json_files + excel_files:
            file_path = os.path.join(ANALYSE_DIR, file)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024
                print(f"   {file} ({file_size:.1f} KB)")
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur analyse: {e}")
        print(f"Erreur analyse: {e}")
        return False

def phase_simulation_mcu_obc():
    """Phase 3: Simulation de l'architecture MCU + OBC"""
    print("\n" + "="*70)
    print("PHASE 3: SIMULATION ARCHITECTURE MCU + OBC")
    print("="*70)
    
    try:
        print("Simulation MCU (IA Simple):")
        print("   Surveillance temps-reel")
        print("   Detection anomalies deterministes")
        print("   Actions immediates protection")
        
        print("\nSimulation OBC (IA Complexe):")
        print("   Analyse approfondie patterns")
        print("   Optimisation strategique")
        print("   Prise de decision long terme")
        
        print("\nWorkflow hybride:")
        print("   1. MCU detection -> alerte immediate")
        print("   2. MCU verification coherence")
        print("   3. MCU alerte OBC -> analyse causes")
        print("   4. OBC optimisation -> nouveaux parametres")
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur simulation MCU/OBC: {e}")
        return False

def generate_summary_table(df, simulation_time, args):
    """Genere un tableau de resume CSV"""
    try:
        from pro_eps_analyzer import ProEPSAnalyzer
        
        # Calcul des metriques
        analyzer = ProEPSAnalyzer()
        battery_health = analyzer._calculate_battery_health(df)
        system_stability = analyzer._calculate_system_stability(df)
        solar_efficiency = analyzer._calculate_solar_efficiency(df)
        physical_consistency = analyzer.check_physical_consistency(df)
        
        # Creation du tableau de resume
        summary_data = {
            'simulation_date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'simulation_mode': ['FAST' if args.fast else 'FULL'],
            'total_samples': [len(df)],
            'normal_samples': [len(df[df['anomaly_type'] == 'normal'])],
            'anomaly_samples': [len(df[df['anomaly_type'] != 'normal'])],
            'critical_anomalies': [len(df[df['anomaly_type'].isin(['batt_overheat', 'batt_undervoltage', 'batt_overcurrent'])])],
            'battery_health_score': [round(battery_health, 1)],
            'system_stability_score': [round(system_stability, 1)],
            'solar_efficiency_score': [round(solar_efficiency, 3)],
            'physical_consistency': ['VALIDATED' if physical_consistency else 'FAILED'],
            'simulation_time_seconds': [round(simulation_time, 1)],
            'data_quality': ['HIGH' if len(df) > 1000 else 'MEDIUM'],
            'random_seed': [args.seed]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Sauvegarde en CSV
        summary_file = os.path.join(ANALYSE_DIR, "simulation_summary_table.csv")
        summary_df.to_csv(summary_file, index=False)
        
        print(f"Tableau de resume genere: {summary_file}")
        return summary_df
        
    except Exception as e:
        logger.error(f"Erreur generation tableau resume: {e}")
        return None

def generate_final_report(df, simulation_time, args):
    """Genere un rapport final de simulation"""
    print("\n" + "="*70)
    print("RAPPORT FINAL DE SIMULATION")
    print("="*70)
    
    import json
    
    try:
        # Chargement du resume OBC
        summary_path = os.path.join(ANALYSE_DIR, 'eps_obc_summary_realistic.json')
        
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            print(f"\nSynthese performances:")
            print(f"   Sante batterie: {summary['performance_metrics']['battery_health_score']}%")
            print(f"   Stabilite systeme: {summary['performance_metrics']['system_stability_score']}%")
            print(f"   Efficacite solaire: {summary['performance_metrics']['solar_efficiency_score']:.1%}")
            print(f"   Coherence physique: {summary['system_overview']['physical_consistency']}")
            
            print(f"\nRapport anomalies:")
            print(f"   Total anomalies: {summary['anomaly_report']['total_anomalies']}")
            print(f"   Anomalies critiques: {summary['anomaly_report']['critical_anomalies']}")
        
        # Generation du tableau de resume
        summary_table = generate_summary_table(df, simulation_time, args)
        
        # Verification des fichiers
        print(f"\nFichiers generes:")
        files_to_check = {
            'Dataset principal': os.path.join(DATASET_DIR, 'pro_eps_dataset.csv'),
            'Metadonnees': os.path.join(DATASET_DIR, 'pro_eps_dataset_metadata.json'),
            'Dashboard': os.path.join(VISUALIZATIONS_DIR, 'eps_main_dashboard_realistic.png'),
            'Series temporelles': os.path.join(VISUALIZATIONS_DIR, 'eps_timeseries_detailed_realistic.png'),
            'Analyse anomalies': os.path.join(VISUALIZATIONS_DIR, 'eps_anomaly_analysis_realistic.png'),
            'Distributions': os.path.join(VISUALIZATIONS_DIR, 'eps_distributions_realistic.png'),
            'Rapport OBC': os.path.join(ANALYSE_DIR, 'eps_obc_summary_realistic.json'),
            'Rapport Excel': os.path.join(ANALYSE_DIR, 'eps_summary_stats.xlsx'),
            'Tableau resume': os.path.join(ANALYSE_DIR, 'simulation_summary_table.csv')
        }
        
        for description, file_path in files_to_check.items():
            if os.path.exists(file_path):
                size_kb = os.path.getsize(file_path) / 1024
                filename = os.path.basename(file_path)
                print(f"   {description}: {filename} ({size_kb:.1f} KB)")
                
    except Exception as e:
        print(f"Erreur generation rapport: {e}")

def main():
    """Fonction principale orchestrant la simulation"""
    
    # Parse des arguments
    args = parse_arguments()
    
    # Configuration logging debug si demande
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Mode DEBUG active")
    
    # Affichage banniere
    print_banner()
    
    # Demarrage chrono
    start_time = time.time()
    simulation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\nDebut simulation: {simulation_date}")
    print(f"Repertoire data: {DATA_DIR}")
    print(f"Seed RNG: {args.seed}")
    
    if args.fast:
        print("Mode rapide active")
    
    # Verification environnement
    if not check_environment():
        print("Arret simulation - environnement non conforme")
        return 1
    
    try:
        # Phase 1: Generation des donnees
        df = phase_generation(args)
        if df is None:
            return 1
        
        # Phase 2: Analyse complete
        if not phase_analysis(df, args):
            return 1
        
        # Phase 3: Simulation architecture
        if not phase_simulation_mcu_obc():
            return 1
        
        # Calcul du temps total
        total_time = time.time() - start_time
        
        # Rapport final
        generate_final_report(df, total_time, args)
        
        print(f"\nSIMULATION TERMINEE AVEC SUCCES!")
        print(f"Temps total: {total_time:.1f}s")
        print(f"Resultats dans: {DATA_DIR}")
        print(f"  - Dataset: {DATASET_DIR}")
        print(f"  - Visualisations: {VISUALIZATIONS_DIR}")
        print(f"  - Analyse: {ANALYSE_DIR}")
        print(f"  - Logs: {LOGS_DIR}")
        
        logger.info(f"Simulation terminee avec succes en {total_time:.1f}s")
        
        # Force le flush des logs pour Windows
        for handler in logging.getLogger().handlers:
            handler.flush()
        
        return 0
        
    except KeyboardInterrupt:
        print("\nSimulation interrompue par l'utilisateur")
        logger.info("Simulation interrompue par l'utilisateur")
        return 1
        
    except Exception as e:
        print(f"\nErreur critique: {e}")
        logger.error(f"Erreur critique: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
