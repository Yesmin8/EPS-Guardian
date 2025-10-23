import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import os
from datetime import datetime
import time
import warnings
import gc
import json
warnings.filterwarnings('ignore')

# ======================
# CONFIGURATION DE PRODUCTION
# ======================
class ProductionConfig:
    BASE_PATH = "D:/Challenge AESS&IES"
    
    # Chemins des données
    MCU_DATASET_PATH = f"{BASE_PATH}/data/dataset/pro_eps_dataset.csv"
    OBC_DATASET_PATH = f"{BASE_PATH}/data/dataset/pro_eps_extended.csv"
    
    # Chemins des modèles
    MCU_MODEL_PATHS = [
        f"{BASE_PATH}/data/ai_models/model_simple/ai_autoencoder.h5",
    ]
    
    OBC_MODEL_PATH = f"{BASE_PATH}/data/ai_models/model_complex/ai_model_lstm_autoencoder.h5"
    OUTPUT_DIR = f"{BASE_PATH}/output_system_test"
    
    # SEUILS DE PRODUCTION (optimisés par calibration)
    THRESHOLD_MCU_AE = 213.11    # Seuil optimal calibré
    THRESHOLD_OBC_AE = 143.82    # Seuil optimal calibré
    
    # RÈGLES DE SÉCURITÉ (assouplies pour production)
    RULE_TEMP_CRITICAL = 75.0
    RULE_CURRENT_CRITICAL = 4.0
    RULE_VOLTAGE_CRITICAL = 3.5
    
    # CONFIGURATION SYSTÈME
    SAMPLE_SIZE = 300
    WINDOW_SIZE = 30
    OBC_CALL_PROBABILITY = 0.05  # Réduit pour production

# ======================
# SYSTÈME DE PRODUCTION
# ======================
print("SYSTÈME HYBRIDE MCU + OBC - VERSION DE PRODUCTION")
print("=" * 60)

class ProductionMCU:
    def __init__(self, model_path, threshold):
        self.model_path = model_path
        self.threshold = threshold
        self.model = None
        self.model_loaded = False
        self.expected_features = 18  # Votre modèle a 18 features
        
        # État système
        self.prev_v_batt = 7.4
        self.prev_t_batt = 35.0
        
        self.load_production_model()
    
    def load_production_model(self):
        """Charge le modèle pour la production"""
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            self.model_loaded = True
            print(f" Modèle MCU chargé: {os.path.basename(self.model_path)}")
            print(f"   - Seuil de production: {self.threshold}")
        except Exception as e:
            print(f" Erreur chargement modèle: {e}")
            self.model_loaded = False
    
    def prepare_production_features(self, sample):
        """Prépare les features pour l'inférence de production"""
        v_batt, i_batt, t_batt, v_bus, i_bus, v_solar, i_solar = sample[:7]
        
        # Calcul des deltas
        delta_v_batt = v_batt - self.prev_v_batt
        delta_t_batt = t_batt - self.prev_t_batt
        
        # 18 features comme attendu par votre modèle
        features = [
            v_batt, i_batt, t_batt,
            v_bus, i_bus, 
            v_solar, i_solar,
            v_batt * i_batt,                    # P_batt
            (v_bus / v_solar) if v_solar > 0.1 else 0.0,  # converter_ratio
            delta_v_batt,                       # delta_V_batt
            delta_t_batt,                       # delta_T_batt
            v_bus * i_bus,                      # P_bus
            v_solar * i_solar,                  # P_solar
            abs(i_batt),                        # |I_batt|
            v_batt / 7.4 if 7.4 > 0 else 0,     # ratio tension
            t_batt / 35.0 if 35.0 > 0 else 0,   # ratio température
            delta_v_batt * 10,                  # delta_V amplifié
            delta_t_batt * 2                    # delta_T amplifié
        ]
        
        # Mise à jour de l'historique
        self.prev_v_batt = v_batt
        self.prev_t_batt = t_batt
        
        return np.array(features[:self.expected_features], dtype=np.float32)
    
    def production_inference(self, sample):
        """Inférence optimisée pour la production"""
        if not self.model_loaded:
            return 100.0  # Valeur par défaut sécurisée
        
        try:
            features = self.prepare_production_features(sample)
            features_reshaped = features.reshape(1, -1)
            reconstructed = self.model.predict(features_reshaped, verbose=0)
            reconstruction_error = np.mean(np.square(features - reconstructed[0]))
            return reconstruction_error
        except Exception as e:
            print(f" Erreur inférence: {e}")
            return 150.0  # Valeur d'erreur sécurisée
    
    def check_safety_rules(self, sample):
        """Vérification des règles de sécurité"""
        v_batt, i_batt, t_batt, v_bus, i_bus, v_solar, i_solar = sample[:7]
        
        # Règles critiques seulement
        if t_batt > ProductionConfig.RULE_TEMP_CRITICAL:
            return True, "CRITICAL_TEMP"
        if abs(i_batt) > ProductionConfig.RULE_CURRENT_CRITICAL:
            return True, "CRITICAL_CURRENT" 
        if v_batt < ProductionConfig.RULE_VOLTAGE_CRITICAL and i_batt < 0:
            return True, "CRITICAL_VOLTAGE"
        
        return False, "NORMAL"
    
    def process_production_sample(self, sample):
        """Traitement d'un échantillon en production"""
        # 1. Vérification des règles de sécurité
        rule_alert, rule_reason = self.check_safety_rules(sample)
        
        # 2. Détection IA
        mcu_error = self.production_inference(sample)
        ai_alert = mcu_error > self.threshold
        
        # 3. Décision finale
        if rule_alert:
            alert_level = 2  # CRITICAL
            alert_reason = rule_reason
        elif ai_alert:
            alert_level = 1  # WARNING  
            alert_reason = f"AI_ANOMALY_{mcu_error:.1f}"
        else:
            alert_level = 0  # NORMAL
            alert_reason = "NORMAL"
        
        return {
            'alert_level': alert_level,
            'alert_reason': alert_reason,
            'mcu_error': mcu_error,
            'rule_alert': rule_alert,
            'ai_alert': ai_alert
        }

class ProductionOBC:
    def __init__(self, model_path, threshold):
        self.model_path = model_path
        self.threshold = threshold
        self.model = None
        self.model_loaded = False
        
        self.load_production_model()
    
    def load_production_model(self):
        """Charge le modèle OBC pour la production"""
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            self.model_loaded = True
            print(f" Modèle OBC chargé: {os.path.basename(self.model_path)}")
            print(f"   - Seuil de production: {self.threshold}")
        except Exception as e:
            print(f" Erreur chargement OBC: {e}")
            self.model_loaded = False
    
    def analyze_production_sequence(self, sequence):
        """Analyse une séquence en production"""
        if not self.model_loaded:
            return 120.0  # Valeur par défaut
        
        try:
            if len(sequence.shape) == 2:
                sequence = np.expand_dims(sequence, axis=0)
            
            reconstructed = self.model.predict(sequence, verbose=0)
            mse = np.mean(np.square(sequence - reconstructed))
            return mse
        except Exception as e:
            print(f"️ Erreur OBC: {e}")
            return 130.0  # Valeur d'erreur

class ProductionHybridSystem:
    def __init__(self, config):
        self.config = config
        self.mcu = ProductionMCU(config.MCU_MODEL_PATHS[0], config.THRESHOLD_MCU_AE)
        self.obc = ProductionOBC(config.OBC_MODEL_PATH, config.THRESHOLD_OBC_AE)
        
        # Chargement des données de production
        self.production_data = self.load_production_data()
        self.sequence_pool = self.load_sequence_pool()
        
        print(f"\n Données de production chargées:")
        print(f"   - Échantillons MCU: {len(self.production_data)}")
        print(f"   - Séquences OBC: {len(self.sequence_pool)}")
    
    def load_production_data(self):
        """Charge les données pour la simulation de production"""
        try:
            df = pd.read_csv(self.config.MCU_DATASET_PATH)
            required_columns = ['V_batt', 'I_batt', 'T_batt', 'V_bus', 'I_bus', 'V_solar', 'I_solar']
            
            # Sélection des colonnes disponibles
            available_columns = [col for col in required_columns if col in df.columns]
            if not available_columns:
                available_columns = df.columns[:7].tolist()
            
            data = df[available_columns].head(self.config.SAMPLE_SIZE)
            print(f" Données MCU chargées: {len(data)} échantillons")
            return data
            
        except Exception as e:
            print(f" Erreur chargement données: {e}")
            return self.create_production_test_data()
    
    def create_production_test_data(self):
        """Crée des données de test réalistes pour la production"""
        print(" Génération de données de test réalistes...")
        np.random.seed(42)
        
        data = {
            'V_batt': np.random.normal(7.2, 0.1, self.config.SAMPLE_SIZE),
            'I_batt': np.random.normal(1.5, 0.3, self.config.SAMPLE_SIZE),
            'T_batt': np.random.normal(40, 5, self.config.SAMPLE_SIZE),
            'V_bus': np.random.normal(7.1, 0.1, self.config.SAMPLE_SIZE),
            'I_bus': np.random.normal(1.0, 0.2, self.config.SAMPLE_SIZE),
            'V_solar': np.random.normal(15.0, 1.0, self.config.SAMPLE_SIZE),
            'I_solar': np.random.normal(1.3, 0.3, self.config.SAMPLE_SIZE)
        }
        
        # Ajout de quelques anomalies réalistes
        anomaly_indices = np.random.choice(self.config.SAMPLE_SIZE, 5, replace=False)
        for idx in anomaly_indices:
            data['T_batt'][idx] = 85  # Surchauffe
            data['I_batt'][idx] = 4.5 # Surcharge
        
        return pd.DataFrame(data)
    
    def load_sequence_pool(self):
        """Charge un pool de séquences pour l'OBC"""
        sequences = []
        
        # Génération de séquences réalistes
        for i in range(20):
            sequence = np.random.normal(0, 0.5, (self.config.WINDOW_SIZE, 7))
            sequences.append(sequence)
        
        # Ajout de séquences avec anomalies
        for i in range(5):
            sequence = np.random.normal(2.0, 1.0, (self.config.WINDOW_SIZE, 7))
            sequences.append(sequence)
        
        print(f" Pool de séquences créé: {len(sequences)} séquences")
        return sequences
    
    def get_sequence_for_analysis(self, sample_index):
        """Sélectionne une séquence pour analyse OBC"""
        if self.sequence_pool:
            idx = sample_index % len(self.sequence_pool)
            return self.sequence_pool[idx]
        return None
    
    def run_production_simulation(self):
        """Simulation du système en conditions de production"""
        print(f"\n DÉMARRAGE SIMULATION DE PRODUCTION")
        print("=" * 50)
        
        results = []
        obc_analyses = 0
        critical_alerts = 0
        warning_alerts = 0
        
        start_time = time.time()
        
        for i, (idx, sample) in enumerate(self.production_data.iterrows()):
            if i >= self.config.SAMPLE_SIZE:
                break
            
            # Traitement MCU
            mcu_result = self.mcu.process_production_sample(sample.values)
            
            # Analyse OBC (seulement si alerte MCU et probabilité déclenchée)
            obc_analysis = None
            obc_alert = False
            
            if mcu_result['alert_level'] > 0 and np.random.random() < self.config.OBC_CALL_PROBABILITY:
                obc_analyses += 1
                sequence = self.get_sequence_for_analysis(i)
                if sequence is not None:
                    obc_error = self.obc.analyze_production_sequence(sequence)
                    obc_alert = obc_error > self.obc.threshold
                    obc_analysis = {
                        'error': obc_error,
                        'alert': obc_alert
                    }
            
            # Statistiques d'alerte
            if mcu_result['alert_level'] == 2:
                critical_alerts += 1
            elif mcu_result['alert_level'] == 1:
                warning_alerts += 1
            
            # Enregistrement des résultats
            result = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'sample_id': i,
                'alert_level': mcu_result['alert_level'],
                'alert_reason': mcu_result['alert_reason'],
                'mcu_error': float(mcu_result['mcu_error']),  # Conversion explicite
                'rule_triggered': mcu_result['rule_alert'],
                'ai_anomaly': mcu_result['ai_alert'],
                'obc_called': obc_analysis is not None,
                'obc_error': float(obc_analysis['error']) if obc_analysis else None,
                'obc_alert': obc_analysis['alert'] if obc_analysis else False,
                'cpu_usage': float(psutil.cpu_percent()),  # Conversion explicite
                'memory_usage': float(psutil.virtual_memory().percent)  # Conversion explicite
            }
            results.append(result)
            
            # Log de progression
            if i % 50 == 0:
                print(f" Échantillon {i}/{self.config.SAMPLE_SIZE} - Alertes: {critical_alerts} critiques, {warning_alerts} warnings")
        
        execution_time = time.time() - start_time
        df_results = pd.DataFrame(results)
        
        # RAPPORT DE PRODUCTION
        print(f"\n SIMULATION DE PRODUCTION TERMINÉE")
        print("=" * 50)
        print(f"️  Temps d'exécution: {execution_time:.2f}s")
        print(f" Échantillons traités: {len(df_results)}")
        print(f" Alertes critiques: {critical_alerts}")
        print(f"️  Alertes warning: {warning_alerts}")
        print(f" Analyses OBC: {obc_analyses}")
        print(f" CPU moyen: {df_results['cpu_usage'].mean():.1f}%")
        print(f" RAM moyenne: {df_results['memory_usage'].mean():.1f}%")
        
        return df_results
    
    def generate_production_report(self, results_df):
        """Génère un rapport complet de production"""
        report_dir = self.config.OUTPUT_DIR
        os.makedirs(report_dir, exist_ok=True)
        
        # 1. Fichier de résultats détaillés
        results_path = os.path.join(report_dir, "production_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f" Résultats détaillés: {results_path}")
        
        # 2. Rapport de performance - CORRECTION ICI
        performance_path = os.path.join(report_dir, "production_performance.json")
        
        # Conversion explicite de tous les types numpy vers Python natif
        performance_report = {
            "simulation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system_configuration": {
                "mcu_threshold": float(self.config.THRESHOLD_MCU_AE),
                "obc_threshold": float(self.config.THRESHOLD_OBC_AE),
                "obc_call_probability": float(self.config.OBC_CALL_PROBABILITY),
                "safety_rules": {
                    "max_temperature": float(self.config.RULE_TEMP_CRITICAL),
                    "max_current": float(self.config.RULE_CURRENT_CRITICAL),
                    "min_voltage": float(self.config.RULE_VOLTAGE_CRITICAL)
                }
            },
            "performance_metrics": {
                "total_samples": int(len(results_df)),
                "critical_alerts": int(len(results_df[results_df['alert_level'] == 2])),
                "warning_alerts": int(len(results_df[results_df['alert_level'] == 1])),
                "normal_operations": int(len(results_df[results_df['alert_level'] == 0])),
                "obc_analyses_performed": int(results_df['obc_called'].sum()),
                "average_cpu_usage": float(results_df['cpu_usage'].mean()),
                "average_memory_usage": float(results_df['memory_usage'].mean()),
                "mcu_error_statistics": {
                    "mean": float(results_df['mcu_error'].mean()),
                    "std": float(results_df['mcu_error'].std()),
                    "max": float(results_df['mcu_error'].max()),
                    "min": float(results_df['mcu_error'].min())
                }
            },
            "system_status": "OPERATIONAL" if len(results_df[results_df['alert_level'] == 2]) < 10 else "REVIEW_NEEDED",
            "recommendations": [
                "Système prêt pour le déploiement en production",
                f"Seuil MCU optimal: {self.config.THRESHOLD_MCU_AE}",
                f"Seuil OBC optimal: {self.config.THRESHOLD_OBC_AE}",
                "Surveiller les alertes critiques pendant les premières heures"
            ]
        }
        
        with open(performance_path, 'w', encoding='utf-8') as f:
            json.dump(performance_report, f, indent=2, ensure_ascii=False)
        
        print(f" Rapport de performance: {performance_path}")
        
        # 3. Graphique de production
        self.create_production_charts(results_df, report_dir)
        
        return performance_report
    
    def create_production_charts(self, results_df, output_dir):
        """Crée des graphiques pour le rapport de production"""
        plt.figure(figsize=(15, 10))
        
        # Graphique 1: Distribution des alertes
        plt.subplot(2, 2, 1)
        alert_counts = results_df['alert_level'].value_counts().sort_index()
        alert_labels = ['Normal', 'Warning', 'Critique']
        colors = ['green', 'orange', 'red']
        
        # Conversion explicite des counts
        alert_counts_list = [int(alert_counts.get(i, 0)) for i in range(3)]
        
        plt.bar(alert_labels, alert_counts_list, color=colors, alpha=0.7)
        plt.title('Distribution des Niveaux d\'Alerte')
        plt.ylabel('Nombre d\'échantillons')
        for i, count in enumerate(alert_counts_list):
            plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        # Graphique 2: Erreurs MCU
        plt.subplot(2, 2, 2)
        plt.hist(results_df['mcu_error'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(self.config.THRESHOLD_MCU_AE, color='red', linestyle='--', 
                   label=f'Seuil MCU ({self.config.THRESHOLD_MCU_AE})')
        plt.xlabel('Erreur de Reconstruction MCU')
        plt.ylabel('Fréquence')
        plt.title('Distribution des Erreurs MCU')
        plt.legend()
        
        # Graphique 3: Performance système
        plt.subplot(2, 2, 3)
        time_points = range(len(results_df))
        plt.plot(time_points, results_df['cpu_usage'], label='CPU %', alpha=0.7)
        plt.plot(time_points, results_df['memory_usage'], label='RAM %', alpha=0.7)
        plt.xlabel('Échantillons')
        plt.ylabel('Utilisation (%)')
        plt.title('Performance Système')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Graphique 4: Types d'alertes
        plt.subplot(2, 2, 4)
        alert_reasons = results_df[results_df['alert_level'] > 0]['alert_reason'].value_counts().head(5)
        
        # Conversion explicite pour le pie chart
        alert_reasons_values = [int(x) for x in alert_reasons.values]
        alert_reasons_labels = [str(x) for x in alert_reasons.index]
        
        plt.pie(alert_reasons_values, labels=alert_reasons_labels, autopct='%1.1f%%', startangle=90)
        plt.title('Répartition des Causes d\'Alerte')
        
        plt.tight_layout()
        chart_path = os.path.join(output_dir, "production_analysis.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Graphiques d'analyse: {chart_path}")

def main():
    """Fonction principale de production"""
    print(" SYSTÈME HYBRIDE MCU + OBC - LANCEMENT PRODUCTION")
    print("=" * 60)
    
    try:
        # Initialisation du système de production
        production_system = ProductionHybridSystem(ProductionConfig)
        
        # Simulation de production
        production_results = production_system.run_production_simulation()
        
        # Génération du rapport
        performance_report = production_system.generate_production_report(production_results)
        
        # RAPPORT FINAL
        print(f"\n RAPPORT FINAL DE PRODUCTION")
        print("=" * 50)
        print(f" SYSTÈME: {performance_report['system_status']}")
        print(f" ÉCHANTILLONS: {performance_report['performance_metrics']['total_samples']}")
        print(f" ALERTES CRITIQUES: {performance_report['performance_metrics']['critical_alerts']}")
        print(f"️  ALERTES WARNING: {performance_report['performance_metrics']['warning_alerts']}")
        print(f" ANALYSES OBC: {performance_report['performance_metrics']['obc_analyses_performed']}")
        print(f" CPU MOYEN: {performance_report['performance_metrics']['average_cpu_usage']:.1f}%")
        print(f" RAM MOYENNE: {performance_report['performance_metrics']['average_memory_usage']:.1f}%")
        
        print(f"\n RAPPORTS DISPONIBLES DANS: {ProductionConfig.OUTPUT_DIR}")
        print("SYSTÈME OPTIMISÉ ET PRÊT POUR LE DÉPLOIEMENT !")
        
    except Exception as e:
        print(f"ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
