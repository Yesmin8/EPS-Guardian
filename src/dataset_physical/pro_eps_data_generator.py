import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import json
import logging
import os
import sys
import math

# === CONFIGURATION CORRIGÉE DES CHEMINS ===
BASE_DIR = r"D:\Challenge AESS&IES"
DATA_DIR = os.path.join(BASE_DIR, "data")
ANALYSE_DIR = os.path.join(DATA_DIR, "analyse")
VISUALIZATIONS_DIR = os.path.join(ANALYSE_DIR, "visualizations")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
DATA_TRAIN_DIR = os.path.join(DATA_DIR, "data_train")
LOGS_DIR = os.path.join(DATA_DIR, "logs")

# Configuration du logging - UNIQUEMENT CONSOLE
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # SUPPRIMER FileHandler
)

logger = logging.getLogger(__name__)

class ProEPSSensorDataGenerator:
    """Generateur de donnees pour systeme EPS de CubeSat"""
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Configuration des capteurs
        self.expected_cols = [
            'timestamp', 'V_batt', 'I_batt', 'T_batt', 'V_solar', 'I_solar',
            'V_bus', 'I_bus', 'T_eps', 'SOC', 'anomaly_type', 'orbit_sunlight'
        ]
        
        # Plages NORMALES des capteurs
        self.normal_ranges = {
            'V_batt': (7.2, 8.4),
            'I_batt': (-2.5, 2.5),
            'T_batt': (-10, 60),
            'V_solar': (15.0, 18.0),
            'I_solar': (0, 2.0),
            'V_bus': (7.6, 8.2),
            'I_bus': (0.05, 1.5),
            'T_eps': (-10, 55),
            'SOC': (15, 95)
        }
        
        # Etat du systeme
        self.system_state = {
            'battery_health': 1.0,
            'solar_efficiency': 1.0,
            'converter_efficiency': 0.92,
            'load_power': 0.7,
            'internal_resistance': 0.05,
            'degradation_cycles': 0,
            'orbit_position': 0.0,
            'thermal_mass_batt': 25.0,
            'thermal_mass_eps': 20.0,
            'thermal_history_batt': [],
            'thermal_history_eps': []
        }
        
        # Configuration des chemins CORRIGÉE
        self.data_dir = DATASET_DIR
        self.output_dir = VISUALIZATIONS_DIR
        self.logs_dir = LOGS_DIR
        self.rejected_samples = 0
        self.start_time = None
        
        logger.info("Generateur EPS initialise")

    def _simulate_orbit_physics(self, timestamp):
        """Simule la physique orbitale"""
        orbit_period = 90 * 60
        total_seconds = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
        orbit_progress = (total_seconds % orbit_period) / orbit_period
        
        in_sunlight = orbit_progress < 0.6
        
        operation_cycle = np.sin(orbit_progress * 4 * np.pi)
        load_variation = 0.3 * operation_cycle
        
        thermal_cycle = np.sin(orbit_progress * 2 * np.pi)
        thermal_variation = 12 * thermal_cycle
        
        self.system_state['orbit_position'] = orbit_progress
        
        return in_sunlight, load_variation, thermal_variation

    def _add_stochastic_noise(self, data_dict):
        """Ajoute un bruit stochastique"""
        noise_config = {
            'V_batt': (0, 0.015),
            'I_batt': (0, 0.02),
            'T_batt': (0, 0.1),
            'V_solar': (0, 0.025),
            'I_solar': (0, 0.015),
            'V_bus': (0, 0.012),
            'I_bus': (0, 0.008),
            'T_eps': (0, 0.08),
            'SOC': (0, 0.05)
        }
        
        for param, (mean, std) in noise_config.items():
            if param in data_dict and not pd.isna(data_dict[param]):
                noise = np.random.normal(mean, std)
                data_dict[param] += noise
                if param in ['V_batt', 'I_batt', 'V_solar', 'I_solar', 'V_bus', 'I_bus']:
                    data_dict[param] = round(data_dict[param], 3)
                elif param in ['T_batt', 'T_eps']:
                    data_dict[param] = round(data_dict[param], 1)
                elif param == 'SOC':
                    data_dict[param] = round(data_dict[param], 1)
        
        return data_dict

    def _simulate_thermal_lag(self, current_temp, thermal_variation, mass_type='batt'):
        """Simule le retard thermique"""
        thermal_mass = self.system_state['thermal_mass_batt'] if mass_type == 'batt' else self.system_state['thermal_mass_eps']
        lag_factor = 0.8 if mass_type == 'eps' else 1.0
        
        lagged_variation = thermal_variation * lag_factor
        
        thermal_key = f'thermal_history_{mass_type}'
        
        self.system_state[thermal_key].append(current_temp + lagged_variation)
        
        if len(self.system_state[thermal_key]) > 10:
            self.system_state[thermal_key] = self.system_state[thermal_key][-10:]
        
        if len(self.system_state[thermal_key]) > 1:
            lagged_temp = np.mean(self.system_state[thermal_key][-3:]) if len(self.system_state[thermal_key]) >= 3 else current_temp + lagged_variation
        else:
            lagged_temp = current_temp + lagged_variation
        
        return lagged_temp

    def _check_data_quality(self, data_dict):
        """Verifie la qualite physique des donnees generees"""
        issues = []
        
        has_nan = any(pd.isna(v) for v in data_dict.values())
        
        checks = [
            (data_dict.get('V_batt', 0) < 5.5 or data_dict.get('V_batt', 0) > 8.6, 
             f"V_batt hors limites: {data_dict.get('V_batt', 'NaN')}V"),
            (data_dict.get('T_batt', 0) < -25 or data_dict.get('T_batt', 0) > 85, 
             f"T_batt hors limites: {data_dict.get('T_batt', 'NaN')}°C"),
            (data_dict.get('SOC', 0) < 0 or data_dict.get('SOC', 0) > 105, 
             f"SOC hors limites: {data_dict.get('SOC', 'NaN')}%"),
            (abs(data_dict.get('I_batt', 0)) > 3.0,
             f"I_batt excessif: {data_dict.get('I_batt', 'NaN')}A"),
            (data_dict.get('I_bus', 0) < 0,
             f"I_bus negatif: {data_dict.get('I_bus', 'NaN')}A")
        ]
        
        for check, message in checks:
            if check and not has_nan:
                issues.append(message)
                self.rejected_samples += 1
            
        return len(issues) == 0

    def generate_normal_data(self, timestamp):
        """Genere des donnees EPS normales"""
        try:
            in_sunlight, load_variation, thermal_variation = self._simulate_orbit_physics(timestamp)
            
            solar_factor = 1.0 if in_sunlight else 0.05
            solar_factor *= self.system_state['solar_efficiency']

            base_temp = 8 if in_sunlight else -8
            
            V_solar = np.random.uniform(15.0, 18.0) * solar_factor
            I_solar = np.random.uniform(0.5, 2.0) * solar_factor
            
            solar_power = V_solar * I_solar * self.system_state['converter_efficiency']
            load_demand = (0.7 + load_variation) * self.system_state['load_power'] * 8
            
            power_balance = solar_power - load_demand
            
            if power_balance > 0.5:
                I_batt = min(power_balance / 8.0, 2.5)
                V_batt = np.random.uniform(8.0, 8.4) * self.system_state['battery_health']
            else:
                I_batt = max(power_balance / 8.0, -2.5)
                V_batt = np.random.uniform(7.2, 7.8) * self.system_state['battery_health']
            
            T_batt_base = base_temp + abs(I_batt) * 1.2 + random.uniform(-2, 2)
            T_batt = self._simulate_thermal_lag(T_batt_base, thermal_variation, 'batt')
            
            T_eps_base = base_temp + (abs(I_batt) + abs(I_solar)) * 0.8 + random.uniform(-2, 2)
            T_eps = self._simulate_thermal_lag(T_eps_base, thermal_variation * 0.8, 'eps')
            
            soc_base = 50
            soc_variation = (V_batt - 7.8) / (8.4 - 7.8) * 35
            SOC = max(15, min(95, soc_base + soc_variation)) * self.system_state['battery_health']
            
            I_bus = max(load_demand / 8.0, 0.05)
            
            V_bus = V_batt - abs(I_batt) * self.system_state['internal_resistance']
            
            data = {
                'timestamp': timestamp,
                'V_batt': round(V_batt, 3),
                'I_batt': round(I_batt, 3),
                'T_batt': round(T_batt, 1),
                'V_solar': round(V_solar, 3),
                'I_solar': round(I_solar, 3),
                'V_bus': round(V_bus, 3),
                'I_bus': round(I_bus, 3),
                'T_eps': round(T_eps, 1),
                'SOC': round(SOC, 1),
                'anomaly_type': 'normal',
                'orbit_sunlight': in_sunlight
            }
            
            data = self._add_stochastic_noise(data)
            
            if not self._check_data_quality(data):
                return None
                
            return data
            
        except Exception as e:
            logger.error(f"Erreur generation donnees: {e}")
            return None

    def generate_anomaly(self, base_data, anomaly_type):
        """Genere differents types d'anomalies"""
        data = base_data.copy()
        data['anomaly_type'] = anomaly_type
        
        if anomaly_type == 'batt_overheat':
            data['T_batt'] = round(random.uniform(55, 70), 1)
            data['V_batt'] = round(data['V_batt'] * 0.98, 3)
            data['I_batt'] = round(min(abs(data['I_batt']) * -1.2, -1.8), 3)
            
        elif anomaly_type == 'batt_undervoltage':
            data['V_batt'] = round(random.uniform(6.0, 6.5), 3)
            data['I_batt'] = round(max(-abs(data['I_batt']) * 1.5, -2.2), 3)
            data['SOC'] = round(random.uniform(10, 20), 1)
            
        elif anomaly_type == 'batt_overvoltage':
            data['V_batt'] = round(random.uniform(8.5, 8.6), 3)
            data['I_batt'] = round(min(abs(data['I_batt']) * 1.3, 2.3), 3)
            data['T_batt'] += 8
            
        elif anomaly_type == 'batt_overcurrent':
            data['I_batt'] = round(random.uniform(2.6, 3.0), 3)
            data['T_batt'] += 12
            data['V_batt'] = round(data['V_batt'] * 0.96, 3)
            
        elif anomaly_type == 'solar_fault':
            data['V_solar'] = round(random.uniform(2.0, 5.0), 3)
            data['I_solar'] = round(random.uniform(0, 0.1), 3)
            data['I_batt'] = round(max(-abs(data['I_batt']) * 1.2, -2.0), 3)
            
        elif anomaly_type == 'converter_failure':
            data['V_bus'] = round(random.uniform(4.0, 6.0), 3)
            data['I_bus'] = round(data['I_bus'] * 1.3, 3)
            data['T_eps'] += 15
            
        elif anomaly_type == 'eps_overheat':
            data['T_eps'] = round(random.uniform(60, 75), 1)
            data['V_bus'] = round(data['V_bus'] * 0.9, 3)
            data['I_bus'] = round(data['I_bus'] * 0.8, 3)
            
        elif anomaly_type == 'sensor_fault':
            data['SOC'] = float('nan')
            data['V_batt'] = float('nan')
            
        elif anomaly_type == 'battery_degradation':
            data['V_batt'] = round(data['V_batt'] * 0.85, 3)
            data['I_batt'] = round(data['I_batt'] * 1.3, 3)
            data['SOC'] = round(data['SOC'] * 0.9, 1)
            
        elif anomaly_type == 'progressive_overheat':
            data['T_batt'] += 20
            data['T_eps'] += 15
            data['V_batt'] = round(data['V_batt'] * 0.98, 3)
            
        elif anomaly_type == 'oscillation_coupled':
            orbit_period = 90 * 60
            if 'timestamp' in data:
                timestamp = data['timestamp']
                total_seconds = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
                orbit_progress = (total_seconds % orbit_period) / orbit_period
                data['V_bus'] = round(data['V_bus'] + 0.3 * np.sin(orbit_progress * 8 * np.pi), 3)
                data['I_bus'] = round(data['I_bus'] + 0.2 * np.sin(orbit_progress * 8 * np.pi + np.pi/4), 3)
                data['V_batt'] = round(data['V_batt'] + 0.1 * np.sin(orbit_progress * 8 * np.pi), 3)
            
        elif anomaly_type == 'solar_partial_failure':
            data['I_solar'] = round(data['I_solar'] * 0.3, 3)
            data['V_solar'] = round(data['V_solar'] * 0.8, 3)
            
        elif anomaly_type == 'unknown_pattern':
            for key in ['V_batt', 'I_batt', 'V_bus', 'I_bus']:
                if key in data and not pd.isna(data[key]):
                    data[key] = round(data[key] + random.uniform(-0.3, 0.3), 3)
            data['T_batt'] = round(data['T_batt'] + random.uniform(-8, 8), 1)
        
        if 'I_bus' in data and data['I_bus'] < 0:
            data['I_bus'] = 0.05
            
        if 'V_bus' in data and 'V_batt' in data and data['V_bus'] > data['V_batt'] * 1.05:
            data['V_bus'] = round(data['V_batt'] * 0.98, 3)
        
        data = self._add_stochastic_noise(data)
                    
        return data

    def simulate_system_degradation(self, cycles):
        """Simule la degradation progressive du systeme"""
        degradation_factors = {
            'battery_health': 0.9997,
            'solar_efficiency': 0.9999, 
            'converter_efficiency': 0.99995,
            'internal_resistance': 1.0001
        }
        
        for param, factor in degradation_factors.items():
            if param == 'internal_resistance':
                self.system_state[param] *= (factor ** cycles)
            else:
                self.system_state[param] *= (factor ** cycles)
        
        self.system_state['degradation_cycles'] += cycles

    def generate_dataset(self, num_normal=5000, num_anomalies=1000, duration_hours=24, low_res=False):
        """Genere un dataset complet"""
        logger.info(f"Generation dataset: {num_normal} normaux, {num_anomalies} anomalies")
        
        if low_res:
            num_normal = min(num_normal, 500)
            num_anomalies = min(num_anomalies, 100)
            logger.info(f"Mode LOW_RES active")
        
        dataset = []
        start_time = datetime.now()
        self.start_time = start_time
        self.rejected_samples = 0
        
        # Distribution des anomalies
        anomaly_distribution = {
            'batt_overheat': 0.15,
            'batt_undervoltage': 0.12,
            'batt_overvoltage': 0.10,
            'batt_overcurrent': 0.08,
            'solar_fault': 0.10,
            'converter_failure': 0.08,
            'eps_overheat': 0.07,
            'sensor_fault': 0.05,
            'battery_degradation': 0.08,
            'progressive_overheat': 0.06,
            'oscillation_coupled': 0.05,
            'solar_partial_failure': 0.04,
            'unknown_pattern': 0.02
        }
        
        # Generation donnees normales
        normal_generated = 0
        max_attempts = num_normal * 3
        attempts = 0
        
        while normal_generated < num_normal and attempts < max_attempts:
            attempts += 1
            timestamp = start_time + timedelta(seconds=normal_generated * (duration_hours * 3600 / num_normal))
            data = self.generate_normal_data(timestamp)
            
            if data is not None:
                dataset.append(data)
                normal_generated += 1
            else:
                self.rejected_samples += 1
                
            if normal_generated % 100 == 0:
                self.simulate_system_degradation(1)
    
        if normal_generated < num_normal:
            logger.warning(f"Seulement {normal_generated}/{num_normal} echantillons generes")
        
        # Generation anomalies
        total_anomalies_generated = 0
        
        for anomaly_type, percentage in anomaly_distribution.items():
            anomalies_count = int(num_anomalies * percentage)
            
            for i in range(anomalies_count):
                if total_anomalies_generated >= num_anomalies:
                    break
                    
                if len(dataset) == 0:
                    continue
                    
                insert_pos = random.randint(0, len(dataset) - 1)
                timestamp = dataset[insert_pos]['timestamp']
                
                base_data = self.generate_normal_data(timestamp)
                if base_data:
                    anomaly_data = self.generate_anomaly(base_data, anomaly_type)
                    anomaly_data['timestamp'] = timestamp
                    
                    if self._check_data_quality(anomaly_data):
                        dataset.insert(insert_pos, anomaly_data)
                        total_anomalies_generated += 1
                    else:
                        self.rejected_samples += 1
        
        random.shuffle(dataset)
        
        logger.info(f"Dataset genere: {len(dataset)} echantillons valides")
        logger.info(f"Normaux: {normal_generated}, Anomalies: {total_anomalies_generated}, Rejetes: {self.rejected_samples}")
        
        return pd.DataFrame(dataset)

    def calculate_derived_features(self, df):
        """Calcule les features derivees pour l'IA"""
        try:
            logger.info("Calcul des features derivees...")
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Features de puissance
            df['P_batt'] = df['V_batt'] * df['I_batt']
            df['P_solar'] = ((df['V_solar'] * df['I_solar']).clip(lower=0, upper=40))
            df['P_bus'] = df['V_bus'] * df['I_bus']
            
            # Ratios systeme
            df['converter_ratio'] = np.where(
                df['V_solar'] > 0.1,
                df['V_bus'] / df['V_solar'],
                0
            )
            
            # Efficacites
            df['charge_efficiency'] = np.where(
                (df['I_batt'] > 0) & (df['P_solar'] > 0.1),
                df['P_batt'] / df['P_solar'],
                0
            )
            
            # Derivees temporelles
            time_cols = ['V_batt', 'I_batt', 'T_batt', 'V_bus', 'I_bus', 'SOC']
            for col in time_cols:
                df[f'delta_{col}'] = df[col].diff().fillna(0)
                
            # Statistiques glissantes
            for col in ['V_batt', 'I_batt', 'T_batt']:
                df[f'rolling_std_{col}'] = df[col].rolling(window=10, min_periods=1).std().fillna(0)
                df[f'rolling_mean_{col}'] = df[col].rolling(window=10, min_periods=1).mean().fillna(df[col])
            
            # Ratios de coherence
            df['V_batt_V_bus_ratio'] = np.where(
                df['V_bus'] > 0.1,
                df['V_batt'] / df['V_bus'],
                1.0
            )
            df['power_balance'] = df['P_solar'] - df['P_batt'] - df['P_bus']
            
            logger.info("Features derivees calculees avec succes")
            return df
            
        except Exception as e:
            logger.error(f"Erreur calcul features: {e}")
            return df

    def save_dataset(self, df, filename="pro_eps_dataset.csv"):
        """Sauvegarde le dataset avec metadonnees"""
        try:
            missing_cols = [col for col in self.expected_cols if col not in df.columns]
            
            metadata = {
                'generation_date': datetime.now().isoformat(),
                'dataset_info': {
                    'total_samples': len(df),
                    'normal_samples': len(df[df['anomaly_type'] == 'normal']),
                    'anomaly_samples': len(df[df['anomaly_type'] != 'normal']),
                    'rejected_samples': self.rejected_samples,
                    'data_quality_score': round((len(df) / (len(df) + self.rejected_samples)) * 100, 1),
                    'low_resolution': len(df) <= 600
                },
                'anomaly_distribution': df['anomaly_type'].value_counts().to_dict(),
                'system_health': self.system_state.copy(),
                'data_quality': {
                    'missing_columns': missing_cols,
                    'has_nan': df.isna().sum().to_dict(),
                    'data_integrity': len(missing_cols) == 0,
                    'physical_consistency': 'HIGH' if self.rejected_samples < len(df) * 0.05 else 'MEDIUM'
                }
            }
            
            file_path = os.path.join(DATASET_DIR, filename)
            df.to_csv(file_path, index=False)
            
            metadata_file = os.path.join(DATASET_DIR, "pro_eps_dataset_metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dataset sauvegarde: {file_path}")
            
            print(f"\nSynthese dataset:")
            print(f"   Echantillons valides: {len(df)}")
            print(f"   Qualite donnees: {metadata['data_quality']['physical_consistency']}")
            print(f"   Anomalies rejetees: {self.rejected_samples}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde: {e}")
            return None
