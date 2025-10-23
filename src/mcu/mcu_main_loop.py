import pandas as pd
import time
import os
import json 
from datetime import datetime
from mcu_rule_engine import MCU_RuleEngine
from mcu_resource_monitor import measure_resource_usage, ResourceMonitor
from mcu_data_interface import DataInterface
from mcu_logger import setup_logger


class MCU_MainLoop:
    def __init__(self, data_source="csv", data_path=None):
        self.logger = setup_logger("mcu_main_loop.log")
        
        # Utiliser le chemin absolu si aucun chemin n'est fourni
        if data_path is None:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_path = os.path.join(BASE_DIR, "data", "dataset", "pro_eps_dataset.csv")
        
        self.rule_engine = MCU_RuleEngine()
        self.data_interface = DataInterface(data_source, data_path)
        self.resource_monitor = ResourceMonitor()
        
        self.results = []
        self.alert_count = {
            "ALERT_CRITICAL": 0, "SUMMARY": 0, "DIAGNOSTIC_SENSOR": 0, "STATUS_HEARTBEAT": 0
        }
        
        self.logger.info("MCU_MainLoop initialisé")

    @measure_resource_usage
    def process_sample(self, sample):
        return self.rule_engine.apply_rules(sample)

    def run_simulation(self, max_samples=None):
        total_samples = self.data_interface.get_total_samples()
        if max_samples and max_samples < total_samples:
            total_samples = max_samples
            
        self.logger.info(f"Début simulation sur {total_samples} échantillons")
        start_time = time.time()
        
        for i in range(total_samples):
            sample = self.data_interface.get_next()
            if sample is None: break
                
            resource_usage = self.process_sample(sample)
            actions, message_type, details = resource_usage["result"]
            
            result_entry = {
                "timestamp": sample.get("timestamp", f"sample_{i}"),
                "V_batt": sample["V_batt"], "I_batt": sample["I_batt"], "T_batt": sample["T_batt"],
                "V_bus": sample["V_bus"], "V_solar": sample["V_solar"],
                "actions": "|".join(actions), "message_type": message_type,
                "rule_triggered": details.get("rule_triggered", "None"),
                "exec_time_ms": resource_usage["exec_time_ms"],
                "mem_used_MB": resource_usage["mem_used_MB"], "details": str(details)
            }
            
            self.results.append(result_entry)
            self.alert_count[message_type] += 1
            
            if i % 100 == 0:
                current_usage = self.resource_monitor.get_current_usage()
                self.logger.info(f"Progression: {i}/{total_samples} | Mémoire: {current_usage['memory_MB']}MB")
        
        execution_time = time.time() - start_time
        self.logger.info(f"Simulation terminée en {execution_time:.2f}s")
        self.save_results()
        return self.generate_report(execution_time)

    def save_results(self):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        outputs_dir = os.path.join(BASE_DIR, "data", "mcu", "outputs")
        obc_messages_dir = os.path.join(outputs_dir, "obc_messages")
        os.makedirs(obc_messages_dir, exist_ok=True)
            
        # Sauvegarder les résultats détaillés
        results_df = pd.DataFrame(self.results)
        results_path = os.path.join(outputs_dir, "mcu_simulation_results.csv")
        results_df.to_csv(results_path, index=False)
        
        # Sauvegarder les statistiques
        stats_path = os.path.join(outputs_dir, "mcu_simulation_stats.csv")
        stats_df = pd.DataFrame([{
            "timestamp": datetime.now().isoformat(), "total_samples": len(self.results),
            "alert_critical": self.alert_count["ALERT_CRITICAL"], "alert_summary": self.alert_count["SUMMARY"],
            "alert_diagnostic": self.alert_count["DIAGNOSTIC_SENSOR"], "status_heartbeat": self.alert_count["STATUS_HEARTBEAT"],
            "avg_exec_time_ms": results_df["exec_time_ms"].mean(), "max_exec_time_ms": results_df["exec_time_ms"].max(),
            "avg_mem_used_MB": results_df["mem_used_MB"].mean()
        }])
        stats_df.to_csv(stats_path, index=False)
        
        # Sauvegarder les messages OBC
        summary_messages_path = os.path.join(obc_messages_dir, "summary_messages.json")
        with open(summary_messages_path, 'w') as f:
            json.dump({
                "simulation_summary": {
                    "total_samples": len(self.results),
                    "alerts": self.alert_count,
                    "timestamp": datetime.now().isoformat()
                }
            }, f, indent=2)

    def generate_report(self, execution_time):
        report = {
            "timestamp": datetime.now().isoformat(), "total_samples": len(self.results),
            "execution_time_seconds": round(execution_time, 2),
            "samples_per_second": round(len(self.results) / execution_time, 2),
            "alerts_total": sum(self.alert_count.values()) - self.alert_count["STATUS_HEARTBEAT"],
            "alerts_breakdown": self.alert_count.copy(),
            "resource_usage": self.resource_monitor.get_current_usage()
        }
        
        print("\n" + "="*50)
        print("RAPPORT SIMULATION MCU")
        print("="*50)
        print(f"Échantillons traités: {report['total_samples']}")
        print(f"Temps d'exécution: {report['execution_time_seconds']}s")
        print(f"Débit: {report['samples_per_second']} éch./s")
        print(f"Alertes critiques: {report['alerts_breakdown']['ALERT_CRITICAL']}")
        print(f"Résumés: {report['alerts_breakdown']['SUMMARY']}")
        print(f"Défauts capteurs: {report['alerts_breakdown']['DIAGNOSTIC_SENSOR']}")
        print(f"Utilisation mémoire: {report['resource_usage']['memory_MB']}MB")
        print("="*50)
        
        return report

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(BASE_DIR, "data", "dataset", "pro_eps_dataset.csv")
    
    print(f"Recherche du fichier: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"ERREUR: Fichier données introuvable: {data_path}")
        return None
    
    print("Fichier de données trouvé!")
    print("Initialisation de la simulation MCU...")
    
    try:
        mcu_sim = MCU_MainLoop(data_source="csv", data_path=data_path)
        report = mcu_sim.run_simulation(max_samples=1000)
        return report
    except Exception as e:
        print(f"ERREUR lors de la simulation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
mcu_data_interface.py :
import pandas as pd
import random
import math
import os

class DataInterface:
    def __init__(self, source="csv", path=None):
        self.source = source
        self.current_index = 0
        
        if source == "csv" and path:
            # Vérifier que le fichier existe
            if not os.path.exists(path):
                raise FileNotFoundError(f"Fichier introuvable: {path}")
            
            self.df = pd.read_csv(path)
            self.data = self.df.to_dict(orient="records")
            self.log_size = len(self.data)
            print(f"Chargé {self.log_size} échantillons depuis {path}")
        elif source == "simulated":
            self.data = None
            self.log_size = 1000  # Taille par défaut pour simulation
    
    def get_next(self, index=None):
        if self.source == "csv":
            if index is not None:
                self.current_index = index
                
            if self.current_index < len(self.data):
                sample = self.data[self.current_index]
                self.current_index += 1
                return sample
            else:
                return None
                
        elif self.source == "simulated":
            # Génération de données simulées réalistes
            return {
                "timestamp": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d} "
                           f"{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}",
                "V_batt": random.uniform(6.5, 8.4),
                "I_batt": random.uniform(-2.5, 2.5),
                "T_batt": random.uniform(15, 45),
                "V_bus": random.uniform(7.5, 8.2),
                "I_bus": random.uniform(0.1, 1.5),
                "V_solar": random.uniform(10, 18),
                "I_solar": random.uniform(0, 2.0)
            }
    
    def get_sample(self, index):
        """Récupère un échantillon spécifique"""
        if self.source == "csv" and 0 <= index < len(self.data):
            return self.data[index]
        return None
    
    def get_total_samples(self):
        """Retourne le nombre total d'échantillons"""
        if self.source == "csv":
            return len(self.data)
        return self.log_size
    
    def reset(self):
        """Réinitialise l'index de lecture"""
        self.current_index = 0
mcu_resource_monitor.py :

import time
import psutil
import os
import functools
import json
import pandas as pd
from datetime import datetime

def measure_resource_usage(func):
    """Décorateur pour mesurer temps d'exécution et mémoire"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        
        # Mémoire avant
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Temps d'exécution
        start_time = time.time()
        start_perf = time.perf_counter()
        
        # Exécution fonction
        result = func(*args, **kwargs)
        
        end_perf = time.perf_counter()
        end_time = time.time()
        
        # Mémoire après
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        
        return {
            "exec_time_ms": round((end_perf - start_perf) * 1000, 3),
            "wall_time_ms": round((end_time - start_time) * 1000, 3),
            "mem_used_MB": round(mem_after - mem_before, 4),
            "mem_total_MB": round(mem_after, 2),
            "result": result
        }
    return wrapper

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.process.memory_info().rss / (1024 * 1024)
        self.usage_data = []
        
        # Créer le dossier de sortie
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.output_dir = os.path.join(BASE_DIR, "data", "mcu", "outputs", "resource_monitoring")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_current_usage(self):
        """Retourne l'utilisation actuelle des ressources"""
        memory = self.process.memory_info().rss / (1024 * 1024)
        cpu = self.process.cpu_percent()
        
        usage_data = {
            "timestamp": datetime.now().isoformat(),
            "memory_MB": round(memory, 2),
            "memory_delta_MB": round(memory - self.initial_memory, 2),
            "cpu_percent": round(cpu, 1)
        }
        
        self.usage_data.append(usage_data)
        
        # Sauvegarder périodiquement
        if len(self.usage_data) % 10 == 0:
            self.save_usage_data()
        
        return usage_data
    
    def save_usage_data(self):
        """Sauvegarde les données d'utilisation des ressources"""
        try:
            # Sauvegarder en CSV
            csv_path = os.path.join(self.output_dir, "cpu_memory_usage.csv")
            df = pd.DataFrame(self.usage_data)
            df.to_csv(csv_path, index=False)
            
            # Sauvegarder un rapport JSON
            report_path = os.path.join(self.output_dir, "performance_report.json")
            if len(self.usage_data) > 0:
                report = {
                    "timestamp": datetime.now().isoformat(),
                    "samples_collected": len(self.usage_data),
                    "average_memory_MB": round(df["memory_MB"].mean(), 2),
                    "max_memory_MB": round(df["memory_MB"].max(), 2),
                    "average_cpu_percent": round(df["cpu_percent"].mean(), 1),
                    "max_cpu_percent": round(df["cpu_percent"].max(), 1)
                }
                
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                    
        except Exception as e:
            print(f"Erreur sauvegarde données ressources: {e}")
