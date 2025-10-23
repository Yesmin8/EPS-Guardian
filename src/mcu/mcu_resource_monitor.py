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