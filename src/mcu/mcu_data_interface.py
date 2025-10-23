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