"""
Package IA du système OBC
Contient les modèles d'apprentissage automatique pour l'analyse avancée
"""

from .ai_complex_inference import OBC_AI, obc_ai, analyze_sensor_sequence

__all__ = ['OBC_AI', 'obc_ai', 'analyze_sensor_sequence']
__version__ = '1.0.0'