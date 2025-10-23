"""
Package Simulation du système OBC
Tests et validation du système
"""

from .obc_simulate_incoming_data import create_sample_mcu_message, simulate_critical_anomaly, test_obc_system
from .obc_realtime_fusion_test import RealtimeFusionSimulator

__all__ = [
    'create_sample_mcu_message',
    'simulate_critical_anomaly', 
    'test_obc_system',
    'RealtimeFusionSimulator'
]
__version__ = '1.0.0'