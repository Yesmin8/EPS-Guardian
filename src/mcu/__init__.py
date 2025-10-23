"""
Module MCU - Système embarqué EPS GUARDIAN
"""

from mcu_rule_engine import MCU_RuleEngine
from mcu_main_loop import MCU_MainLoop
from mcu_data_interface import DataInterface
from mcu_resource_monitor import ResourceMonitor, measure_resource_usage
from mcu_logger import setup_logger

__version__ = "1.0.0"
__author__ = "EPS GUARDIAN Team"
__description__ = "Système de détection d'anomalies pour microcontrôleur ESP32"

__all__ = [
    'MCU_RuleEngine',
    'MCU_MainLoop', 
    'DataInterface',
    'ResourceMonitor',
    'measure_resource_usage',
    'setup_logger'
]