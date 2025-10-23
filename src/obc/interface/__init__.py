"""
Package Interface du système OBC
Gestion de la communication MCU ↔ OBC
"""

from .obc_message_handler import OBCMessageHandler, message_handler, process_incoming_message
from .obc_response_generator import OBCResponseGenerator, response_generator

__all__ = [
    'OBCMessageHandler', 
    'message_handler', 
    'process_incoming_message',
    'OBCResponseGenerator', 
    'response_generator'
]
__version__ = '1.0.0'