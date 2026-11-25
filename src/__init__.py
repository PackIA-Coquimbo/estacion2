"""
Paquete principal - Estación 2: Análisis Visual de Calidad
"""

try:
    from src.model import OrangeQualityCNN, build_model, DEFAULT_CLASSES
    __all__ = ["OrangeQualityCNN", "build_model", "DEFAULT_CLASSES"]
except ImportError:
    # Entorno Edge sin PyTorch
    from src.classes import DEFAULT_CLASSES
    __all__ = ["DEFAULT_CLASSES"]
