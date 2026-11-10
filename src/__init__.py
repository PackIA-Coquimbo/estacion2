"""
Paquete principal - Estación 2: Análisis Visual de Calidad
"""

try:
    from src.model import OrangeQualityCNN, build_model, DEFAULT_CLASSES
    __all__ = ["OrangeQualityCNN", "build_model", "DEFAULT_CLASSES"]
except ImportError:
    DEFAULT_CLASSES = [
        "buen_estado",
        "marginalmente_buena",
        "mal_estado",
    ]
    __all__ = ["DEFAULT_CLASSES"]
