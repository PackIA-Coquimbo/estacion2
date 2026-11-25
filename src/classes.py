"""
Clases de calidad visual y colores de visualización (sin dependencias pesadas).
(Estación 2: Análisis Visual de Calidad - OrangeQuality-AI)

Fuente única de las clases: el orden define el índice de salida del modelo.
"""

from typing import Dict, List, Tuple

# Clases por defecto del proyecto según la estructura de dataset/
DEFAULT_CLASSES: List[str] = [
    "buen_estado",
    "marginalmente_buena",
    "mal_estado",
]

# Paleta de colores para visualización BGR (OpenCV)
CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "buen_estado": (40, 200, 40),          # Verde
    "marginalmente_buena": (0, 200, 240),  # Amarillo / Ámbar
    "mal_estado": (40, 40, 220),           # Rojo
}
