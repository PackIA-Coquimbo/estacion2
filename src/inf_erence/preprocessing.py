"""
Preprocesamiento de frames y utilidades numéricas para inferencia en Edge
(Estación 2: Análisis Visual de Calidad - OrangeQuality-AI)

Solo depende de NumPy y OpenCV: apto para Raspberry Pi sin PyTorch.
"""

import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    cv2 = None
    HAS_OPENCV = False


# Normalización estándar (ImageNet) pre-vectorizada para NumPy (1, 3, 1, 1)
NORM_MEAN_NP = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)


NORM_STD_NP = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)



def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    """
    Calcula la función Softmax de forma numéricamente estable usando exclusivamente NumPy.
    Sustituye la dependencia de torch.nn.functional.softmax en la Raspberry Pi.
    """
    shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)



def preprocess_cv2_fast(frame_bgr: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Preprocesamiento vectorizado directo (Zero-Copy / Fast Path) desde OpenCV.
    Elimina por completo la sobrecarga de conversión a PIL.Image y torchvision.transforms.

    Operaciones:
    1. cv2.resize: Redimensionamiento rápido con interpolación lineal a (target_size, target_size).
    2. cv2.cvtColor: Conversión de espacio de color BGR a RGB.
    3. Transposición matricial: Reordenamiento de dimensiones HWC a CHW.
    4. Normalización matricial directa en float32: (img / 255.0 - mean) / std.

    Retorna:
        np.ndarray contiguo en memoria con forma (1, 3, target_size, target_size) y dtype float32.
    """
    if not HAS_OPENCV:
        raise RuntimeError("OpenCV no está disponible. Instale 'opencv-python' o 'opencv-python-headless'.")

    # 1. Resize rápido
    resized = cv2.resize(frame_bgr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # 2. BGR -> RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 3. HWC a CHW y añadir dimensión de batch (1, C, H, W)
    chw = np.transpose(rgb, (2, 0, 1))
    tensor = np.expand_dims(chw, axis=0).astype(np.float32) / 255.0

    # 4. Normalización matricial directa
    tensor = (tensor - NORM_MEAN_NP) / NORM_STD_NP

    return np.ascontiguousarray(tensor, dtype=np.float32)

