"""
Calificación por naranja a partir de varias fotos
(Estación 2: Análisis Visual de Calidad - OrangeQuality-AI)

Contiene en un único lugar:
- La convención de nombres de las fotos (lectura y escritura).
- La agrupación de fotos por naranja.
- El promedio de probabilidades (soft voting) y el JSON de salida por naranja.

Para cambiar el formato de los nombres basta con editar ORANGE_PHOTO_NAME_FORMAT y
ORANGE_PHOTO_PATTERN (deben describir el mismo formato; lo verifica tests/test_multi_photo.py).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import re
import sys
import time

import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    cv2 = None
    HAS_OPENCV = False


# Convención de nombres de las fotos que entrega la Estación 1 (varias fotos por naranja):
#   <uid>-<n_naranja>.<n_foto>[.ext]   ej: 7712323-1.1.jpg, 7712323-1.2.jpg, 7712323-2.1.jpg
# Escritura (sin extensión) y lectura deben describir el mismo formato.
ORANGE_PHOTO_NAME_FORMAT = "{uid}-{orange}.{photo}"
ORANGE_PHOTO_PATTERN = re.compile(r"^(?P<uid>.+)-(?P<orange>\d+)\.(?P<photo>\d+)$")
# Identificador de la naranja en el JSON de salida y en el nombre del archivo de resultado
ORANGE_ID_FORMAT = "{uid}-{orange}"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


EXPECTED_PHOTOS_PER_ORANGE = 2



def format_orange_photo_name(uid: str, orange_number: int, photo_number: int, ext: str = ".jpg") -> str:
    """Construye el nombre de archivo de una foto: format_orange_photo_name('7712323', 1, 2) -> '7712323-1.2.jpg'."""
    return ORANGE_PHOTO_NAME_FORMAT.format(uid=uid, orange=orange_number, photo=photo_number) + ext


def make_orange_id(uid: str, orange_number: int) -> str:
    """Identificador de la naranja: make_orange_id('7712323', 1) -> '7712323-1'."""
    return ORANGE_ID_FORMAT.format(uid=uid, orange=orange_number)


def parse_orange_photo_name(path: Union[str, Path]) -> Optional[Tuple[str, int, int]]:
    """
    Extrae (uid, numero_naranja, numero_foto) del nombre de archivo.
    Acepta nombres con o sin extensión de imagen: '7712323-1.2.jpg' o '7712323-1.2'.
    Retorna None si el nombre no sigue la convención.
    """
    name = Path(path).name
    if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
        name = Path(name).stem
    match = ORANGE_PHOTO_PATTERN.match(name)
    if match is None:
        return None
    return match.group("uid"), int(match.group("orange")), int(match.group("photo"))



def group_photos_by_orange(
    paths: Sequence[Union[str, Path]],
) -> Tuple[Dict[Tuple[str, int], List[Path]], List[Path]]:
    """
    Agrupa rutas de fotos por naranja (uid, numero_naranja), ordenadas por número de foto.
    Retorna (grupos, rutas_que_no_siguen_la_convencion).
    """
    groups: Dict[Tuple[str, int], List[Tuple[int, Path]]] = {}
    unmatched: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        parsed = parse_orange_photo_name(path)
        if parsed is None:
            unmatched.append(path)
            continue
        uid, orange_number, photo_number = parsed
        groups.setdefault((uid, orange_number), []).append((photo_number, path))

    ordered = {
        key: [path for _, path in sorted(items)]
        for key, items in sorted(groups.items())
    }
    return ordered, unmatched



def aggregate_probabilities(
    probs_list: Sequence[np.ndarray],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    """
    Promedia las probabilidades Softmax de varias fotos de una misma naranja
    (soft voting) y entrega la clase final con su confianza promedio.
    """
    if len(probs_list) == 0:
        raise ValueError("Se requiere al menos una distribución de probabilidades.")

    stacked = np.stack([np.asarray(p, dtype=np.float64).reshape(-1) for p in probs_list])
    if stacked.shape[1] != len(class_names):
        raise ValueError(
            f"Cada foto debe tener {len(class_names)} probabilidades; se recibieron {stacked.shape[1]}."
        )

    mean_probs = stacked.mean(axis=0)
    best_idx = int(np.argmax(mean_probs))
    best_conf = float(mean_probs[best_idx])
    per_photo_classes = {int(i) for i in stacked.argmax(axis=1)}

    return {
        "predicted_class": class_names[best_idx],
        "confidence": round(best_conf, 4),
        "confidence_percent": round(best_conf * 100.0, 2),
        "probabilities": {
            name: round(float(prob), 4) for name, prob in zip(class_names, mean_probs)
        },
        "photos_agree": len(per_photo_classes) == 1,
    }



def build_orange_payload(
    uid: str,
    orange_number: int,
    orange_result: Dict[str, Any],
    expected_photos: int = EXPECTED_PHOTOS_PER_ORANGE,
) -> Dict[str, Any]:
    """Arma el JSON estandarizado de salida por naranja (CLI y servicio inter-estaciones)."""
    return {
        "station_id": 2,
        "timestamp": round(time.time(), 3),
        "uid": uid,
        "orange_number": orange_number,
        "orange_id": make_orange_id(uid, orange_number),
        "predicted_class": orange_result["predicted_class"],
        "confidence_percent": orange_result["confidence_percent"],
        "probabilities": orange_result["probabilities"],
        "photos_count": orange_result["photos_count"],
        "expected_photos": expected_photos,
        "complete": orange_result["photos_count"] == expected_photos,
        "photos_agree": orange_result["photos_agree"],
        "photos": orange_result["photos"],
        "latency_preprocess_ms": orange_result["latency_preprocess_ms"],
        "latency_inference_ms": orange_result["latency_inference_ms"],
        "latency_total_ms": orange_result["latency_ms"],
    }



def classify_oranges(
    predictor: Any,
    image_paths: Sequence[Union[str, Path]],
    expected_photos: int = EXPECTED_PHOTOS_PER_ORANGE,
) -> List[Dict[str, Any]]:
    """
    Agrupa las fotos por naranja según su nombre (<uid>-<naranja>.<foto>),
    califica cada naranja promediando sus fotos y retorna un resultado por naranja.
    Los avisos se emiten por stderr para no contaminar la salida JSON.
    """
    if not HAS_OPENCV:
        raise RuntimeError("OpenCV es requerido para leer las imágenes.")

    groups, unmatched = group_photos_by_orange(image_paths)
    for path in unmatched:
        print(
            f"[AVISO] '{path.name}' no sigue el formato <uid>-<naranja>.<foto>; se omite.",
            file=sys.stderr,
        )

    results: List[Dict[str, Any]] = []
    for (uid, orange_number), paths in groups.items():
        orange_id = make_orange_id(uid, orange_number)
        frames: List[np.ndarray] = []
        names: List[str] = []
        for path in paths:
            frame = cv2.imread(str(path)) if path.exists() else None
            if frame is None:
                print(f"[AVISO] No se pudo leer la foto {path} (naranja {orange_id}).", file=sys.stderr)
                continue
            frames.append(frame)
            names.append(path.name)

        if not frames:
            print(f"[AVISO] Naranja {orange_id} sin fotos válidas; no se califica.", file=sys.stderr)
            continue
        if len(frames) != expected_photos:
            print(
                f"[AVISO] Naranja {orange_id}: {len(frames)} foto(s) válidas, "
                f"se esperaban {expected_photos}. Se promedian las disponibles.",
                file=sys.stderr,
            )

        res = predictor.predict_orange(frames, photo_names=names)
        results.append(build_orange_payload(uid, orange_number, res, expected_photos))

    return results

