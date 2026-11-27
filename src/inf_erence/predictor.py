"""
Motor de predicción de calidad visual (ONNX Runtime en Edge, PyTorch opcional en PC)
(Estación 2: Análisis Visual de Calidad - OrangeQuality-AI)

PyTorch se importa solo al cargar un modelo .pth, por lo que en la Raspberry Pi
basta con onnxruntime, numpy y opencv.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import importlib.util
import sys
import time

import numpy as np

# Asegurar raíz del proyecto en sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    cv2 = None
    HAS_OPENCV = False

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    ort = None
    HAS_ONNXRUNTIME = False

# PyTorch es opcional: se detecta sin importarlo para no pagar su tiempo de carga en Edge
HAS_TORCH = importlib.util.find_spec("torch") is not None

from src.classes import DEFAULT_CLASSES, CLASS_COLORS_BGR
from src.inf_erence.preprocessing import preprocess_cv2_fast, softmax_numpy
from src.inf_erence.oranges import aggregate_probabilities


class QualityPredictor:
    """
    Motor de predicción optimizado para inferencia en CPU ARM (Raspberry Pi 4 / 5) y PC.
    Soporta ONNX Runtime puro (sin dependencias pesadas) y PyTorch (.pth) opcional.
    """
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        use_onnx: bool = True,
        num_threads: int = 4,
    ):
        self.use_onnx = use_onnx
        self.class_names = DEFAULT_CLASSES
        self.num_threads = num_threads

        if self.use_onnx:
            self._load_onnx_session(model_path)
        else:
            if not HAS_TORCH:
                raise ImportError(
                    "PyTorch no está instalado en este entorno. "
                    "Para inferencia en Edge/Raspberry Pi, use modelos ONNX con use_onnx=True."
                )
            self._load_pytorch_model(model_path, device=device)

    def _load_onnx_session(self, model_path: Optional[Union[str, Path]]) -> None:
        """Carga y configura una sesión de ONNX Runtime optimizada para CPU ARM."""
        if not HAS_ONNXRUNTIME:
            raise ImportError("onnxruntime no está instalado. Ejecute 'pip install onnxruntime'.")

        if model_path is None:
            default_int8 = PROJECT_ROOT / "weights" / "orange_model_int8.onnx"
            default_fp32 = PROJECT_ROOT / "weights" / "orange_model.onnx"
            # FP32 primero: el INT8 cuantizado clasificó mal fruta con moho en pruebas reales
            if default_fp32.exists():
                model_path = default_fp32
            elif default_int8.exists():
                model_path = default_int8
            else:
                raise FileNotFoundError(
                    "No se encontró un modelo ONNX por defecto en weights/. "
                    "Especifique la ruta con --onnx weights/<modelo>.onnx"
                )

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Archivo ONNX no encontrado: {model_path}")

        # Configuración de optimización de sesión en ONNX Runtime (Fase 2)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.num_threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        print(f"[QualityPredictor] Sesión ONNX cargada: {model_path.name}", file=sys.stderr)
        print(f"[QualityPredictor] Hilos intra-op: {self.num_threads} | Proveedores: {self.session.get_providers()}", file=sys.stderr)

    def _load_pytorch_model(self, model_path: Optional[Union[str, Path]], device: Optional[str] = None) -> None:
        """Carga el modelo PyTorch tradicional (.pth). PyTorch se importa solo aquí."""
        import torch
        from src.model import build_model

        self._torch = torch
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = build_model(num_classes=len(self.class_names))
        if model_path is not None and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.class_names = checkpoint.get("class_names", self.class_names)
            elif isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint)
            print(f"[QualityPredictor] Modelo PyTorch cargado desde: {model_path}", file=sys.stderr)
        else:
            print("[QualityPredictor] Aviso: Sin archivo de pesos. Usando arquitectura no entrenada.", file=sys.stderr)
        self.model.to(self.device)
        self.model.eval()

    def _infer_probs(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Ejecuta preprocesamiento + inferencia de un frame BGR.
        Retorna (probabilidades[num_classes], latencia_preproceso_ms, latencia_inferencia_ms).
        """
        # 1. Preprocesamiento Zero-Copy
        t0 = time.perf_counter()
        input_tensor = preprocess_cv2_fast(frame_bgr, target_size=224)
        prep_latency_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Inferencia pura
        t_inf_start = time.perf_counter()
        if self.use_onnx:
            raw_logits = self.session.run(None, {self.input_name: input_tensor})[0]
            probs_arr = softmax_numpy(raw_logits)[0]
        else:
            torch = self._torch
            with torch.no_grad():
                torch_in = torch.from_numpy(input_tensor).to(self.device)
                logits = self.model(torch_in)
                probs_arr = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        inf_latency_ms = (time.perf_counter() - t_inf_start) * 1000.0

        return np.asarray(probs_arr, dtype=np.float32), prep_latency_ms, inf_latency_ms

    def predict_orange(
        self,
        frames_bgr: Sequence[np.ndarray],
        photo_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Califica una naranja a partir de varias fotos (p. ej. 2 por naranja).
        Promedia las probabilidades de todas las fotos y entrega la clase final
        con su confianza promedio, junto al detalle de cada foto.
        """
        if len(frames_bgr) == 0:
            raise ValueError("Se requiere al menos una foto por naranja.")
        if photo_names is not None and len(photo_names) != len(frames_bgr):
            raise ValueError("photo_names debe tener el mismo largo que frames_bgr.")

        probs_list: List[np.ndarray] = []
        photos: List[Dict[str, Any]] = []
        prep_total_ms = 0.0
        inf_total_ms = 0.0

        for i, frame in enumerate(frames_bgr):
            probs, prep_ms, inf_ms = self._infer_probs(frame)
            probs_list.append(probs)
            prep_total_ms += prep_ms
            inf_total_ms += inf_ms

            photo_result = aggregate_probabilities([probs], self.class_names)
            photo_result.pop("photos_agree")
            if photo_names is not None:
                photo_result = {"image_name": photo_names[i], **photo_result}
            photos.append(photo_result)

        result = aggregate_probabilities(probs_list, self.class_names)
        result.update({
            "photos_count": len(frames_bgr),
            "photos": photos,
            "latency_preprocess_ms": round(prep_total_ms, 2),
            "latency_inference_ms": round(inf_total_ms, 2),
            "latency_ms": round(prep_total_ms + inf_total_ms, 2),
        })
        return result

    def predict_cv2(self, frame_bgr: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Inferencia de alto rendimiento Zero-Copy para frames OpenCV en BGR.
        Mide por separado el tiempo de preprocesamiento y de inferencia pura.

        Retorna:
            (resultado_dict, frame_anotado)
        """
        probs_arr, prep_latency_ms, inf_latency_ms = self._infer_probs(frame_bgr)
        total_latency_ms = prep_latency_ms + inf_latency_ms
        best_idx = int(np.argmax(probs_arr))

        best_class = self.class_names[best_idx]
        best_conf = float(probs_arr[best_idx])

        result = {
            "predicted_class": best_class,
            "confidence": round(best_conf, 4),
            "confidence_percent": round(best_conf * 100.0, 2),
            "probabilities": {
                name: round(float(prob), 4) for name, prob in zip(self.class_names, probs_arr)
            },
            "latency_preprocess_ms": round(prep_latency_ms, 2),
            "latency_inference_ms": round(inf_latency_ms, 2),
            "latency_ms": round(total_latency_ms, 2),
        }

        # 3. Superposición visual (UI HUD)
        annotated = frame_bgr.copy()
        color = CLASS_COLORS_BGR.get(best_class, (200, 200, 200))
        h, w = annotated.shape[:2]

        # Barra superior informativa
        cv2.rectangle(annotated, (0, 0), (w, 55), (20, 20, 20), -1)
        cv2.rectangle(annotated, (0, 0), (12, 55), color, -1)

        text_title = f"Calidad: {best_class.upper()} ({result['confidence_percent']}%)"
        cv2.putText(annotated, text_title, (24, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

        text_sub = (
            f"Pre: {prep_latency_ms:.1f}ms | Inf: {inf_latency_ms:.1f}ms | "
            f"Total: {total_latency_ms:.1f}ms | Estacion 2 (Edge)"
        )
        cv2.putText(annotated, text_sub, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        return result, annotated

    def predict_pil(self, image: Any) -> Dict[str, Any]:
        """
        Soporte de compatibilidad para objetos PIL.Image.
        Convierte a BGR y deriva la llamada a predict_cv2 para consistencia total.
        """
        if not HAS_OPENCV:
            raise RuntimeError("OpenCV es requerido para realizar la inferencia.")
        rgb_arr = np.array(image.convert("RGB"))
        bgr_frame = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        result, _ = self.predict_cv2(bgr_frame)
        return result

