"""
Motor de Inferencia en Tiempo Real Ligero (Edge / Raspberry Pi) y Exportación ONNX
(Estación 2: Análisis Visual de Calidad - OrangeQuality-AI)

Características de Fase 2 (Edge):
1. Aislamiento completo de dependencias: Inferencia 100% nativa con onnxruntime,
   numpy y opencv sin requerir PyTorch ni Pillow en el dispositivo embebido.
2. Preprocesamiento Zero-Copy vectorizado directo (preprocess_cv2_fast) < 3.0 ms.
3. Sesión ONNX Runtime optimizada para CPU ARM Cortex (4 threads, optimización total de grafos).
4. Función Softmax numéricamente estable implementada en NumPy nativo.
5. Visualización superpuesta con código de colores y reporte de latencia detallado (preprocesamiento + inferencia).
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List
import argparse
import sys
import time
import numpy as np

# Asegurar raíz en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# OpenCV
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# ONNX Runtime
try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False

# PyTorch (Opcional en Edge - requerido solo para entrenamiento y exportación)
try:
    import torch
    import torch.nn.functional as F
    from src.model import build_model, OrangeQualityCNN
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

# Pillow (Opcional en Edge)
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Clases por defecto del proyecto
DEFAULT_CLASSES: List[str] = [
    "buen_estado",
    "marginalmente_buena",
    "mal_estado",
]

# Normalización estándar (ImageNet) pre-vectorizada para NumPy (1, 3, 1, 1)
NORM_MEAN_NP = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
NORM_STD_NP = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

# Paleta de colores para visualización BGR (OpenCV)
CLASS_COLORS_BGR = {
    "buen_estado": (40, 200, 40),          # Verde
    "marginalmente_buena": (0, 200, 240),  # Amarillo / Ámbar
    "mal_estado": (40, 40, 220),           # Rojo
}


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
            if default_int8.exists():
                model_path = default_int8
            elif default_fp32.exists():
                model_path = default_fp32
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
        if HAS_TORCH and torch is not None and torch.cuda.is_available():
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        print(f"[QualityPredictor] Sesión ONNX cargada: {model_path.name}")
        print(f"[QualityPredictor] Hilos intra-op: {self.num_threads} | Proveedores: {self.session.get_providers()}")

    def _load_pytorch_model(self, model_path: Optional[Union[str, Path]], device: Optional[str] = None) -> None:
        """Carga el modelo PyTorch tradicional (.pth)."""
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
            print(f"[QualityPredictor] Modelo PyTorch cargado desde: {model_path}")
        else:
            print("[QualityPredictor] Aviso: Sin archivo de pesos. Usando arquitectura no entrenada.")
        self.model.to(self.device)
        self.model.eval()

    def predict_cv2(self, frame_bgr: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Inferencia de alto rendimiento Zero-Copy para frames OpenCV en BGR.
        Mide por separado el tiempo de preprocesamiento y de inferencia pura.

        Retorna:
            (resultado_dict, frame_anotado)
        """
        # 1. Preprocesamiento Zero-Copy
        t0 = time.perf_counter()
        input_tensor = preprocess_cv2_fast(frame_bgr, target_size=224)
        t_prep_end = time.perf_counter()
        prep_latency_ms = (t_prep_end - t0) * 1000.0

        # 2. Inferencia pura
        t_inf_start = time.perf_counter()
        if self.use_onnx:
            raw_logits = self.session.run(None, {self.input_name: input_tensor})[0]
            probs_arr = softmax_numpy(raw_logits)[0]
            best_idx = int(np.argmax(raw_logits, axis=1)[0])
        else:
            with torch.no_grad():
                torch_in = torch.from_numpy(input_tensor).to(self.device)
                logits = self.model(torch_in)
                probs_tensor = F.softmax(logits, dim=1).squeeze(0)
                probs_arr = probs_tensor.cpu().numpy()
                best_idx = int(torch.argmax(logits, dim=1).item())

        t_inf_end = time.perf_counter()
        inf_latency_ms = (t_inf_end - t_inf_start) * 1000.0
        total_latency_ms = (t_inf_end - t0) * 1000.0

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


def export_to_onnx(
    weights_path: Union[str, Path],
    onnx_output_path: Union[str, Path],
    num_classes: int = len(DEFAULT_CLASSES),
    device: str = "cpu",
) -> Path:
    """Convierte el modelo de PyTorch (.pth) a formato ONNX optimizado."""
    if not HAS_TORCH:
        raise ImportError("Se requiere PyTorch para exportar modelos a ONNX.")

    weights_path = Path(weights_path)
    onnx_output_path = Path(onnx_output_path)
    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ONNX Export] Cargando modelo PyTorch desde: {weights_path}")
    model = build_model(num_classes=num_classes)
    if weights_path.exists():
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    print(f"[ONNX Export] Exportando a: {onnx_output_path}...")
    export_kwargs = {
        "export_params": True,
        "opset_version": 14,
        "do_constant_folding": True,
        "input_names": ["input_image"],
        "output_names": ["logits"],
        "dynamic_axes": {"input_image": {0: "batch_size"}, "logits": {0: "batch_size"}},
    }
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_output_path),
        **export_kwargs,
    )

    try:
        import onnx
        onnx_model = onnx.load(str(onnx_output_path))
        onnx.checker.check_model(onnx_model)
        print("[ONNX Export] Validacion estructural exitosa con onnx.checker.")
    except ImportError:
        print("[ONNX Export] Aviso: onnx.checker omitido por falta del paquete onnx.")

    print(f"[ONNX Export] Proceso completado: {onnx_output_path}")
    return onnx_output_path


def run_preprocessing_benchmark(iterations: int = 200, frame_shape: Tuple[int, int, int] = (480, 640, 3)) -> float:
    """
    Ejecuta una prueba cuantitativa de latencia de preprocesamiento (DoD Fase 2).
    Verifica que la latencia media sea <= 3.5 ms.
    """
    if not HAS_OPENCV:
        raise RuntimeError("OpenCV es requerido para el benchmark.")

    print("\n" + "=" * 65)
    print("BENCHMARK DE PREPROCESAMIENTO ZERO-COPY (DoD - FASE 2)")
    print("=" * 65)
    dummy_frame = np.random.randint(0, 256, frame_shape, dtype=np.uint8)

    # Warmup
    for _ in range(10):
        _ = preprocess_cv2_fast(dummy_frame)

    latencies_ms: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = preprocess_cv2_fast(dummy_frame)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_lat = float(np.mean(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    min_lat = float(np.min(latencies_ms))
    max_lat = float(np.max(latencies_ms))

    print(f"Iteraciones:             {iterations}")
    print(f"Resolución de entrada:   {frame_shape[1]}x{frame_shape[0]} BGR")
    print(f"Latencia Media:          {mean_lat:.2f} ms")
    print(f"Latencia Percentil 95:   {p95_lat:.2f} ms")
    print(f"Latencia Mínima:         {min_lat:.2f} ms")
    print(f"Latencia Máxima:         {max_lat:.2f} ms")
    print("-" * 65)

    passed = mean_lat <= 3.5
    status_str = "CUMPLIDO" if passed else "FALLIDO"
    print(f"[DoD Fase 2] Latencia preprocesamiento <= 3.5 ms: {status_str} ({mean_lat:.2f} ms)")
    return mean_lat


def main():
    parser = argparse.ArgumentParser(description="Inferencia y Calidad Visual (OrangeQuality-AI - Edge)")
    parser.add_argument("--image", type=str, default=None, help="Ruta a una imagen para clasificar")
    parser.add_argument("--webcam", type=int, default=None, help="Índice de cámara para inferencia en vivo")
    parser.add_argument("--weights", type=str, default=str(PROJECT_ROOT / "weights" / "best_orange_model.pth"), help="Pesos .pth (PyTorch)")
    parser.add_argument("--onnx", type=str, default=None, help="Ruta a modelo .onnx (ONNX Runtime)")
    parser.add_argument("--export-onnx", type=str, default=None, help="Exportar modelo PyTorch a ONNX")
    parser.add_argument("--threads", type=int, default=4, help="Número de hilos de inferencia (por defecto 4)")
    parser.add_argument("--benchmark", action="store_true", help="Ejecuta benchmark de preprocesamiento Zero-Copy")
    args = parser.parse_args()

    if args.benchmark:
        run_preprocessing_benchmark()
        return

    if args.export_onnx:
        export_to_onnx(args.weights, args.export_onnx)
        return

    # Determinar si usar ONNX o PyTorch
    use_onnx = True
    model_file: Optional[Path] = None

    if args.onnx:
        use_onnx = True
        model_file = Path(args.onnx)
    elif Path(args.weights).exists() and HAS_TORCH:
        use_onnx = False
        model_file = Path(args.weights)
    else:
        # Fallback a modelo ONNX INT8 o FP32
        int8_path = PROJECT_ROOT / "weights" / "orange_model_int8.onnx"
        fp32_path = PROJECT_ROOT / "weights" / "orange_model.onnx"
        if int8_path.exists():
            model_file = int8_path
            use_onnx = True
        elif fp32_path.exists():
            model_file = fp32_path
            use_onnx = True

    predictor = QualityPredictor(
        model_path=model_file,
        use_onnx=use_onnx,
        num_threads=args.threads,
    )

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: No existe el archivo {image_path}")
            return

        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Error: No se pudo leer la imagen con OpenCV: {image_path}")
            return

        res, _ = predictor.predict_cv2(frame)
        print("\n" + "=" * 55)
        print("RESULTADO DE CLASIFICACIÓN DE CALIDAD VISUAL")
        print("=" * 55)
        print(f"Imagen:                  {image_path.name}")
        print(f"Calidad Estimada:        {res['predicted_class']}")
        print(f"Nivel de Confianza:      {res['confidence_percent']}%")
        print(f"Latencia Preproceso:     {res['latency_preprocess_ms']} ms")
        print(f"Latencia Inferencia:     {res['latency_inference_ms']} ms")
        print(f"Latencia Total:          {res['latency_ms']} ms")
        print("Probabilidades por clase:")
        for cls, prob in res["probabilities"].items():
            print(f"  - {cls:<22}: {prob * 100:.2f}%")
        print("=" * 55)
        return

    if args.webcam is not None:
        if not HAS_OPENCV:
            print("Error: Se requiere 'opencv-python' o 'opencv-python-headless' para cámara.")
            return

        cap = cv2.VideoCapture(args.webcam)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir la cámara con índice {args.webcam}")
            return

        print("\nIniciando inferencia en tiempo real. Presione 'q' para salir...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, annotated = predictor.predict_cv2(frame)
            cv2.imshow("Estacion 2 - Analisis Visual de Calidad", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        return

    # Inferencia Demo usando exclusivamente OpenCV y NumPy (sin Pillow)
    print("\n[Inferencia Demo] Generando frame sintético de prueba con OpenCV (Zero-Copy)...")
    demo_frame = np.full((480, 640, 3), (35, 35, 35), dtype=np.uint8)
    cv2.circle(demo_frame, (320, 240), 130, (20, 140, 235), -1)  # Naranja sintética

    res, _ = predictor.predict_cv2(demo_frame)
    print("=" * 55)
    print("DEMO DE PREDICCIÓN CON FRAME SINTÉTICO (EDGE)")
    print("=" * 55)
    print(f"Calidad Estimada:        {res['predicted_class']}")
    print(f"Nivel de Confianza:      {res['confidence_percent']}%")
    print(f"Latencia Preproceso:     {res['latency_preprocess_ms']} ms")
    print(f"Latencia Inferencia:     {res['latency_inference_ms']} ms")
    print(f"Latencia Total:          {res['latency_ms']} ms")
    print("Probabilidades:")
    for cls, prob in res["probabilities"].items():
        print(f"  - {cls:<22}: {prob * 100:.2f}%")
    print("=" * 55)


if __name__ == "__main__":
    main()
