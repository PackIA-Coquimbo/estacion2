"""
Motor de Inferencia en Tiempo Real y Exportación ONNX
(Estación 2: Análisis Visual de Calidad)

Capacidades:
1. Inferencia sobre imagen estática, lote de imágenes o flujo de webcam (OpenCV / PIL).
2. Soporte dual: PyTorch directo (.pth) u ONNX Runtime (.onnx).
3. Salida estandarizada: Clase estimada + Nivel de Confianza (%).
4. Renderizado visual con código de colores:
   - buen_estado: Verde
   - marginalmente_buena: Amarillo / Ámbar
   - mal_estado: Rojo
5. Función de exportación de PyTorch a formato ONNX.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import argparse
import sys
import time

# Asegurar raíz en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

from src.model import build_model, OrangeQualityCNN, DEFAULT_CLASSES
from src.dataset_loader import get_val_transforms, NORM_MEAN, NORM_STD

# Paleta de colores para visualización BGR (OpenCV) y RGB (PIL)
CLASS_COLORS_BGR = {
    "buen_estado": (40, 200, 40),          # Verde
    "marginalmente_buena": (0, 200, 240),  # Amarillo / Ámbar
    "mal_estado": (40, 40, 220),           # Rojo
}

CLASS_COLORS_RGB = {
    "buen_estado": (40, 200, 40),
    "marginalmente_buena": (240, 200, 0),
    "mal_estado": (220, 40, 40),
}


class QualityPredictor:
    """
    Clase de inferencia optimizada para CPU o GPU integrada.
    Carga un modelo PyTorch (.pth) o un modelo ONNX (.onnx).
    """
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        use_onnx: bool = False,
    ):
        self.use_onnx = use_onnx
        self.class_names = DEFAULT_CLASSES
        self.transform = get_val_transforms(image_size=224)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.use_onnx:
            self._load_onnx_session(model_path)
        else:
            self._load_pytorch_model(model_path)

    def _load_pytorch_model(self, model_path: Optional[Union[str, Path]]) -> None:
        self.model = build_model(num_classes=len(self.class_names))
        if model_path is not None and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.class_names = checkpoint.get("class_names", self.class_names)
            else:
                self.model.load_state_dict(checkpoint)
            print(f"[QualityPredictor] Modelo PyTorch cargado desde: {model_path}")
        else:
            print("[QualityPredictor] Aviso: Sin archivo de pesos. Usando arquitectura no entrenada (modo prueba).")
        self.model.to(self.device)
        self.model.eval()

    def _load_onnx_session(self, model_path: Optional[Union[str, Path]]) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime no está instalado. Ejecute 'pip install onnxruntime'.")

        if model_path is None or not Path(model_path).exists():
            raise FileNotFoundError(f"Archivo ONNX no encontrado: {model_path}")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[QualityPredictor] Sesión ONNX cargada con proveedores: {self.session.get_providers()}")

    def predict_pil(self, image: Image.Image) -> Dict[str, Any]:
        """
        Realiza la predicción sobre un objeto PIL.Image.
        Retorna diccionario con:
            - predicted_class: str
            - confidence: float (0.0 a 1.0)
            - confidence_percent: float (0 a 100%)
            - probabilities: Dict[str, float]
            - latency_ms: float
        """
        start_time = time.perf_counter()
        rgb_image = image.convert("RGB")
        input_tensor = self.transform(rgb_image).unsqueeze(0)  # (1, 3, 224, 224)

        if self.use_onnx:
            np_input = input_tensor.numpy()
            raw_logits = self.session.run(None, {self.input_name: np_input})[0]
            logits = torch.from_numpy(raw_logits)
        else:
            with torch.no_grad():
                logits = self.model(input_tensor.to(self.device))

        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        best_idx = int(torch.argmax(logits, dim=1).item())
        best_class = self.class_names[best_idx]
        best_conf = probs[best_idx]
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "predicted_class": best_class,
            "confidence": round(best_conf, 4),
            "confidence_percent": round(best_conf * 100.0, 2),
            "probabilities": {
                name: round(prob, 4) for name, prob in zip(self.class_names, probs)
            },
            "latency_ms": round(latency_ms, 2),
        }

    def predict_cv2(self, frame_bgr) -> Tuple[Dict[str, Any], Any]:
        """
        Predice sobre un frame en formato BGR (OpenCV) y devuelve el resultado
        junto con el frame anotado con la interfaz visual requerida.
        """
        if not HAS_OPENCV:
            raise RuntimeError("OpenCV no está disponible. Instale 'opencv-python'.")

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        result = self.predict_pil(pil_img)

        # Anotación sobre el frame
        annotated = frame_bgr.copy()
        cls_name = result["predicted_class"]
        conf_pct = result["confidence_percent"]
        latency = result["latency_ms"]
        color = CLASS_COLORS_BGR.get(cls_name, (200, 200, 200))

        # Banner de información superior
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w, 55), (20, 20, 20), -1)
        cv2.rectangle(annotated, (0, 0), (12, 55), color, -1)

        text_title = f"Calidad: {cls_name.upper()} ({conf_pct}%)"
        cv2.putText(annotated, text_title, (24, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

        text_sub = f"Latencia: {latency:.1f}ms | Estacion 2: Calidad Visual"
        cv2.putText(annotated, text_sub, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        return result, annotated


def export_to_onnx(
    weights_path: Union[str, Path],
    onnx_output_path: Union[str, Path],
    num_classes: int = len(DEFAULT_CLASSES),
    device: str = "cpu",
) -> Path:
    """
    Convierte el modelo de PyTorch (.pth) a formato ONNX optimizado.
    """
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


def main():
    parser = argparse.ArgumentParser(description="Inferencia y Calidad Visual (OrangeQuality-AI)")
    parser.add_argument("--image", type=str, default=None, help="Ruta a una imagen para clasificar")
    parser.add_argument("--webcam", type=int, default=None, help="Índice de cámara para inferencia en vivo (ej: 0)")
    parser.add_argument("--weights", type=str, default=str(PROJECT_ROOT / "weights" / "best_orange_model.pth"), help="Pesos .pth")
    parser.add_argument("--onnx", type=str, default=None, help="Ruta a modelo .onnx")
    parser.add_argument("--export-onnx", type=str, default=None, help="Ruta destino para exportar modelo a ONNX")
    args = parser.parse_args()

    if args.export_onnx:
        export_to_onnx(args.weights, args.export_onnx)
        return

    predictor = QualityPredictor(
        model_path=args.onnx if args.onnx else args.weights,
        use_onnx=bool(args.onnx),
    )

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: No existe el archivo {image_path}")
            return
        pil_img = Image.open(image_path)
        res = predictor.predict_pil(pil_img)
        print("\n" + "=" * 50)
        print("RESULTADO DE CLASIFICACIÓN DE CALIDAD VISUAL")
        print("=" * 50)
        print(f"Imagen:              {image_path.name}")
        print(f"Calidad Estimada:    {res['predicted_class']}")
        print(f"Nivel de Confianza:  {res['confidence_percent']}%")
        print(f"Latencia:            {res['latency_ms']} ms")
        print("Probabilidades por clase:")
        for cls, prob in res["probabilities"].items():
            print(f"  - {cls:<22}: {prob * 100:.2f}%")
        print("=" * 50)
        return

    if args.webcam is not None:
        if not HAS_OPENCV:
            print("Error: Se requiere 'opencv-python' para usar la cámara web.")
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

    # Si no se pasó argumento, ejecutar prueba con una imagen sintética
    print("[Inferencia Demo] Generando imagen sintética de prueba...")
    demo_img = Image.new("RGB", (256, 256), color=(235, 140, 20))
    res = predictor.predict_pil(demo_img)
    print("\n" + "=" * 50)
    print("DEMO DE PREDICCIÓN CON IMAGEN SINTÉTICA")
    print("=" * 50)
    print(f"Calidad Estimada:    {res['predicted_class']}")
    print(f"Nivel de Confianza:  {res['confidence_percent']}%")
    print(f"Latencia:            {res['latency_ms']} ms")
    print("Probabilidades:")
    for cls, prob in res["probabilities"].items():
        print(f"  - {cls:<22}: {prob * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
