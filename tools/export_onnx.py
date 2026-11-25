"""
Exportación del modelo PyTorch (.pth) a ONNX para despliegue en Edge
(Estación 2: Análisis Visual de Calidad - OrangeQuality-AI)

Uso:
    python tools/export_onnx.py --weights weights/best_orange_model.pth --output weights/orange_model.onnx
"""

from pathlib import Path
from typing import Union
import argparse
import inspect
import sys

# Asegurar raíz del proyecto en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.classes import DEFAULT_CLASSES
from src.model import build_model


def export_to_onnx(
    weights_path: Union[str, Path],
    onnx_output_path: Union[str, Path],
    num_classes: int = len(DEFAULT_CLASSES),
    device: str = "cpu",
) -> Path:
    """Convierte el modelo de PyTorch (.pth) a formato ONNX optimizado."""
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
    parser = argparse.ArgumentParser(description="Exporta el modelo PyTorch (.pth) a ONNX")
    parser.add_argument("--weights", type=str, default=str(PROJECT_ROOT / "weights" / "best_orange_model.pth"))
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "weights" / "orange_model.onnx"))
    args = parser.parse_args()
    export_to_onnx(args.weights, args.output)


if __name__ == "__main__":
    main()
