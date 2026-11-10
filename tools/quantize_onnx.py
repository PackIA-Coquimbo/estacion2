"""
Herramienta de Cuantización a INT8 para Modelos ONNX
(Estación 2: Análisis Visual de Calidad - OrangeQuality-AI)

Optimiza y cuantiza modelos ONNX FP32 a INT8 para despliegue de alto rendimiento
en la CPU ARM de una Raspberry Pi 4 / 5.

Reduce el tamaño del modelo en un ~75% (< 6 MB) manteniendo una discrepancia de precisión < 1.5%.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import argparse
import sys
import os
import numpy as np

# Asegurar raíz del proyecto en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    quantize_dynamic,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)

from src.dataset_loader import create_dataloaders, SyntheticOrangeDataset, get_val_transforms


class DatasetCalibrationDataReader(CalibrationDataReader):
    """
    Proveedor de datos de calibración para la cuantización estática INT8.
    Utiliza imágenes reales del dataset o muestras sintéticas si no hay imágenes recolectadas.
    """
    def __init__(self, input_name: str = "input_image", max_samples: int = 50, dataset_dir: Optional[Path] = None):
        super().__init__()
        self.input_name = input_name
        if dataset_dir is None:
            dataset_dir = PROJECT_ROOT / "dataset"

        dataloaders = create_dataloaders(
            dataset_dir=dataset_dir,
            batch_size=1,
            fallback_to_synthetic=True,
            return_weights=False,
        )
        self.samples: List[np.ndarray] = []
        for loader_key in ("val", "train", "test"):
            loader = dataloaders.get(loader_key)
            if loader is None:
                continue
            for images, _ in loader:
                self.samples.append(images.numpy().astype(np.float32))
                if len(self.samples) >= max_samples:
                    break
            if len(self.samples) >= max_samples:
                break

        # Si aún faltan muestras, generarlas con el pipeline sintético oficial
        if len(self.samples) < max_samples:
            synth_ds = SyntheticOrangeDataset(
                num_samples_per_class=int(np.ceil((max_samples - len(self.samples)) / 3)) + 2,
                transform=get_val_transforms(224),
            )
            for img, _ in synth_ds:
                self.samples.append(img.unsqueeze(0).numpy().astype(np.float32))
                if len(self.samples) >= max_samples:
                    break

        self.iterator = iter(self.samples)
        print(f"[CALIBRACIÓN] Lector preparado con {len(self.samples)} muestras para calibrar rangos INT8.")

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        batch = next(self.iterator, None)
        if batch is None:
            return None
        return {self.input_name: batch}

    def rewind(self) -> None:
        self.iterator = iter(self.samples)


def validate_onnx_model(model_path: Path) -> bool:
    """Verifica la validez estructural y semántica del modelo ONNX usando onnx.checker."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Archivo ONNX no encontrado: {model_path}")

    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    print(f"[CHECKER] Validación exitosa con onnx.checker: {model_path.name}")
    return True


def get_file_size_mb(path: Path) -> float:
    """Retorna el tamaño en Megabytes (MB)."""
    return Path(path).stat().st_size / (1024 * 1024)


def quantize_onnx(
    input_path: Path,
    output_path: Path,
    mode: str = "static",
    max_calibration_samples: int = 50,
) -> Path:
    """
    Cuantiza un modelo ONNX FP32 a INT8.

    Args:
        input_path: Ruta al archivo .onnx original (FP32).
        output_path: Ruta al archivo .onnx de destino (INT8).
        mode: 'static' (QDQ Conv+Linear, ~1.17 MB) o 'dynamic' (Linear, ~4.5 MB).
        max_calibration_samples: Muestras para calibración en modo static.

    Returns:
        Ruta del archivo cuantizado.
    """
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CUANTIZACIÓN A INT8 PARA RASPBERRY PI - ORANGEQUALITY-AI")
    print("=" * 70)
    print(f"Modelo FP32 de entrada:  {input_path}")
    print(f"Modelo INT8 de salida:   {output_path}")
    print(f"Modo de cuantización:    {mode.upper()}")

    # 1. Validar modelo de entrada
    validate_onnx_model(input_path)

    # 2. Obtener nombre del nodo de entrada
    session_preview = ort.InferenceSession(str(input_path), providers=["CPUExecutionProvider"])
    input_name = session_preview.get_inputs()[0].name

    # 3. Ejecutar cuantización
    if mode == "static":
        print(f"\n[CUANTIZACIÓN] Aplicando quantize_static (QDQ - Conv + Gemm)...")
        data_reader = DatasetCalibrationDataReader(
            input_name=input_name,
            max_samples=max_calibration_samples,
        )
        quantize_static(
            model_input=str(input_path),
            model_output=str(output_path),
            calibration_data_reader=data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=False,
            reduce_range=False,
        )
    else:
        print(f"\n[CUANTIZACIÓN] Aplicando quantize_dynamic (MatMul / Gemm)...")
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
            per_channel=False,
            reduce_range=False,
        )

    # 4. Validar modelo de salida
    validate_onnx_model(output_path)

    # 5. Métricas de compresión
    size_fp32 = get_file_size_mb(input_path)
    size_int8 = get_file_size_mb(output_path)
    reduction = ((size_fp32 - size_int8) / size_fp32) * 100 if size_fp32 > 0 else 0

    print("-" * 70)
    print(f"Tamaño FP32 original:   {size_fp32:.2f} MB")
    print(f"Tamaño INT8 cuantizado: {size_int8:.2f} MB")
    print(f"Factor de compresión:   {reduction:.1f}% de reducción")
    print("-" * 70)

    if size_int8 < 6.0:
        print(f"[DoD CUMPLIDO] Peso del archivo ({size_int8:.2f} MB) es inferior al límite de 6.0 MB.")
    else:
        print(f"[DoD ALERTA] Peso del archivo ({size_int8:.2f} MB) supera 6.0 MB.")

    return output_path


def compare_precision(
    fp32_model_path: Path,
    int8_model_path: Path,
    num_samples: int = 100,
    tolerance_pct: float = 1.5,
) -> Dict[str, Any]:
    """
    Evalúa cuantitativamente la pérdida de precisión entre el modelo FP32 y el modelo INT8.
    Verifica que la diferencia de exactitud en datos de prueba sea inferior al 1.5% (DoD de Fase 1).
    """
    print("\n" + "=" * 70)
    print("VALIDACIÓN DE PRECISIÓN EN DATOS DE PRUEBA: FP32 VS INT8 (CRITERIO DoD)")
    print("=" * 70)

    session_fp32 = ort.InferenceSession(str(fp32_model_path), providers=["CPUExecutionProvider"])
    session_int8 = ort.InferenceSession(str(int8_model_path), providers=["CPUExecutionProvider"])

    input_name_fp32 = session_fp32.get_inputs()[0].name
    input_name_int8 = session_int8.get_inputs()[0].name

    # Cargar datos de prueba
    dataloaders = create_dataloaders(
        dataset_dir=PROJECT_ROOT / "dataset",
        batch_size=1,
        fallback_to_synthetic=True,
    )

    test_data: List[Tuple[np.ndarray, int]] = []
    for images, targets in dataloaders["test"]:
        test_data.append((images.numpy().astype(np.float32), targets.item()))

    for images, targets in dataloaders["val"]:
        test_data.append((images.numpy().astype(np.float32), targets.item()))

    if len(test_data) < num_samples:
        synth_ds = SyntheticOrangeDataset(
            num_samples_per_class=int(np.ceil((num_samples - len(test_data)) / 3)) + 5,
            transform=get_val_transforms(224),
        )
        for img, target in synth_ds:
            test_data.append((img.unsqueeze(0).numpy().astype(np.float32), target))
            if len(test_data) >= num_samples:
                break

    test_data = test_data[:num_samples]

    correct_fp32 = 0
    correct_int8 = 0
    matches = 0
    max_abs_diff = 0.0
    cosine_sims: List[float] = []

    for sample_x, target_y in test_data:
        out_fp32 = session_fp32.run(None, {input_name_fp32: sample_x})[0]
        out_int8 = session_int8.run(None, {input_name_int8: sample_x})[0]

        pred_fp32 = int(np.argmax(out_fp32, axis=1)[0])
        pred_int8 = int(np.argmax(out_int8, axis=1)[0])

        if pred_fp32 == target_y:
            correct_fp32 += 1
        if pred_int8 == target_y:
            correct_int8 += 1

        if pred_fp32 == pred_int8:
            matches += 1

        abs_diff = float(np.max(np.abs(out_fp32 - out_int8)))
        if abs_diff > max_abs_diff:
            max_abs_diff = abs_diff

        v1 = out_fp32.flatten()
        v2 = out_int8.flatten()
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
        cosine_sims.append(float(np.dot(v1, v2) / denom))

    total = len(test_data)
    acc_fp32 = (correct_fp32 / total) * 100.0
    acc_int8 = (correct_int8 / total) * 100.0
    acc_diff = abs(acc_fp32 - acc_int8)
    concordance_pct = (matches / total) * 100.0
    mean_cosine_sim = float(np.mean(cosine_sims))

    print(f"Muestras de prueba evaluadas:           {total}")
    print(f"Exactitud Modelo FP32:                  {acc_fp32:.2f}% ({correct_fp32}/{total})")
    print(f"Exactitud Modelo INT8:                  {acc_int8:.2f}% ({correct_int8}/{total})")
    print(f"Diferencia de exactitud (|FP32 - INT8|):{acc_diff:.2f}%")
    print(f"Concordancia entre predicciones:        {concordance_pct:.2f}% ({matches}/{total})")
    print(f"Máxima diferencia absoluta logits:      {max_abs_diff:.6f}")
    print(f"Similitud de Coseno promedio:           {mean_cosine_sim:.6f}")
    print("-" * 70)

    passed = acc_diff <= tolerance_pct
    if passed:
        print(f"[DoD CUMPLIDO] La diferencia de precisión ({acc_diff:.2f}%) es inferior al límite de {tolerance_pct}%.")
    else:
        print(f"[DoD FALLIDO] La diferencia ({acc_diff:.2f}%) supera la tolerancia de {tolerance_pct}%.")

    return {
        "num_samples": total,
        "acc_fp32": acc_fp32,
        "acc_int8": acc_int8,
        "acc_diff": acc_diff,
        "concordance_pct": concordance_pct,
        "max_abs_diff": max_abs_diff,
        "mean_cosine_sim": mean_cosine_sim,
        "passed_dod": passed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cuantizador INT8 de Modelo ONNX para Raspberry Pi (OrangeQuality-AI)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "weights" / "orange_model.onnx"),
        help="Ruta al modelo ONNX FP32",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "weights" / "orange_model_int8.onnx"),
        help="Ruta de guardado para el modelo ONNX INT8",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["static", "dynamic"],
        default="static",
        help="Modo de cuantización: 'static' (QDQ completo, tamaño ~1.17 MB) o 'dynamic' (MatMul/Gemm)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Ejecuta la validación de precisión frente a FP32",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Número de muestras de prueba para la validación del DoD",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    quantize_onnx(
        input_path=input_path,
        output_path=output_path,
        mode=args.mode,
    )

    if args.verify:
        compare_precision(
            fp32_model_path=input_path,
            int8_model_path=output_path,
            num_samples=args.samples,
        )


if __name__ == "__main__":
    main()
