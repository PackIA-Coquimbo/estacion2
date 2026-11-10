"""
Suite de Pruebas Automatizadas - Fase 2: Motor de Inferencia Edge y Zero-Copy
(Estación 2: Análisis Visual de Calidad - OrangeQuality-AI)

Verifica todos los criterios de la Definición de Terminado (DoD) de la Fase 2:
1. Preprocesamiento Zero-Copy (preprocess_cv2_fast) con latencia <= 3.5 ms.
2. Función softmax_numpy numéricamente estable.
3. Inferencia con sesión ONNX optimizada (4 hilos, graph optimization all).
4. Estructura y formato del diccionario de resultados de inferencia.
"""

from pathlib import Path
import sys
import time
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from src.inference import (
    preprocess_cv2_fast,
    softmax_numpy,
    QualityPredictor,
    DEFAULT_CLASSES,
)


def test_preprocess_fast_output_properties():
    """Valida dimensiones, tipo y continuidad en memoria del preprocesamiento."""
    print("[1/4] Verificando propiedades del tensor generado por preprocess_cv2_fast()...")
    dummy_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    tensor = preprocess_cv2_fast(dummy_bgr, target_size=224)

    assert tensor.shape == (1, 3, 224, 224), f"Forma inesperada: {tensor.shape}"
    assert tensor.dtype == np.float32, f"Tipo de dato inesperado: {tensor.dtype}"
    assert tensor.flags.c_contiguous, "El tensor debe ser contiguo en memoria (C-contiguous)."
    assert not np.isnan(tensor).any(), "El tensor contiene valores NaN."
    assert not np.isinf(tensor).any(), "El tensor contiene valores Infinitos."
    print("  -> Propiedades del tensor validadas correctamente (1, 3, 224, 224) float32 contiguo.")


def test_preprocess_latency():
    """Valida el criterio DoD de latencia de preprocesamiento <= 3.5 ms."""
    print("[2/4] Evaluando latencia de preprocess_cv2_fast (DoD <= 3.5 ms)...")
    dummy_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(15):
        _ = preprocess_cv2_fast(dummy_bgr)

    latencies: list[float] = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = preprocess_cv2_fast(dummy_bgr)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    print(f"  -> Latencia promedio: {avg_lat:.2f} ms | Percentil 95: {p95_lat:.2f} ms")
    assert avg_lat <= 3.5, f"La latencia promedio ({avg_lat:.2f} ms) supera el umbral de 3.5 ms."
    print(f"  -> Criterio de latencia cumplido exitosamente ({avg_lat:.2f} ms <= 3.5 ms).")


def test_softmax_numpy():
    """Valida que softmax_numpy sea numéricamente estable y sume 1.0."""
    print("[3/4] Verificando estabilidad y cálculo de softmax_numpy()...")
    # Prueba estándar
    logits = np.array([[2.0, 1.0, 0.1]], dtype=np.float32)
    probs = softmax_numpy(logits)
    assert np.isclose(np.sum(probs), 1.0, atol=1e-5), "Las probabilidades deben sumar 1.0."
    assert np.argmax(probs) == 0, "La clase 0 debe tener la máxima probabilidad."

    # Prueba con números extremos (estabilidad numérica frente a overflow)
    extreme_logits = np.array([[1000.0, 1005.0, 990.0]], dtype=np.float32)
    extreme_probs = softmax_numpy(extreme_logits)
    assert not np.isnan(extreme_probs).any(), "Softmax generó NaN con valores grandes."
    assert np.isclose(np.sum(extreme_probs), 1.0, atol=1e-5), "Softmax extremo debe sumar 1.0."
    assert np.argmax(extreme_probs) == 1, "La clase 1 debe ser la de mayor probabilidad."
    print("  -> Cálculo de softmax numéricamente estable verificado.")


def test_onnx_predictor_edge():
    """Valida la inferencia completa en ONNX Runtime con modelo cuantizado INT8."""
    print("[4/4] Verificando inferencia completa con QualityPredictor en INT8...")
    model_path = PROJECT_ROOT / "weights" / "orange_model_int8.onnx"
    assert model_path.exists(), f"El modelo INT8 debe existir: {model_path}"

    predictor = QualityPredictor(model_path=model_path, use_onnx=True, num_threads=4)
    dummy_frame = np.full((480, 640, 3), (35, 35, 35), dtype=np.uint8)
    cv2.circle(dummy_frame, (320, 240), 120, (20, 140, 235), -1)

    result, annotated = predictor.predict_cv2(dummy_frame)

    assert "predicted_class" in result
    assert result["predicted_class"] in DEFAULT_CLASSES
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0.0 <= result["confidence_percent"] <= 100.0
    assert "latency_preprocess_ms" in result
    assert "latency_inference_ms" in result
    assert "latency_ms" in result
    assert annotated.shape == dummy_frame.shape

    print(f"  -> Predicción: {result['predicted_class']} ({result['confidence_percent']}%)")
    print(f"  -> Latencias: Pre {result['latency_preprocess_ms']}ms | Inf {result['latency_inference_ms']}ms | Total {result['latency_ms']}ms")
    print("  -> Predictor validado exitosamente.")


def run_all_tests():
    print("=" * 65)
    print("EJECUTANDO TEST SUITE - FASE 2: EDGE & ZERO-COPY")
    print("=" * 65)
    test_preprocess_fast_output_properties()
    test_preprocess_latency()
    test_softmax_numpy()
    test_onnx_predictor_edge()
    print("=" * 65)
    print("TODAS LAS PRUEBAS DE LA FASE 2 PASARON EXITOSAMENTE (DoD CUMPLIDO)")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()

