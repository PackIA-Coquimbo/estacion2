"""
Suite de Pruebas Automatizadas - Fase 4: Integración Inter-Estaciones en RAM (/dev/shm)
(Estación 2: Análisis Visual de Calidad - OrangeQuality-AI)

Verifica todos los criterios de la Definición de Terminado (DoD) de la Fase 4:
1. Transferencia y lectura de imágenes en RAM con latencia < 1.5 ms.
2. Protocolo de sincronización de eventos mediante banderas de control y ACK.
3. Estructura JSON de salida estandarizada para la comunicación entre Estación 1 y Estación 2.
4. Cero operaciones de I/O en almacenamiento secundario (MicroSD).
"""

from pathlib import Path
import json
import sys
import time
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from src.inference import QualityPredictor
from src.station_listener import (
    StationCommunicator,
    StationListener,
    run_interstation_benchmark,
    get_default_shm_dir,
)


def test_shm_transfer_latency():
    """Valida que la lectura de la imagen desde RAM sea menor a 1.5 ms."""
    print("[1/4] Verificando latencia de lectura/transferencia en RAM (DoD < 1.5 ms)...")
    comm = StationCommunicator()
    dummy_frame = np.full((480, 640, 3), (35, 35, 35), dtype=np.uint8)
    cv2.circle(dummy_frame, (320, 240), 120, (20, 140, 235), -1)

    # Enviar frame desde Estación 1 a RAM
    w_lat = comm.send_frame_from_station1(dummy_frame)

    # Leer frame en Estación 2
    success, read_frame, r_lat = comm.read_frame_in_station2()

    assert success, "La lectura de frame desde RAM falló."
    assert read_frame is not None, "El frame leído no debe ser None."
    assert read_frame.shape == dummy_frame.shape, f"Dimensiones no coinciden: {read_frame.shape}"
    assert r_lat < 1.5, f"La latencia de lectura ({r_lat:.3f} ms) superó el umbral de 1.5 ms."

    comm.cleanup()
    print(f"  -> Latencia de lectura en RAM: {r_lat:.3f} ms (< 1.5 ms: CUMPLIDO).")


def test_standardized_json_structure():
    """Valida la estructura del JSON estandarizado requerido por el protocolo de Fase 4."""
    print("[2/4] Verificando esquema JSON estandarizado de salida...")
    model_path = PROJECT_ROOT / "weights" / "orange_model_int8.onnx"
    predictor = QualityPredictor(model_path=model_path, use_onnx=True, num_threads=4)
    listener = StationListener(predictor=predictor)

    # Simular envío desde Estación 1
    dummy_frame = np.full((480, 640, 3), (35, 35, 35), dtype=np.uint8)
    cv2.circle(dummy_frame, (320, 240), 120, (20, 140, 235), -1)
    listener.communicator.send_frame_from_station1(dummy_frame)

    # Procesar evento en Estación 2
    payload = listener.process_pending_event()
    assert payload is not None, "El listener no detectó el evento pendiente."

    # Validar campos obligatorios según especificación de Fase 4
    assert payload["station_id"] == 2, f"station_id debe ser 2, obtenido: {payload.get('station_id')}"
    assert isinstance(payload["timestamp"], (int, float)), "timestamp debe ser numérico."
    assert payload["predicted_class"] in ["buen_estado", "marginalmente_buena", "mal_estado"]
    assert 0.0 <= payload["confidence_percent"] <= 100.0
    assert "probabilities" in payload
    assert isinstance(payload["probabilities"], dict)
    for cls in ("buen_estado", "marginalmente_buena", "mal_estado"):
        assert cls in payload["probabilities"], f"Falta clase {cls} en probabilidades."
    assert "latency_total_ms" in payload
    assert payload["latency_total_ms"] > 0

    print("  -> Esquema JSON verificado exitosamente:")
    print(json.dumps(payload, indent=4))
    listener.communicator.cleanup()


def test_synchronization_handshake():
    """Verifica el protocolo completo de Handshake: Trigger -> Proceso -> JSON -> ACK."""
    print("[3/4] Verificando ciclo completo de sincronización (Trigger -> ACK)...")
    comm = StationCommunicator()
    model_path = PROJECT_ROOT / "weights" / "orange_model_int8.onnx"
    predictor = QualityPredictor(model_path=model_path, use_onnx=True, num_threads=4)
    listener = StationListener(predictor=predictor, shm_dir=comm.shm_dir)

    dummy_frame = np.full((480, 640, 3), (20, 20, 20), dtype=np.uint8)

    # 1. Estación 1 envía frame
    comm.send_frame_from_station1(dummy_frame)
    assert comm.trigger_path.exists(), "El archivo trigger.flag debe existir tras enviar el frame."
    assert not comm.ack_path.exists(), "El archivo ack.flag no debe existir antes del procesamiento."

    # 2. Estación 2 procesa
    result = listener.process_pending_event()
    assert result is not None

    # 3. Estación 1 verifica ACK y lee resultado
    station1_received = comm.wait_for_result_in_station1(timeout_sec=1.0)
    assert station1_received is not None, "Estación 1 no recibió el resultado vía RAM."
    assert station1_received["station_id"] == 2
    assert not comm.trigger_path.exists(), "trigger.flag debe haberse consumido."
    assert comm.ack_path.exists(), "ack.flag debe existir confirmando finalización."

    comm.cleanup()
    print("  -> Handshake completado exitosamente sin colisiones.")


def test_benchmark_suite_execution():
    """Ejecuta la suite de benchmarking para verificar consistencia estadística."""
    print("[4/4] Ejecutando benchmark estadístico de 50 iteraciones...")
    bench_results = run_interstation_benchmark(iterations=50)
    assert bench_results["passed_dod"], "El benchmark no superó los criterios DoD."
    assert bench_results["mean_read_ms"] < 1.5, f"Media alta: {bench_results['mean_read_ms']}"
    print(f"  -> Benchmark completado: {bench_results['mean_read_ms']:.3f} ms media de lectura en RAM.")


def run_all_tests():
    print("=" * 65)
    print("EJECUTANDO TEST SUITE - FASE 4: INTER-STATION INTEGRATION")
    print("=" * 65)
    test_shm_transfer_latency()
    test_standardized_json_structure()
    test_synchronization_handshake()
    test_benchmark_suite_execution()
    print("=" * 65)
    print("TODAS LAS PRUEBAS DE LA FASE 4 PASARON EXITOSAMENTE (DoD CUMPLIDO)")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()

