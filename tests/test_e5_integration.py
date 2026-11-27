"""
Pruebas de integración de Estación 2 con el contrato v3 de Estación 5 (E5).

Verifica:
1. Mapeo de clases locales (`buen_estado`, `marginalmente_buena`, `mal_estado`)
   a `calidad_visual`, `confianza_visual` [0..1] y `defectos_detectados`.
2. Estructura exacta del payload `POST /api/v1/ingest/e2` y timestamp ISO 8601 con zona horaria.
3. Cliente HTTP `E5Client`: 201 Created, reintento idempotente con mismo JSON/timestamp tras timeout/503,
   y no-reintento en 409 Conflict / 422 Unprocessable Entity.
4. Flujo completo `StationListener` leyendo manifiesto E1 (`id_lote`, `id_fruta`) desde RAM
   y enviando `POST /api/v1/ingest/e2` a un servidor HTTP local simulado.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer as HTTPServer
from pathlib import Path
import json
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.e5_client import (
    E2_INGEST_PATH,
    E5Client,
    build_e2_payload,
    map_prediction_to_e2_datos,
)


class _MockE5Handler(BaseHTTPRequestHandler):
    """Servidor HTTP local que simula las respuestas de E5."""

    received_bodies = []
    received_paths = []
    behavior = "ok"
    attempts = 0

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        type(self).received_paths.append(self.path)
        type(self).received_bodies.append(raw)
        type(self).attempts += 1

        if type(self).behavior == "timeout_then_ok" and type(self).attempts == 1:
            time.sleep(0.35)
            return

        if type(self).behavior == "503_then_ok" and type(self).attempts == 1:
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b'{"detail":"unavailable"}')
            return

        if type(self).behavior == "422":
            self.send_response(422)
            self.end_headers()
            self.wfile.write(b'{"detail":"invalid payload"}')
            return

        if type(self).behavior == "409":
            self.send_response(409)
            self.end_headers()
            self.wfile.write(b'{"detail":"id_fruta conflict"}')
            return

        duplicado = type(self).attempts > 1 and type(self).behavior == "timeout_then_ok"
        resp = json.dumps({"lectura_id": 1, "duplicado": duplicado}).encode("utf-8")
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, format, *args):
        return


def _start_mock_server():
    _MockE5Handler.received_bodies = []
    _MockE5Handler.received_paths = []
    _MockE5Handler.behavior = "ok"
    _MockE5Handler.attempts = 0
    server = HTTPServer(("127.0.0.1", 0), _MockE5Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


class TestE5Integration(unittest.TestCase):
    def test_class_mapping_and_payload_v3(self):
        self.assertEqual(
            map_prediction_to_e2_datos("buen_estado", confidence=0.92),
            {"calidad_visual": "alta", "confianza_visual": 0.92, "defectos_detectados": []},
        )
        self.assertEqual(
            map_prediction_to_e2_datos("marginalmente_buena", confidence_percent=75.5),
            {
                "calidad_visual": "media",
                "confianza_visual": 0.755,
                "defectos_detectados": ["defecto_leve"],
            },
        )
        self.assertEqual(
            map_prediction_to_e2_datos("mal_estado", confidence=0.88),
            {
                "calidad_visual": "baja",
                "confianza_visual": 0.88,
                "defectos_detectados": ["defecto_visible"],
            },
        )

        payload = build_e2_payload(
            id_lote="LOTE-2026-11-A",
            id_fruta="M-0001",
            predicted_class="buen_estado",
            confidence=0.92,
        )
        self.assertEqual(payload["id_lote"], "LOTE-2026-11-A")
        self.assertEqual(payload["id_fruta"], "M-0001")
        self.assertEqual(payload["estacion"], "E2")
        self.assertTrue(payload["timestamp"].endswith("Z"))
        self.assertEqual(payload["datos"]["calidad_visual"], "alta")
        self.assertEqual(payload["datos"]["confianza_visual"], 0.92)
        self.assertEqual(payload["datos"]["defectos_detectados"], [])

    def test_e5_client_201_and_idempotent_retry_on_timeout(self):
        server = _start_mock_server()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            client = E5Client(base_url=base_url, timeout_sec=0.15, max_retries=2, backoff_base_sec=0.05)

            _MockE5Handler.behavior = "timeout_then_ok"
            payload = build_e2_payload("LOTE-2026-11-A", "M-0001", "buen_estado", confidence=0.92)
            res = client.send_e2_ingest(payload)

            self.assertTrue(res["ok"])
            self.assertEqual(res["status_code"], 201)
            self.assertEqual(res["attempts"], 2)
            self.assertEqual(len(_MockE5Handler.received_bodies), 2)
            self.assertEqual(_MockE5Handler.received_paths[0], E2_INGEST_PATH)
            # Idempotencia: exactamente el mismo JSON y mismo timestamp en el reintento
            self.assertEqual(_MockE5Handler.received_bodies[0], _MockE5Handler.received_bodies[1])
        finally:
            server.shutdown()
            server.server_close()

    def test_e5_client_no_retry_on_409_and_422(self):
        server = _start_mock_server()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            client = E5Client(base_url=base_url, timeout_sec=1.0, max_retries=3, backoff_base_sec=0.01)

            for code in ("422", "409"):
                _MockE5Handler.attempts = 0
                _MockE5Handler.behavior = code
                payload = build_e2_payload("LOTE-2026-11-A", "M-0002", "mal_estado", confidence=0.81)
                res = client.send_e2_ingest(payload)
                self.assertFalse(res["ok"])
                self.assertEqual(res["status_code"], int(code))
                self.assertEqual(res["attempts"], 1)
        finally:
            server.shutdown()
            server.server_close()

    def test_station_listener_sends_e5_with_e1_manifest(self):
        try:
            import numpy as np
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("OpenCV/NumPy no disponibles en este entorno")

        from src.station_listener import StationListener

        server = _start_mock_server()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            client = E5Client(base_url=base_url, timeout_sec=1.0, max_retries=1, backoff_base_sec=0.01)

            mock_predictor = MagicMock()
            mock_predictor.predict_cv2.return_value = (
                {
                    "predicted_class": "buen_estado",
                    "confidence": 0.94,
                    "confidence_percent": 94.0,
                    "probabilities": {"buen_estado": 0.94, "marginalmente_buena": 0.05, "mal_estado": 0.01},
                    "latency_preprocess_ms": 1.0,
                    "latency_inference_ms": 5.0,
                    "latency_ms": 6.0,
                },
                None,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                listener = StationListener(predictor=mock_predictor, shm_dir=tmpdir, e5_client=client)
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                listener.communicator.send_frame_from_station1(
                    frame, id_lote="LOTE-2026-11-A", id_fruta="M-0001"
                )
                result = listener.process_pending_event()

                self.assertIsNotNone(result)
                self.assertEqual(result["id_lote"], "LOTE-2026-11-A")
                self.assertEqual(result["id_fruta"], "M-0001")
                self.assertEqual(result["calidad_visual"], "alta")
                self.assertTrue(result["e5_sync"]["ok"])
                self.assertEqual(len(_MockE5Handler.received_bodies), 1)
                sent = json.loads(_MockE5Handler.received_bodies[0])
                self.assertEqual(sent["estacion"], "E2")
                self.assertEqual(sent["id_lote"], "LOTE-2026-11-A")
                self.assertEqual(sent["id_fruta"], "M-0001")
                self.assertEqual(sent["datos"]["calidad_visual"], "alta")
        finally:
            server.shutdown()
            server.server_close()

    def test_orange_two_photos_sends_e5_with_manifest(self):
        try:
            import numpy as np
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("OpenCV/NumPy no disponibles en este entorno")

        from src.inference import DEFAULT_CLASSES, QualityPredictor
        from src.station_listener import StationListener

        predictor = QualityPredictor.__new__(QualityPredictor)
        predictor.class_names = DEFAULT_CLASSES
        predictor._infer_probs = lambda frame: (np.asarray([0.8, 0.15, 0.05], np.float32), 1.0, 2.0)

        server = _start_mock_server()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            client = E5Client(base_url=base_url, timeout_sec=1.0, max_retries=1, backoff_base_sec=0.01)
            with tempfile.TemporaryDirectory() as tmpdir:
                listener = StationListener(predictor=predictor, shm_dir=tmpdir, e5_client=client)
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                listener.communicator.send_orange_photos_from_station1(
                    "7712323", 1, [frame, frame], id_lote="LOTE-2026-11-A", id_fruta="M-0001"
                )
                results = listener.process_pending_oranges()

                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["id_fruta"], "M-0001")
                self.assertEqual(results[0]["calidad_visual"], "alta")
                self.assertTrue(results[0]["e5_sync"]["ok"])
                self.assertEqual(list(listener.communicator.inbox_dir.iterdir()), [])
                sent = json.loads(_MockE5Handler.received_bodies[0])
                self.assertEqual(sent["id_lote"], "LOTE-2026-11-A")
                self.assertEqual(sent["datos"]["confianza_visual"], 0.8)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()

